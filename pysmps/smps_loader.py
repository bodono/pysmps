# -*- coding: utf-8 -*-
"""Created on Sun Sep  8 13:28:53 2019.

@author: Julian Märte

Updated by: Brendan O'Dongohue, bodonoghue85@gmail.com, Oct 14th 2020
"""

import re
import numpy as np
import scipy.sparse

CORE_FILE_ROW_MODE = 'ROWS'
CORE_FILE_COL_MODE = 'COLUMNS'
CORE_FILE_RHS_MODE = 'RHS'
CORE_FILE_BOUNDS_MODE = 'BOUNDS'

CORE_FILE_BOUNDS_MODE_NAME_GIVEN = 'BOUNDS_NAME'
CORE_FILE_BOUNDS_MODE_NO_NAME = 'BOUNDS_NO_NAME'
CORE_FILE_RHS_MODE_NAME_GIVEN = 'RHS_NAME'
CORE_FILE_RHS_MODE_NO_NAME = 'RHS_NO_NAME'

ROW_MODE_OBJ = 'N'


def load_mps(path):
    mode = ''
    name = None
    objective_name = None
    row_names = []
    types = []
    col_names = []
    col_types = []
    A = scipy.sparse.dok_matrix((0, 0), dtype=np.float64)
    c = np.array([])
    rhs_names = []
    rhs = {}
    bnd_names = []
    bnd = {}
    integral_marker = False

    with open(path, 'r') as reader:
        for line in reader:
            line = re.split(' |\t', line)
            line = [x.strip() for x in line]
            line = list(filter(None, line))

            if line[0] == 'ENDATA':
                break
            if line[0] == '*':
                continue
            if line[0] == 'NAME':
                name = line[1]
            elif line[0] in [CORE_FILE_ROW_MODE, CORE_FILE_COL_MODE]:
                mode = line[0]
            elif line[0] == CORE_FILE_RHS_MODE and len(line) <= 2:
                if len(line) > 1:
                    rhs_names.append(line[1])
                    rhs[line[1]] = np.zeros(len(row_names))
                    mode = CORE_FILE_RHS_MODE_NAME_GIVEN
                else:
                    mode = CORE_FILE_RHS_MODE_NO_NAME
            elif line[0] == CORE_FILE_BOUNDS_MODE and len(line) <= 2:
                if len(line) > 1:
                    bnd_names.append(line[1])
                    bnd[line[1]] = {'LO': np.zeros(
                        len(col_names)), 'UP': np.repeat(np.inf, len(col_names))}
                    mode = CORE_FILE_BOUNDS_MODE_NAME_GIVEN
                else:
                    mode = CORE_FILE_BOUNDS_MODE_NO_NAME
            elif mode == CORE_FILE_ROW_MODE:
                if line[0] == ROW_MODE_OBJ:
                    objective_name = line[1]
                else:
                    types.append(line[0])
                    row_names.append(line[1])
            elif mode == CORE_FILE_COL_MODE:
                if len(line) > 1 and line[1] == "'MARKER'":
                    if line[2] == "'INTORG'":
                        integral_marker = True
                    elif line[2] == "'INTEND'":
                        integral_marker = False
                    continue
                try:
                    i = col_names.index(line[0])
                except:
                    if A.shape[1] == 0:
                        A = scipy.sparse.dok_matrix(
                            (len(row_names), 1), dtype=np.float64)
                    else:
                        new_col = scipy.sparse.dok_matrix(
                            (len(row_names), 1), dtype=np.float64)
                        A = scipy.sparse.hstack((A, new_col), format='dok')
                    col_names.append(line[0])
                    col_types.append(integral_marker * 'integral' +
                                     (not integral_marker) * 'continuous')
                    c = np.append(c, 0)
                    i = -1
                j = 1
                while j < len(line) - 1:
                    if line[j] == objective_name:
                        c[i] = float(line[j + 1])
                    else:
                        A[row_names.index(line[j]), i] = float(line[j + 1])
                    j = j + 2
            elif mode == CORE_FILE_RHS_MODE_NAME_GIVEN:
                if line[0] != rhs_names[-1]:
                    raise Exception(
                        'Other RHS name was given even though name was set after RHS tag.')
                for kk in range((len(line) - 1) // 2):
                    idx = kk * 2
                    try:
                      rhs[line[0]][row_names.index(
                          line[idx+1])] = float(line[idx+2])
                    except Exception as e:
                      if objective_name == line[idx+1]:
                        print("MPS read warning: objective appearing in RHS, ignoring")
                      else:
                        raise e
            elif mode == CORE_FILE_RHS_MODE_NO_NAME:
                if len(line) % 2 == 1: # odd: RHS named
                  try:
                      i = rhs_names.index(line[0])
                  except:
                      rhs_names.append(line[0])
                      rhs[line[0]] = np.zeros(len(row_names))
                      i = -1
                  for kk in range((len(line) - 1) // 2):
                      idx = kk * 2
                      try:
                        rhs[line[0]][row_names.index(
                            line[idx+1])] = float(line[idx+2])
                      except Exception as e:
                        if objective_name == line[idx+1]:
                          print("MPS read warning: objective appearing in RHS, ignoring")
                        else:
                          raise e
                else: # even, no RHS name
                  try:
                      i = rhs_names.index("TEMP")
                  except:
                      rhs_names.append("TEMP")
                      rhs["TEMP"] = np.zeros(len(row_names))
                      i = -1
                  for kk in range(len(line) // 2):
                    idx = kk * 2
                    try:
                      rhs["TEMP"][row_names.index(
                        line[idx])] = float(line[idx+1])
                    except Exception as e:
                      if objective_name == line[idx]:
                        print("MPS read warning: objective appearing in RHS, ignoring")
                      else:
                        raise e

            elif mode == CORE_FILE_BOUNDS_MODE_NAME_GIVEN:
                if line[1] != bnd_names[-1]:
                    raise Exception(
                        'Other BOUNDS name was given even though name was set after BOUNDS tag.')
                if line[0] in ['LO', 'UP']:
                    bnd[line[1]][line[0]][col_names.index(
                        line[2])] = float(line[3])
                elif line[0] == 'FX':
                    bnd[line[1]]['LO'][col_names.index(
                        line[2])] = float(line[3])
                    bnd[line[1]]['UP'][col_names.index(
                        line[2])] = float(line[3])
                elif line[0] == 'PL': # free positive (aka default)
                    bnd[line[1]]['LO'][col_names.index(line[2])] = 0
                elif line[0] == 'FR': # free
                    bnd[line[1]]['LO'][col_names.index(line[2])] = -np.inf
                elif line[0] == 'BV': # binary value
                    bnd[line[1]]['LO'][col_names.index(
                        line[2])] = 0.
                    bnd[line[1]]['UP'][col_names.index(
                        line[2])] = 1.

            elif mode == CORE_FILE_BOUNDS_MODE_NO_NAME:
              _bnds = ['FR', 'BV', 'PL']
              if (len(line) % 2 == 0 and line[0] not in _bnds) or (len(line) % 2 == 1 and line[0] in _bnds): # even, bound has name
                  try:
                      i = bnd_names.index(line[1])
                  except:
                      bnd_names.append(line[1])
                      bnd[line[1]] = {'LO': np.zeros(
                          len(col_names)), 'UP': np.repeat(np.inf, len(col_names))}
                      i = -1
                  if line[0] in ['LO', 'UP']:
                      bnd[line[1]][line[0]][col_names.index(
                          line[2])] = float(line[3])
                  elif line[0] == 'FX': # fixed
                      bnd[line[1]]['LO'][col_names.index(
                          line[2])] = float(line[3])
                      bnd[line[1]]['UP'][col_names.index(
                          line[2])] = float(line[3])
                  elif line[0] == 'PL': # free positive (aka default)
                      bnd[line[1]]['LO'][col_names.index(line[2])] = 0
                  elif line[0] == 'FR': # free
                      bnd[line[1]]['LO'][col_names.index(line[2])] = -np.inf
                  elif line[0] == 'BV': # binary value
                      bnd[line[1]]['LO'][col_names.index(
                          line[2])] = 0.
                      bnd[line[1]]['UP'][col_names.index(
                          line[2])] = 1.
              else: # odd, bound has no name
                  try:
                      i = bnd_names.index("TEMP_BOUND")
                  except:
                      bnd_names.append("TEMP_BOUND")
                      bnd["TEMP_BOUND"] = {'LO': np.zeros(
                          len(col_names)), 'UP': np.repeat(np.inf, len(col_names))}
                      i = -1
                  if line[0] in ['LO', 'UP']:
                      bnd["TEMP_BOUND"][line[0]][col_names.index(
                          line[1])] = float(line[2])
                  elif line[0] == 'FX':
                      bnd["TEMP_BOUND"]['LO'][col_names.index(
                          line[1])] = float(line[2])
                      bnd["TEMP_BOUND"]['UP'][col_names.index(
                          line[1])] = float(line[2])
                  elif line[0] == 'FR':
                      bnd["TEMP_BOUND"]['LO'][col_names.index(line[1])] = -np.inf

    return dict(name=name, objective_name=objective_name, row_names=row_names,
                col_names=col_names, col_types=col_types, types=types, c=c, A=A,
                rhs_names=rhs_names, rhs=rhs, bnd_names=bnd_names, bnd=bnd)
