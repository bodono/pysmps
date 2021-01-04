from pysmps import smps_loader as mps
import numpy as np
import scs
import scipy.sparse
import osqp
import matplotlib.pyplot as plt

lp = 'lps/afiro'
print("---------------------------------------")
print(lp)
data = mps.load_mps(f'{lp}.mps')
#plt.spy(data['A'])
#plt.show()

import pdb; pdb.set_trace()

