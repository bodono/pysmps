from pysmps import smps_loader as mps
import numpy as np
import scs
import scipy.sparse

data = mps.load_mps('d6cube.mps')

if len(data["rhs_names"]) > 1:
  raise ValueError("more than one rhs")
if len(data["bnd_names"]) > 1:
  raise ValueError("more than one bnd")

A_mps = data["A"]
c = data["c"]
b_mps = data["rhs"][data["rhs_names"][0]]
types = np.array(data["types"])
bounds = data["bnd"][data["bnd_names"][0]]

(_, n) = A_mps.shape

A_l = A_mps[types == "G",:]
A_u = A_mps[types == "L",:]

# check if A_l and A_u are equal
if len((A_l != A_u).data) == 0:
  A_box = -A_u
  l = b_mps[types == "G"]
  u = b_mps[types == "L"]
else:
  A_box = -scipy.sparse.vstack((A_l, A_u))
  l = np.hstack((b_mps[types == "G"], -np.inf*np.ones(sum(types == "L"))))
  u = np.hstack((np.inf*np.ones(sum(types == "G")), b_mps[types == "L"]))

vl = bounds['LO']
vu = bounds['UP']

u_idxs = np.where(~np.isinf(vu))[0]
l_idxs = np.where(~np.isinf(-vl))[0]
idxs = np.hstack((l_idxs, u_idxs))
idxs = np.unique(np.sort(idxs))

A_box = scipy.sparse.vstack((-scipy.sparse.eye(n, format='dok')[idxs, :], 
                            A_box))
l = np.hstack((vl[idxs], l))
u = np.hstack((vu[idxs], u))

(box_len, _) = A_box.shape
A_scs = scipy.sparse.vstack((A_mps[types == "E", :], 
                            scipy.sparse.coo_matrix((1,n), dtype=np.float64),
                            A_box))
b_scs = np.hstack((b_mps[types == "E"], 1, np.zeros(box_len,)))
probdata = dict(A=A_scs.tocsc(), b=b_scs, c=c)
cone = dict(f=int(sum(types=='E')), bu=u.tolist(), bl=l.tolist())
results = scs.solve(probdata, cone, max_iters=int(1e8), 
                    adaptive_scaling=False,scale=1., acceleration_lookback=0)
