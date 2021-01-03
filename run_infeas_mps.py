from pysmps import smps_loader as mps
import numpy as np
import scs
import scipy.sparse
import osqp

infeasible_lps = sorted([
 "afiro", # not actually infeasible
 "ceria3d",
 "chemcom",
 "cplex1",
 "cplex2",
 "forest6",
 "galenet",
 "greenbea",
 "itest2",
 "itest6",
 "klein1",
 "klein2",
 "klein3",
 "mondou2",
 "qual",
 "reactor",
 "refinery",
 "vol1",
 "woodinfe",
 "bgdbg1",
 "bgetam",
 #"bgindy",
 "bgprtr",
 "box1",
 "ex72a",
 "ex73a",
 "gosh",
 "gran",
 "pang",
 "pilot4i",
])

VERBOSE=True
results_dict = {}

for lp in infeasible_lps:
  print("---------------------------------------")
  print(lp)
  data = mps.load_mps(f'lps/{lp}.mps')

  if len(data["rhs_names"]) > 1:
    raise ValueError("more than one rhs")
  if len(data["bnd_names"]) > 1:
    raise ValueError("more than one bnd")

  A_mps = data["A"]
  c = data["c"]
  (m, n) = A_mps.shape
  types = np.array(data["types"])
  if not data["rhs"]: # if RHS totally missing, assume zeros
    b_mps = np.zeros(m)
  else:
    b_mps = data["rhs"][data["rhs_names"][0]]
  if not data["bnd_names"]: # if BOUNDS totally missing don't set them
    bounds = None
  else:
    bounds = data["bnd"][data["bnd_names"][0]]

  A_l = A_mps[types == "G",:]
  A_u = A_mps[types == "L",:]

  # SCS l <= s <= u, s = -Ax
  # OSQP l <= Ax <= u

  # check if A_l and A_u are equal
  if A_l.shape == A_u.shape and len((A_l != A_u).data) == 0:
    A_box = -A_u
    l = b_mps[types == "G"]
    u = b_mps[types == "L"]
  else:
    A_box = scipy.sparse.vstack((A_l, A_u))
    l = np.hstack((b_mps[types == "G"], -np.inf*np.ones(sum(types == "L"))))
    u = np.hstack((np.inf*np.ones(sum(types == "G")), b_mps[types == "L"]))

  # variable bounds vl <= x <= vu
  if bounds:
    vl = bounds['LO']
    vu = bounds['UP']

    assert np.squeeze(vl).shape[0] == n
    assert np.squeeze(vu).shape[0] == n

    #u_idxs = np.where(~np.isinf(vu))[0]
    #l_idxs = np.where(~np.isinf(-vl))[0]
    #idxs = np.hstack((l_idxs, u_idxs))
    #idxs = np.unique(np.sort(idxs))

  else:
    #idxs = []
    vl = np.zeros(n)
    vu = np.inf * np.ones(n)

  #l = np.hstack((vl[idxs], l))
  #u = np.hstack((vu[idxs], u))
  l = np.hstack((vl, l))
  u = np.hstack((vu, u))

  # SCS box cone format (negate A_box)
  #A_scs = scipy.sparse.vstack((-scipy.sparse.eye(n, format='dok')[idxs, :],
  #                            -A_box))
  A_scs = scipy.sparse.vstack((-scipy.sparse.eye(n, format='dok'), -A_box))
  # OSQP box cone format
  #A_osqp = scipy.sparse.vstack((scipy.sparse.eye(n, format='dok')[idxs, :],
  #                            A_box))
  A_osqp = scipy.sparse.vstack((scipy.sparse.eye(n, format='dok'), A_box))


  (box_len, _) = A_scs.shape
  # SCS stack Ax = b on top, add row for perspective var in box cone
  A_scs = scipy.sparse.vstack((A_mps[types == "E", :],
                              scipy.sparse.coo_matrix((1,n), dtype=np.float64),
                              A_scs))
  # SCS: b = [b; 1; zeros], 1 for box cone perspective
  b_scs = np.hstack((b_mps[types == "E"], 1, np.zeros(box_len,)))

  # OSQP stack Ax = b on top
  A_osqp = scipy.sparse.vstack((A_mps[types == "E", :],
                              A_osqp))
  # OSQP stack equality b on top
  l_osqp = np.hstack((b_mps[types == "E"], l))
  u_osqp = np.hstack((b_mps[types == "E"], u))

  probdata = dict(A=A_scs.tocsc(), b=b_scs, c=c)
  cone = dict(f=int(sum(types=='E')), bu=u.tolist(), bl=l.tolist())
  scs_results = scs.solve(probdata, cone,
                          adaptive_scaling=True, eps_infeas=1e-6,
                          verbose=VERBOSE,
                          scale=0.1, max_iters=int(1e5),eps_rel=1e-6, eps_abs=1e-6,
                          acceleration_lookback=0)
  print(f'SCS status {scs_results["info"]["status"]}')
  print(f'SCS iters {scs_results["info"]["iter"]}')

  oss = osqp.OSQP()
  oss.setup(q=c, A=A_osqp.tocsc(), l=l_osqp, u=u_osqp, eps_rel=1e-6, eps_abs=1e-6,
            eps_prim_inf = 1e-6, eps_dual_inf = 1e-6, verbose=VERBOSE,
            adaptive_rho=True, rho=0.1,
            max_iter=int(1e5))
            #sigma=RHO_X, scaling=NORMALIZE, max_iter=N,
            #adaptive_rho=ADAPTIVE_SCALING, polish=False, eps_prim_inf=eps,
            #eps_dual_inf=EPS, eps_abs=EPS, eps_rel=EPS, alpha=ALPHA,
            #verbose=verbose, rho=SCALE)
  osqp_results = oss.solve()

  x_cert = osqp_results.dual_inf_cert
  y_cert = osqp_results.prim_inf_cert

  import pdb; pdb.set_trace()

  print(f'OSQP status {osqp_results.info.status}')
  print(f'OSQP iters {osqp_results.info.iter}')

  results_dict[lp] = {'scs_iters': scs_results["info"]["iter"],
                      'scs_status': scs_results["info"]["status"],
                      'osqp_iters': osqp_results.info.iter,
                      'osqp_status': osqp_results.info.status}

print(results_dict)
