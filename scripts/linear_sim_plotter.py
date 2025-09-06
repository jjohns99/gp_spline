import numpy as np
import matplotlib.pyplot as plt

def main():
  save_figs = True

  plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.serif": "Times"
  })


  directory = "../data/linear_sim/"

  truth_data = np.loadtxt(directory + "truth.txt", dtype=float)

  t_truth = truth_data[:,0]
  x_truth = truth_data[:,1]
  y_truth = truth_data[:,2]
  vx_truth = truth_data[:,3]
  vy_truth = truth_data[:,4]
  ax_truth = truth_data[:,5]
  ay_truth = truth_data[:,6]

  gp_opt_data = np.loadtxt(directory + "gp_opt.txt", dtype=float)

  t_gp_opt = gp_opt_data[:,0]
  x_gp_opt = gp_opt_data[:,1]
  y_gp_opt = gp_opt_data[:,2]
  vx_gp_opt = gp_opt_data[:,3]
  vy_gp_opt = gp_opt_data[:,4]
  ax_gp_opt = gp_opt_data[:,5]
  ay_gp_opt = gp_opt_data[:,6]

  # differentiate these for fun
  dt = t_gp_opt[1] - t_gp_opt[0]
  jx_gp_opt = (ax_gp_opt[1:] - ax_gp_opt[:-1])/dt
  jy_gp_opt = (ay_gp_opt[1:] - ay_gp_opt[:-1])/dt
  sx_gp_opt = (jx_gp_opt[1:] - jx_gp_opt[:-1])/dt
  sy_gp_opt = (jy_gp_opt[1:] - jy_gp_opt[:-1])/dt

  spl_opt_data = np.loadtxt(directory + "spl_opt.txt", dtype=float)

  t_spl_opt = spl_opt_data[:,0]
  x_spl_opt = spl_opt_data[:,1]
  y_spl_opt = spl_opt_data[:,2]
  vx_spl_opt = spl_opt_data[:,3]
  vy_spl_opt = spl_opt_data[:,4]
  ax_spl_opt = spl_opt_data[:,5]
  ay_spl_opt = spl_opt_data[:,6]

  fig, ax = plt.subplots(6,1)
  ax[0].plot(t_truth, x_truth)
  ax[1].plot(t_truth, y_truth)
  ax[2].plot(t_truth, vx_truth)
  ax[3].plot(t_truth, vy_truth)
  ax[4].plot(t_truth, ax_truth)
  ax[5].plot(t_truth, ay_truth)
  ax[0].plot(t_gp_opt, x_gp_opt)
  ax[1].plot(t_gp_opt, y_gp_opt)
  ax[2].plot(t_gp_opt, vx_gp_opt)
  ax[3].plot(t_gp_opt, vy_gp_opt)
  ax[4].plot(t_gp_opt, ax_gp_opt)
  ax[5].plot(t_gp_opt, ay_gp_opt)
  ax[0].plot(t_spl_opt, x_spl_opt)
  ax[1].plot(t_spl_opt, y_spl_opt)
  ax[2].plot(t_spl_opt, vx_spl_opt)
  ax[3].plot(t_spl_opt, vy_spl_opt)
  ax[4].plot(t_spl_opt, ax_spl_opt)
  ax[5].plot(t_spl_opt, ay_spl_opt)
  ax[5].legend(["true", "gp", "spline"])
  fig.suptitle("Trajectory")

  fig, ax = plt.subplots(4,1)
  ax[0].plot(t_gp_opt[1:], jx_gp_opt)
  ax[1].plot(t_gp_opt[1:], jy_gp_opt)
  ax[2].plot(t_gp_opt[2:], sx_gp_opt)
  ax[3].plot(t_gp_opt[2:], sy_gp_opt)

  plt.figure()
  plt.plot(x_truth, y_truth)
  plt.plot(x_gp_opt, y_gp_opt)
  plt.plot(x_spl_opt, y_spl_opt)
  plt.legend(["true", "gp", "spline"])
  plt.title("xy Trajectory")

  # this figure is to show what an overfitted solution looks like.
  # run gp with priors and spline without priors
  plt.figure(figsize=(6,2))
  plt.plot(x_truth, y_truth)
  plt.plot(x_spl_opt, y_spl_opt)
  plt.plot(x_gp_opt, y_gp_opt)
  plt.legend(["Truth", "Est., No MP", "Est., MP"])
  plt.xlabel("x (m)")
  plt.ylabel("y (m)")
  plt.title("Trajectory Estimate w/ and w/o Motion Priors")

  if save_figs:
    plt.savefig("traj_comp_prior.png", dpi=300, bbox_inches="tight")

  plt.show()


if __name__ == "__main__":
  main()