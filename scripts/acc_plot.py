import numpy as np
import matplotlib.pyplot as plt

def main():
  directory = "../data/sim/imu/"
  save_figs = True

  plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.serif": "Times",
    'text.latex.preamble': r'\usepackage{amsfonts, siunitx}'
  })


  spl_4_true_data = np.loadtxt(directory + "spl_4_true.txt", dtype=float)

  t_spl_4_true = spl_4_true_data[:,0]
  gx_spl_4_true = spl_4_true_data[:,1]
  gy_spl_4_true = spl_4_true_data[:,2]
  gz_spl_4_true = spl_4_true_data[:,3]
  ax_spl_4_true = spl_4_true_data[:,4]
  ay_spl_4_true = spl_4_true_data[:,5]
  az_spl_4_true = spl_4_true_data[:,6]

  spl_4_est_data = np.loadtxt(directory + "spl_4_est.txt", dtype=float)

  t_spl_4_est = spl_4_est_data[:,0][::10]
  gx_spl_4_est = spl_4_est_data[:,1][::10]
  gy_spl_4_est = spl_4_est_data[:,2][::10]
  gz_spl_4_est = spl_4_est_data[:,3][::10]
  ax_spl_4_est = spl_4_est_data[:,4][::10]
  ay_spl_4_est = spl_4_est_data[:,5][::10]
  az_spl_4_est = spl_4_est_data[:,6][::10]

  spl_6_true_data = np.loadtxt(directory + "spl_6_true.txt", dtype=float)

  t_spl_6_true = spl_6_true_data[:,0]
  gx_spl_6_true = spl_6_true_data[:,1]
  gy_spl_6_true = spl_6_true_data[:,2]
  gz_spl_6_true = spl_6_true_data[:,3]
  ax_spl_6_true = spl_6_true_data[:,4]
  ay_spl_6_true = spl_6_true_data[:,5]
  az_spl_6_true = spl_6_true_data[:,6]

  spl_6_est_data = np.loadtxt(directory + "spl_6_est.txt", dtype=float)

  t_spl_6_est = spl_6_est_data[:,0][::10]
  gx_spl_6_est = spl_6_est_data[:,1][::10]
  gy_spl_6_est = spl_6_est_data[:,2][::10]
  gz_spl_6_est = spl_6_est_data[:,3][::10]
  ax_spl_6_est = spl_6_est_data[:,4][::10]
  ay_spl_6_est = spl_6_est_data[:,5][::10]
  az_spl_6_est = spl_6_est_data[:,6][::10]

  gp_true_data = np.loadtxt(directory + "gp_true.txt", dtype=float)

  t_gp_true = gp_true_data[:,0]
  gx_gp_true = gp_true_data[:,1]
  gy_gp_true = gp_true_data[:,2]
  gz_gp_true = gp_true_data[:,3]
  ax_gp_true = gp_true_data[:,4]
  ay_gp_true = gp_true_data[:,5]
  az_gp_true = gp_true_data[:,6]

  gp_est_data = np.loadtxt(directory + "gp_est.txt", dtype=float)

  t_gp_est = gp_est_data[:,0][::10]
  gx_gp_est = gp_est_data[:,1][::10]
  gy_gp_est = gp_est_data[:,2][::10]
  gz_gp_est = gp_est_data[:,3][::10]
  ax_gp_est = gp_est_data[:,4][::10]
  ay_gp_est = gp_est_data[:,5][::10]
  az_gp_est = gp_est_data[:,6][::10]

  # plot post-fit accelerometer data
  fig, ax = plt.subplots(1, 3, figsize=(6,2))
  ax[0].plot(t_spl_4_est[:100], az_spl_4_est[:100], linestyle='solid', color="tab:orange")
  ax[0].plot(t_spl_4_true[:100], az_spl_4_true[:100], linestyle=(0, (1, 1)), color="tab:blue")
  ax[1].plot(t_spl_6_est[:100], az_spl_6_est[:100], linestyle='solid', color="tab:orange")
  ax[1].plot(t_spl_6_true[:100], az_spl_6_true[:100], linestyle=(0, (1, 1)), color="tab:blue")
  l_est = ax[2].plot(t_gp_est[:100], az_gp_est[:100], linestyle='solid', color="tab:orange")
  l_true = ax[2].plot(t_gp_true[:100], az_gp_true[:100], linestyle=(0, (1, 1)), color="tab:blue")
  # fig, ax = plt.subplots(1, 3, figsize=(6,2))
  ax_twin = [a.twinx() for a in ax]
  ax_twin[0].plot(t_spl_4_est[:100], az_spl_4_est[:100] - az_spl_4_true[:100], linestyle='solid', color="tab:green")
  ax_twin[1].plot(t_spl_6_est[:100], az_spl_6_est[:100] - az_spl_6_true[:100], linestyle='solid', color="tab:green")
  l_err = ax_twin[2].plot(t_gp_est[:100], az_gp_est[:100] - az_gp_true[:100], linestyle='solid', color="tab:green")

  ax[0].set_ylim([-20, 1])
  ax[1].set_ylim([-20, 1])
  ax[2].set_ylim([-20, 1])
  ax_twin[0].set_ylim([-0.4, 0.4])
  ax_twin[1].set_ylim([-0.4, 0.4])
  ax_twin[2].set_ylim([-0.4, 0.4])
  ax[0].set_xlabel(r't (\si{s})')
  ax[1].set_xlabel(r't (\si{s})')
  ax[2].set_xlabel(r't (\si{s})')
  ax[0].set_ylabel(r'Acc. (\si{m/s^2})')
  ax[2].legend(l_est+l_true+l_err, [r'est', r'truth', r'error'], loc="upper left", fontsize=8)
  ax[1].set_yticks([])
  ax[2].set_yticks([])
  ax_twin[0].set_yticks([])
  ax_twin[1].set_yticks([])
  ax_twin[2].tick_params(axis='y', labelcolor="tab:green")
  ax_twin[2].set_ylabel(r'err. (\si{m/s^2})', color="tab:green")
  ax[0].set_title(r'Spline ($k$=4)')
  ax[1].set_title(r'Spline ($k$=6)')
  ax[2].set_title(r'GP')
  fig.tight_layout()

  if save_figs:
    plt.savefig("post_fit_acc.png", dpi=300, bbox_inches="tight")

  plt.show()

if __name__ == "__main__":
  main()