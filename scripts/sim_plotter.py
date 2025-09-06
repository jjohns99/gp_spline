import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

def plot_coordinate_axes(ax, t, R):
  ax.quiver(t.item(0), t.item(1), t.item(2), R.item(0,0), R.item(1,0), R.item(2,0), length=0.3, color='r')
  ax.quiver(t.item(0), t.item(1), t.item(2), R.item(0,1), R.item(1,1), R.item(2,1), length=0.3, color='g')
  ax.quiver(t.item(0), t.item(1), t.item(2), R.item(0,2), R.item(1,2), R.item(2,2), length=0.3, color='b')

def main():
  directory = "../data/sim/"
  save_figs = True

  plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.serif": "Times",
    'text.latex.preamble': r'\usepackage{amsfonts, siunitx}'
  })

  truth_data = np.loadtxt(directory + "truth.txt", dtype=float)

  t = truth_data[:,0]
  x = truth_data[:,1]
  y = truth_data[:,2]
  z = truth_data[:,3]
  r0 = truth_data[:,4]
  r1 = truth_data[:,5]
  r2 = truth_data[:,6]
  v0 = truth_data[:,7]
  v1 = truth_data[:,8]
  v2 = truth_data[:,9]
  om0 = truth_data[:,10]
  om1 = truth_data[:,11]
  om2 = truth_data[:,12]
  gb0 = truth_data[:,13]
  gb1 = truth_data[:,14]
  gb2 = truth_data[:,15]
  ab0 = truth_data[:,16]
  ab1 = truth_data[:,17]
  ab2 = truth_data[:,18]

  opt_data = np.loadtxt(directory + "opt.txt", dtype=float)

  t_opt = opt_data[:,0]
  x_opt = opt_data[:,1]
  y_opt = opt_data[:,2]
  z_opt = opt_data[:,3]
  r0_opt = opt_data[:,4]
  r1_opt = opt_data[:,5]
  r2_opt = opt_data[:,6]
  v0_opt = opt_data[:,7]
  v1_opt = opt_data[:,8]
  v2_opt = opt_data[:,9]
  om0_opt = opt_data[:,10]
  om1_opt = opt_data[:,11]
  om2_opt = opt_data[:,12]
  gb0_opt = opt_data[:,13]
  gb1_opt = opt_data[:,14]
  gb2_opt = opt_data[:,15]
  ab0_opt = opt_data[:,16]
  ab1_opt = opt_data[:,17]
  ab2_opt = opt_data[:,18]

  imu_data = np.loadtxt(directory + "imu.txt", dtype=float)

  t_imu = imu_data[:,0]
  gx_imu = imu_data[:,1]
  gy_imu = imu_data[:,2]
  gz_imu = imu_data[:,3]
  ax_imu = imu_data[:,4]
  ay_imu = imu_data[:,5]
  az_imu = imu_data[:,6]

  est_imu_data = np.loadtxt(directory + "est_imu.txt", dtype=float)

  t_est_imu = est_imu_data[:,0]
  gx_est_imu = est_imu_data[:,1]
  gy_est_imu = est_imu_data[:,2]
  gz_est_imu = est_imu_data[:,3]
  ax_est_imu = est_imu_data[:,4]
  ay_est_imu = est_imu_data[:,5]
  az_est_imu = est_imu_data[:,6]

  fig, ax = plt.subplots(6,1)
  ax[0].plot(t, x)
  ax[1].plot(t, y)
  ax[2].plot(t, z)
  ax[3].plot(t, r0)
  ax[4].plot(t, r1)
  ax[5].plot(t, r2)
  ax[0].plot(t_opt, x_opt)
  ax[1].plot(t_opt, y_opt)
  ax[2].plot(t_opt, z_opt)
  ax[3].plot(t_opt, r0_opt)
  ax[4].plot(t_opt, r1_opt)
  ax[5].plot(t_opt, r2_opt)
  ax[0].set_title("D0")

  fig, ax = plt.subplots(6,1)
  ax[0].plot(t, v0)
  ax[1].plot(t, v1)
  ax[2].plot(t, v2)
  ax[3].plot(t, om0)
  ax[4].plot(t, om1)
  ax[5].plot(t, om2)
  ax[0].plot(t_opt, v0_opt)
  ax[1].plot(t_opt, v1_opt)
  ax[2].plot(t_opt, v2_opt)
  ax[3].plot(t_opt, om0_opt)
  ax[4].plot(t_opt, om1_opt)
  ax[5].plot(t_opt, om2_opt)
  ax[0].set_title("D1")

  fig, ax = plt.subplots(6,1)
  ax[0].plot(t, gb0)
  ax[1].plot(t, gb1)
  ax[2].plot(t, gb2)
  ax[3].plot(t, ab0)
  ax[4].plot(t, ab1)
  ax[5].plot(t, ab2)
  ax[0].plot(t_opt, gb0_opt)
  ax[1].plot(t_opt, gb1_opt)
  ax[2].plot(t_opt, gb2_opt)
  ax[3].plot(t_opt, ab0_opt)
  ax[4].plot(t_opt, ab1_opt)
  ax[5].plot(t_opt, ab2_opt)
  ax[0].set_title("imu biases")

  fig, ax = plt.subplots(6,1)
  ax[0].plot(t_imu, gx_imu)
  ax[1].plot(t_imu, gy_imu)
  ax[2].plot(t_imu, gz_imu)
  ax[3].plot(t_imu, ax_imu)
  ax[4].plot(t_imu, ay_imu)
  ax[5].plot(t_imu, az_imu)
  ax[0].plot(t_est_imu, gx_est_imu)
  ax[1].plot(t_est_imu, gy_est_imu)
  ax[2].plot(t_est_imu, gz_est_imu)
  ax[3].plot(t_est_imu, ax_est_imu)
  ax[4].plot(t_est_imu, ay_est_imu)
  ax[5].plot(t_est_imu, az_est_imu)
  ax[5].legend(["truth", "est"])
  fig.suptitle("IMU")

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot(x, y, z)
  # plot_coordinate_axes(ax, np.zeros((3,1)), np.eye(3))
  # for i in range(len(t)//500):
  #   t = np.array([x[i*500], y[i*500], z[i*500]])
  #   r = np.array([r0[i*500], r1[i*500], r2[i*500]])
  #   R = Rotation.from_rotvec(r).as_matrix().T
  #   plot_coordinate_axes(ax, t, R)

  ax.plot(x_opt, y_opt, z_opt)

  ax.set_xlim([-0.5, 0.5])
  ax.set_ylim([-0.5, 0.5])
  ax.set_zlim([-2.35, -1.65])
  # ax.set_xlim([-1.0, 1.0])
  # ax.set_ylim([-1.0, 1.0])
  # ax.set_zlim([0.0, 2.0])
  ax.set_xlabel(r'x (\si{m})')
  ax.set_ylabel(r'y (\si{m})')
  ax.set_zlabel(r'z (\si{m})')
  ax.view_init(elev=200, azim=60)
  ax.set_box_aspect((1, 1, 0.5))
  ax.legend([r'truth', r'est'], loc="center left")

  # if save_figs:
  #   plt.savefig("sin_gp_imump.png", dpi=300)#, bbox_inches="tight")

  plt.show()

if __name__ == "__main__":
  main()