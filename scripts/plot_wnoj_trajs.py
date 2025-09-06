import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

def plot_coordinate_axes(ax, t, R):
  ax.quiver(t.item(0), t.item(1), t.item(2), R.item(0,0), R.item(1,0), R.item(2,0), length=0.5, color='r')
  ax.quiver(t.item(0), t.item(1), t.item(2), R.item(0,1), R.item(1,1), R.item(2,1), length=0.5, color='g')
  ax.quiver(t.item(0), t.item(1), t.item(2), R.item(0,2), R.item(1,2), R.item(2,2), length=0.5, color='b')

def main():
  save_figs=False

  plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.serif": "Times",
    'text.latex.preamble': r'\usepackage{amsfonts, siunitx}'
  })
  directory = "../data/sim/wnoj/"
  save_figs = True

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.view_init(elev=200, azim=45)
  plot_coordinate_axes(ax, np.zeros((3,1)), np.eye(3))
  ax.set_xlabel(r'x (\si{m})')
  ax.set_ylabel(r'y (\si{m})')
  ax.set_zlabel(r'z (\si{m})')
  ax.set_xlim([-4, 4])
  ax.set_ylim([-4, 4])
  ax.set_zlim([-4, 1])
  ax.set_box_aspect((1, 1, 0.5))

  for i in range(15):
    data = np.loadtxt(directory + "traj" + str(i) + ".txt", dtype=float)

    t = data[:,0]
    x = data[:,1]
    y = data[:,2]
    z = data[:,3]
    r0 = data[:,4]
    r1 = data[:,5]
    r2 = data[:,6]

    ax.plot(x, y, z, color="tab:blue")
    for i in range(len(t)//500)[9:]:
      t = np.array([x[i*500], y[i*500], z[i*500]])
      r = np.array([r0[i*500], r1[i*500], r2[i*500]])
      R = Rotation.from_rotvec(r).as_matrix().T
      plot_coordinate_axes(ax, t, R)

  if save_figs:
    plt.savefig("wnoj_trajs.png", dpi=300)#, bbox_inches="tight")

  plt.show()
    



if __name__ == "__main__":
  main()