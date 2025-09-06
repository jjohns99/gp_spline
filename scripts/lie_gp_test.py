import numpy as np
import matplotlib.pyplot as plt

def main():
  est_data = np.loadtxt("../data/test/est.txt", dtype=float)
  jerk = (est_data.shape[1] == 4)

  t_est = est_data[:,0]
  r_est = est_data[:,1]
  v_est = est_data[:,2]
  if jerk:
    a_est = est_data[:,3]

  interp_data = np.loadtxt("../data/test/interp.txt", dtype=float)

  t = interp_data[:,0]
  r = interp_data[:,1]
  v = interp_data[:,2]
  if jerk:
    a = interp_data[:,3]

  fig, ax = plt.subplots(est_data.shape[1]-1,1)
  ax[0].plot(t, r)
  ax[0].scatter(t_est, r_est)
  ax[1].plot(t, v)
  ax[1].scatter(t_est, v_est)
  if jerk:
    ax[2].plot(t, a)
    ax[2].scatter(t_est, a_est)

  plt.show()

  

if __name__ == "__main__":
  main()