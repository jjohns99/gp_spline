import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def main():
  save_figs = True
  include_mps = True
  end_time = 2.0

  meas_dt = 0.1
  mp_type = 3 # 2 for acc, 3 for jerk

  spline_dt = 0.1
  spline_k = 6
  mp_dt = mp_type*spline_dt
  num_cps = int(np.ceil(end_time/spline_dt)) + spline_k

  meas_times = [i*meas_dt for i in range(int(end_time/meas_dt)+1)]
  mp_times = [i*mp_dt for i in range(int(end_time/mp_dt)+1)]
  if mp_times[-1] < end_time-1e-6:
    mp_times.append(end_time)

  def get_spline_i(t):
    return int(np.floor(t/spline_dt))

  n_gp = 3
  num_gp = int(np.floor(len(meas_times)/n_gp))+1

  def get_gp_i(t):
    return int(np.floor(t/(n_gp*meas_dt)))

  J_spline = np.zeros((len(mp_times)-1+len(meas_times), num_cps), dtype=bool)
  J_gp = np.zeros((num_gp-1 + len(meas_times), num_gp), dtype=bool)

  if include_mps:
    for im in range(len(mp_times[:-1])):
      i1_spline = get_spline_i(mp_times[im])
      i2_spline = get_spline_i(mp_times[im+1])
      J_spline[im, i1_spline:i1_spline+spline_k] = 1
      J_spline[im, i2_spline:i2_spline+spline_k] = 1

    for im in range(num_gp-1):
      J_gp[im, im:im+2] = 1

  for im, meas_t in enumerate(meas_times):
    i_spline = get_spline_i(meas_t)
    J_spline[im+len(mp_times)-1, i_spline:i_spline+spline_k] = 1

    if meas_t > num_gp * meas_dt * n_gp:
      continue

    if im % n_gp == 0:
      J_gp[num_gp-1+im, int(im/n_gp)] = 1
    else:
      i_gp = get_gp_i(meas_t)
      J_gp[num_gp-1+im, i_gp:i_gp+2] = 1

  # plt.figure()
  # plt.pcolormesh(J_spline, cmap=mpl.colormaps["Greys"], edgecolors='k', linewidth=0.1)
  # ax = plt.gca()
  # ax.set_aspect('equal')
  # ax.invert_yaxis()
  # ax.set_xticks([])
  # ax.set_yticks([])

  # plt.figure()
  # plt.pcolormesh(J_gp, cmap=mpl.colormaps["Greys"], edgecolors='k', linewidth=0.1)
  # ax = plt.gca()
  # ax.set_aspect('equal')
  # ax.invert_yaxis()
  # ax.set_xticks([])
  # ax.set_yticks([])

  plt.figure()
  plt.pcolormesh(J_spline.T @ J_spline, cmap=mpl.colormaps["Greys"], edgecolors='k', linewidth=0.1)
  ax = plt.gca()
  ax.set_aspect('equal')
  ax.invert_yaxis()
  ax.set_xticks([])
  ax.set_yticks([])
  
  if save_figs:
    plt.savefig("spline_" + str(spline_k) + "_info_mat.png", dpi=300, bbox_inches="tight")

  plt.figure()
  plt.pcolormesh(J_gp.T @ J_gp, cmap=mpl.colormaps["Greys"], edgecolors='k', linewidth=0.1)
  ax = plt.gca()
  ax.set_aspect('equal')
  ax.invert_yaxis()
  ax.set_xticks([])
  ax.set_yticks([])

  if save_figs:
    plt.savefig("gp_info_mat.png", dpi=300, bbox_inches="tight")

  plt.show()


if __name__ == "__main__":
  main()