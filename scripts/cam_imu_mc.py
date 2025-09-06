import numpy as np
import matplotlib.pyplot as plt

def main():
  save_figs=True

  plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.serif": "Times",
    'text.latex.preamble': r'\usepackage{amsfonts, siunitx}'
  })

  # [pos_rmse, rot_rmse, vel_rmse, solve_time]
  wnoj_se3spl_4_01_none = [0.100196, 0.0905851, 2.94838, 0.0199509, 15]
  wnoj_se3spl_4_01_mp = [0.0339818, 0.0307259, 0.137362, 0.0477343, 6]
  wnoj_se3spl_4_01_imu = [0.0188098, 0.0054345, 0.0180536, 0.307358, 12]
  wnoj_se3spl_4_01_imump = [0.0183771, 0.00554596, 0.0185434, 0.331839, 12]

  wnoj_se3spl_6_01_none = [0.0981133, 0.0884054, 2.92711, 0.0413508, 18]
  wnoj_se3spl_6_01_mp = [0.0343344, 0.0319277, 0.211903, 0.966016, 22]
  wnoj_se3spl_6_01_imu = [0.018531, 0.00592885, 0.0184611, 0.613495, 12]
  wnoj_se3spl_6_01_imump = [0.0182368, 0.00595922, 0.0178987, 1.92594, 13]

  wnoj_so3xr3spl_4_01_none = [0.10178, 0.0898367, 3.1284, 0.0188041, 17]
  wnoj_so3xr3spl_4_01_mp = [0.0337965, 0.0313611, 0.142031, 0.0310233, 5]
  wnoj_so3xr3spl_4_01_imu = [0.0183076, 0.00589408, 0.0184509, 0.126285, 12]
  wnoj_so3xr3spl_4_01_imump = [0.0174244, 0.00568281, 0.0171491, 0.151226, 12]

  wnoj_so3xr3spl_6_01_none = [0.104614, 0.0896669, 3.42546, 0.0307169, 21]
  wnoj_so3xr3spl_6_01_mp = [0.0356117, 0.0309746, 0.231794, 0.545935, 21]
  wnoj_so3xr3spl_6_01_imu = [0.0182487, 0.00528055, 0.0172433, 0.192706, 11]
  wnoj_so3xr3spl_6_01_imump = [0.0182728, 0.00560519, 0.0175828, 1.38698, 11]

  wnoj_se3gp_1_none = [0.0779646, 0.0770666, 2.23155, 0.00124502, 4]
  wnoj_se3gp_1_mp = [0.0338459, 0.0307103, 0.138732, 0.021455, 7]
  wnoj_se3gp_1_imu = [0.0176892, 0.00548, 0.0179422, 0.396481, 17]
  wnoj_se3gp_1_imump = [0.0181612, 0.00495725, 0.0171303, 0.425723, 18]

  wnoj_so3xr3gp_1_none = [0.0777097, 0.0774771, 2.20629, 0.00149584, 5]
  wnoj_so3xr3gp_1_mp = [0.0338772, 0.0311728, 0.138915, 0.0142431, 7]
  wnoj_so3xr3gp_1_imu = [0.0181312, 0.00538395, 0.017714, 0.188749, 13]
  wnoj_so3xr3gp_1_imump = [0.0174512, 0.00577307, 0.0181564, 0.19887, 13]

  wnoj_none_pos_rmse = [wnoj_se3spl_4_01_none[0], wnoj_se3spl_6_01_none[0], wnoj_se3gp_1_none[0], wnoj_so3xr3spl_4_01_none[0], wnoj_so3xr3spl_6_01_none[0], wnoj_so3xr3gp_1_none[0]]
  wnoj_mp_pos_rmse = [wnoj_se3spl_4_01_mp[0], wnoj_se3spl_6_01_mp[0], wnoj_se3gp_1_mp[0], wnoj_so3xr3spl_4_01_mp[0], wnoj_so3xr3spl_6_01_mp[0], wnoj_so3xr3gp_1_mp[0]]
  wnoj_imu_pos_rmse = [wnoj_se3spl_4_01_imu[0], wnoj_se3spl_6_01_imu[0], wnoj_se3gp_1_imu[0], wnoj_so3xr3spl_4_01_imu[0], wnoj_so3xr3spl_6_01_imu[0], wnoj_so3xr3gp_1_imu[0]]
  wnoj_imump_pos_rmse = [wnoj_se3spl_4_01_imump[0], wnoj_se3spl_6_01_imump[0], wnoj_se3gp_1_imump[0], wnoj_so3xr3spl_4_01_imump[0], wnoj_so3xr3spl_6_01_imump[0], wnoj_so3xr3gp_1_imump[0]]

  wnoj_none_rot_rmse = [wnoj_se3spl_4_01_none[1], wnoj_se3spl_6_01_none[1], wnoj_se3gp_1_none[1], wnoj_so3xr3spl_4_01_none[1], wnoj_so3xr3spl_6_01_none[1], wnoj_so3xr3gp_1_none[1]]
  wnoj_mp_rot_rmse = [wnoj_se3spl_4_01_mp[1], wnoj_se3spl_6_01_mp[1], wnoj_se3gp_1_mp[1], wnoj_so3xr3spl_4_01_mp[1], wnoj_so3xr3spl_6_01_mp[1], wnoj_so3xr3gp_1_mp[1]]
  wnoj_imu_rot_rmse = [wnoj_se3spl_4_01_imu[1], wnoj_se3spl_6_01_imu[1], wnoj_se3gp_1_imu[1], wnoj_so3xr3spl_4_01_imu[1], wnoj_so3xr3spl_6_01_imu[1], wnoj_so3xr3gp_1_imu[1]]
  wnoj_imump_rot_rmse = [wnoj_se3spl_4_01_imump[1], wnoj_se3spl_6_01_imump[1], wnoj_se3gp_1_imump[1], wnoj_so3xr3spl_4_01_imump[1], wnoj_so3xr3spl_6_01_imump[1], wnoj_so3xr3gp_1_imump[1]]

  wnoj_none_vel_rmse = [wnoj_se3spl_4_01_none[2], wnoj_se3spl_6_01_none[2], wnoj_se3gp_1_none[2], wnoj_so3xr3spl_4_01_none[2], wnoj_so3xr3spl_6_01_none[2], wnoj_so3xr3gp_1_none[2]]
  wnoj_mp_vel_rmse = [wnoj_se3spl_4_01_mp[2], wnoj_se3spl_6_01_mp[2], wnoj_se3gp_1_mp[2], wnoj_so3xr3spl_4_01_mp[2], wnoj_so3xr3spl_6_01_mp[2], wnoj_so3xr3gp_1_mp[2]]
  wnoj_imu_vel_rmse = [wnoj_se3spl_4_01_imu[2], wnoj_se3spl_6_01_imu[2], wnoj_se3gp_1_imu[2], wnoj_so3xr3spl_4_01_imu[2], wnoj_so3xr3spl_6_01_imu[2], wnoj_so3xr3gp_1_imu[2]]
  wnoj_imump_vel_rmse = [wnoj_se3spl_4_01_imump[2], wnoj_se3spl_6_01_imump[2], wnoj_se3gp_1_imump[2], wnoj_so3xr3spl_4_01_imump[2], wnoj_so3xr3spl_6_01_imump[2], wnoj_so3xr3gp_1_imump[2]]

  wnoj_none_time = [wnoj_se3spl_4_01_none[3], wnoj_se3spl_6_01_none[3], wnoj_se3gp_1_none[3], wnoj_so3xr3spl_4_01_none[3], wnoj_so3xr3spl_6_01_none[3], wnoj_so3xr3gp_1_none[3]]
  wnoj_mp_time = [wnoj_se3spl_4_01_mp[3], wnoj_se3spl_6_01_mp[3], wnoj_se3gp_1_mp[3], wnoj_so3xr3spl_4_01_mp[3], wnoj_so3xr3spl_6_01_mp[3], wnoj_so3xr3gp_1_mp[3]]
  wnoj_imu_time = [wnoj_se3spl_4_01_imu[3], wnoj_se3spl_6_01_imu[3], wnoj_se3gp_1_imu[3], wnoj_so3xr3spl_4_01_imu[3], wnoj_so3xr3spl_6_01_imu[3], wnoj_so3xr3gp_1_imu[3]]
  wnoj_imump_time = [wnoj_se3spl_4_01_imump[3], wnoj_se3spl_6_01_imump[3], wnoj_se3gp_1_imump[3], wnoj_so3xr3spl_4_01_imump[3], wnoj_so3xr3spl_6_01_imump[3], wnoj_so3xr3gp_1_imump[3]]



  sin_se3spl_4_01_none = [0.0748979, 0.033971, 1.74099, 0.0378308, 26]
  sin_se3spl_4_01_mp = [0.0461993, 0.0205122, 0.265757, 0.0565851, 5]
  sin_se3spl_4_01_imu = [0.00574301, 0.00273485, 0.00518483, 0.244293, 9]
  sin_se3spl_4_01_imump = [0.00546709, 0.00254205, 0.00502112, 0.266599, 9]

  sin_se3spl_6_01_none = [0.0728478, 0.0340282, 1.6091, 0.080575, 33]
  sin_se3spl_6_01_mp = [0.0465473, 0.0205566, 0.285132, 0.555847, 6]
  sin_se3spl_6_01_imu = [0.00567935, 0.0026868, 0.0043, 0.557328, 10]
  sin_se3spl_6_01_imump = [0.00563718, 0.00267561, 0.00423504, 1.69091, 10]

  sin_so3xr3spl_4_01_none = [0.0745713, 0.0335107, 1.71348, 0.025058, 24]
  sin_so3xr3spl_4_01_mp = [0.0570154, 0.0256376, 0.341097, 0.0428948, 5]
  sin_so3xr3spl_4_01_imu = [0.00860862, 0.00411402, 0.0062596, 0.113587, 9]
  sin_so3xr3spl_4_01_imump = [0.00798991, 0.00374206, 0.00581404, 0.116663, 9]

  sin_so3xr3spl_6_01_none = [0.0687248, 0.0312381, 1.40781, 0.0841861, 59]
  sin_so3xr3spl_6_01_mp = [0.0578044, 0.0258506, 0.359278, 0.449972, 6]
  sin_so3xr3spl_6_01_imu = [0.0057374, 0.00270455, 0.00433273, 0.250053, 10]
  sin_so3xr3spl_6_01_imump = [0.00536795, 0.00249652, 0.00413319, 1.29789, 10]

  sin_se3gp_1_none = [0.0621505, 0.032612, 1.86197, 0.00220203, 7]
  sin_se3gp_1_mp = [0.046244, 0.0204649, 0.262776, 0.023205, 7]
  sin_se3gp_1_imu = [0.00519588, 0.00247552, 0.0054885, 0.427674, 19]
  sin_se3gp_1_imump = [0.005222, 0.00241251, 0.00449982, 0.436628, 18]

  sin_so3xr3gp_1_none = [0.0629485, 0.0328792, 1.8658, 0.00237799, 8]
  sin_so3xr3gp_1_mp = [0.0568158, 0.0256147, 0.342163, 0.0156779, 7]
  sin_so3xr3gp_1_imu = [0.00593642, 0.00284081, 0.00586629, 0.189494, 13]
  sin_so3xr3gp_1_imump = [0.00540335, 0.00257594, 0.00454357, 0.199282, 12]

  sin_none_pos_rmse = [sin_se3spl_4_01_none[0], sin_se3spl_6_01_none[0], sin_se3gp_1_none[0], sin_so3xr3spl_4_01_none[0], sin_so3xr3spl_6_01_none[0], sin_so3xr3gp_1_none[0]]
  sin_mp_pos_rmse = [sin_se3spl_4_01_mp[0], sin_se3spl_6_01_mp[0], sin_se3gp_1_mp[0], sin_so3xr3spl_4_01_mp[0], sin_so3xr3spl_6_01_mp[0], sin_so3xr3gp_1_mp[0]]
  sin_imu_pos_rmse = [sin_se3spl_4_01_imu[0], sin_se3spl_6_01_imu[0], sin_se3gp_1_imu[0], sin_so3xr3spl_4_01_imu[0], sin_so3xr3spl_6_01_imu[0], sin_so3xr3gp_1_imu[0]]
  sin_imump_pos_rmse = [sin_se3spl_4_01_imump[0], sin_se3spl_6_01_imump[0], sin_se3gp_1_imump[0], sin_so3xr3spl_4_01_imump[0], sin_so3xr3spl_6_01_imump[0], sin_so3xr3gp_1_imump[0]]

  sin_none_rot_rmse = [sin_se3spl_4_01_none[1], sin_se3spl_6_01_none[1], sin_se3gp_1_none[1], sin_so3xr3spl_4_01_none[1], sin_so3xr3spl_6_01_none[1], sin_so3xr3gp_1_none[1]]
  sin_mp_rot_rmse = [sin_se3spl_4_01_mp[1], sin_se3spl_6_01_mp[1], sin_se3gp_1_mp[1], sin_so3xr3spl_4_01_mp[1], sin_so3xr3spl_6_01_mp[1], sin_so3xr3gp_1_mp[1]]
  sin_imu_rot_rmse = [sin_se3spl_4_01_imu[1], sin_se3spl_6_01_imu[1], sin_se3gp_1_imu[1], sin_so3xr3spl_4_01_imu[1], sin_so3xr3spl_6_01_imu[1], sin_so3xr3gp_1_imu[1]]
  sin_imump_rot_rmse = [sin_se3spl_4_01_imump[1], sin_se3spl_6_01_imump[1], sin_se3gp_1_imump[1], sin_so3xr3spl_4_01_imump[1], sin_so3xr3spl_6_01_imump[1], sin_so3xr3gp_1_imump[1]]

  sin_none_vel_rmse = [sin_se3spl_4_01_none[2], sin_se3spl_6_01_none[2], sin_se3gp_1_none[2], sin_so3xr3spl_4_01_none[2], sin_so3xr3spl_6_01_none[2], sin_so3xr3gp_1_none[2]]
  sin_mp_vel_rmse = [sin_se3spl_4_01_mp[2], sin_se3spl_6_01_mp[2], sin_se3gp_1_mp[2], sin_so3xr3spl_4_01_mp[2], sin_so3xr3spl_6_01_mp[2], sin_so3xr3gp_1_mp[2]]
  sin_imu_vel_rmse = [sin_se3spl_4_01_imu[2], sin_se3spl_6_01_imu[2], sin_se3gp_1_imu[2], sin_so3xr3spl_4_01_imu[2], sin_so3xr3spl_6_01_imu[2], sin_so3xr3gp_1_imu[2]]
  sin_imump_vel_rmse = [sin_se3spl_4_01_imump[2], sin_se3spl_6_01_imump[2], sin_se3gp_1_imump[2], sin_so3xr3spl_4_01_imump[2], sin_so3xr3spl_6_01_imump[2], sin_so3xr3gp_1_imump[2]]

  sin_none_time = [sin_se3spl_4_01_none[3], sin_se3spl_6_01_none[3], sin_se3gp_1_none[3], sin_so3xr3spl_4_01_none[3], sin_so3xr3spl_6_01_none[3], sin_so3xr3gp_1_none[3]]
  sin_mp_time = [sin_se3spl_4_01_mp[3], sin_se3spl_6_01_mp[3], sin_se3gp_1_mp[3], sin_so3xr3spl_4_01_mp[3], sin_so3xr3spl_6_01_mp[3], sin_so3xr3gp_1_mp[3]]
  sin_imu_time = [sin_se3spl_4_01_imu[3], sin_se3spl_6_01_imu[3], sin_se3gp_1_imu[3], sin_so3xr3spl_4_01_imu[3], sin_so3xr3spl_6_01_imu[3], sin_so3xr3gp_1_imu[3]]
  sin_imump_time = [sin_se3spl_4_01_imump[3], sin_se3spl_6_01_imump[3], sin_se3gp_1_imump[3], sin_so3xr3spl_4_01_imump[3], sin_so3xr3spl_6_01_imump[3], sin_so3xr3gp_1_imump[3]]



  labels = [r'SE(3) spline, $k$ = 4', r'SE(3) spline, $k$ = 6', r'SE(3) GP', r'SO(3)$\times \mathbb{R}^3$ spline, $k$ = 4', r'SO(3)$\times \mathbb{R}^3$ spline, $k$ = 6', r'SO(3)$\times \mathbb{R}^3$ GP']
  appended_labels = labels + ['',''] + labels + ['',''] + labels + ['',''] + labels
  colors = ['red', 'green', 'blue', 'lightcoral', 'lightgreen', 'lightblue']
  appended_colors = colors + ['k','k'] + colors + ['k','k'] + colors + ['k','k'] + colors

  fig, ax = plt.subplots(4,1, figsize=(12,7))
  ax[0].bar(range(30), wnoj_none_pos_rmse + [0.0, 0.0] + wnoj_mp_pos_rmse + [0.0, 0.0] + wnoj_imu_pos_rmse + [0.0, 0.0] + wnoj_imump_pos_rmse, 1.0, color=appended_colors, edgecolor='k', log=True)
  ax[1].bar(range(30), wnoj_none_rot_rmse + [0.0, 0.0] + wnoj_mp_rot_rmse + [0.0, 0.0] + wnoj_imu_rot_rmse + [0.0, 0.0] + wnoj_imump_rot_rmse, 1.0, color=appended_colors, edgecolor='k', log=True)
  ax[2].bar(range(30), wnoj_none_vel_rmse + [0.0, 0.0] + wnoj_mp_vel_rmse + [0.0, 0.0] + wnoj_imu_vel_rmse + [0.0, 0.0] + wnoj_imump_vel_rmse, 1.0, color=appended_colors, edgecolor='k', log=True)
  ax[3].bar(range(30), wnoj_none_time + [0.0, 0.0] + wnoj_mp_time + [0.0, 0.0] + wnoj_imu_time + [0.0, 0.0] + wnoj_imump_time, 1.0, color=appended_colors, edgecolor='k')
  ax[0].set_xticks([])
  ax[1].set_xticks([])
  ax[2].set_xticks([])
  ax[3].set_xticks([2.5, 10.5, 18.5, 26.5])
  ax[3].set_xticklabels([r'None', r'Motion Priors', r'IMU', r'Motion Priors + IMU'])
  ax[0].set_ylabel(r'Pos. RMSE (\si{m})')
  ax[1].set_ylabel(r'Rot. RMSE (\si{rad})')
  ax[2].set_ylabel(r'Twist RMSE')
  ax[3].set_ylabel(r'Solve Time (\si{s})')
  ax[3].set_xlabel(r'Regularization Type', fontsize=12)

  legend_handles = [plt.Rectangle((0,0), 1, 1, facecolor=c, edgecolor='k') for c in colors]
  ax[0].legend(legend_handles, labels, ncol=2)
  ax[0].set_title(r'WNOJ Trajectory')

  if(save_figs):
    plt.savefig("wnoj_mc_cam_imu.png", dpi=300, bbox_inches="tight")

  fig, ax = plt.subplots(4,1, figsize=(12,7))
  ax[0].bar(range(30), sin_none_pos_rmse + [0.0, 0.0] + sin_mp_pos_rmse + [0.0, 0.0] + sin_imu_pos_rmse + [0.0, 0.0] + sin_imump_pos_rmse, 1.0, color=appended_colors, edgecolor='k', log=True)
  ax[1].bar(range(30), sin_none_rot_rmse + [0.0, 0.0] + sin_mp_rot_rmse + [0.0, 0.0] + sin_imu_rot_rmse + [0.0, 0.0] + sin_imump_rot_rmse, 1.0, color=appended_colors, edgecolor='k', log=True)
  ax[2].bar(range(30), sin_none_vel_rmse + [0.0, 0.0] + sin_mp_vel_rmse + [0.0, 0.0] + sin_imu_vel_rmse + [0.0, 0.0] + sin_imump_vel_rmse, 1.0, color=appended_colors, edgecolor='k', log=True)
  ax[3].bar(range(30), sin_none_time + [0.0, 0.0] + sin_mp_time + [0.0, 0.0] + sin_imu_time + [0.0, 0.0] + sin_imump_time, 1.0, color=appended_colors, edgecolor='k')
  ax[0].set_xticks([])
  ax[1].set_xticks([])
  ax[2].set_xticks([])
  ax[3].set_xticks([2.5, 10.5, 18.5, 26.5])
  ax[3].set_xticklabels([r'None', r'Motion Priors', r'IMU', r'Motion Priors + IMU'])
  ax[0].set_ylabel(r'Pos. RMSE (\si{m})')
  ax[1].set_ylabel(r'Rot. RMSE (\si{rad})')
  ax[2].set_ylabel(r'Twist RMSE')
  ax[3].set_ylabel(r'Solve Time (\si{s})')
  ax[3].set_xlabel(r'Regularization Type', fontsize=12)

  legend_handles = [plt.Rectangle((0,0), 1, 1, facecolor=c, edgecolor='k') for c in colors]
  ax[0].legend(legend_handles, labels, ncol=2)
  ax[0].set_title(r'Sinusoidal Trajectory')

  if(save_figs):
    plt.savefig("sin_mc_cam_imu.png", dpi=300, bbox_inches="tight")

  # plot breakdown of solve times for wnoj MP + IMU
  se3spline_4_time_break = [0.28812, 0.0420587]
  se3spline_6_time_break = [1.17225, 0.751106]
  se3gp_time_break = [0.331923, 0.0931702]
  so3xr3spline_4_time_break = [0.105926, 0.0456653]
  so3xr3spline_6_time_break = [0.28494, 1.10492]
  so3xr3gp_time_break = [0.118675, 0.0802379]

  jac_eval_times = [se3spline_4_time_break[0], se3spline_6_time_break[0], se3gp_time_break[0], so3xr3spline_4_time_break[0], so3xr3spline_6_time_break[0], so3xr3gp_time_break[0]]
  lin_solve_times = [se3spline_4_time_break[1], se3spline_6_time_break[1], se3gp_time_break[1], so3xr3spline_4_time_break[1], so3xr3spline_6_time_break[1], so3xr3gp_time_break[1]]

  fig, ax = plt.subplots(1,1, figsize=(6,2.5))
  ax.bar(range(6), jac_eval_times, 1.0, color=colors, edgecolor='k', bottom=[0 for i in range(len(jac_eval_times))])
  ax.bar(range(6), lin_solve_times, 1.0, color=colors, edgecolor='k', hatch="oo", bottom=jac_eval_times)
  # time_legend_handles = legend_handles[:3] + [plt.Rectangle((0,0), 1, 1, facecolor="None", edgecolor='k')] + legend_handles[3:] + [plt.Rectangle((0,0), 1, 1, facecolor="None", hatch="oo", edgecolor='k')]
  # time_labels = labels[:3] + [r'Jacobian and residual evaluation'] + labels[3:] + [r'Linear solver']
  time_legend_handles = [plt.Rectangle((0,0), 1, 1, facecolor="None", edgecolor='k'), plt.Rectangle((0,0), 1, 1, facecolor="None", hatch="oo", edgecolor='k')]
  time_labels = [r'Jacobian and residual evaluation', r'Linear solver']
  ax.set_ylabel(r'Computation Time (\si{s})')
  ax.set_xticklabels([''] + labels, rotation=15, ha='right')
  ax.set_title(r'Breakdown of Solve Time')
  legend = ax.legend(time_legend_handles, time_labels, framealpha=0.4)
  for patch in legend.get_patches():
    patch.set_height(8)

  if(save_figs):
    plt.savefig("solve_time_breakdown.png", dpi=300, bbox_inches="tight")

  # plot post-solve trajectory sampling time
  sampling_times = [
    0.01697,    # SE(3) spline, k=4
    0.05068,    # SE(3) spline, k=6
    0.0582908,  # SE(3) GP
    0.012268,   # SO(3)xR3 spline, k=4
    0.0345834,  # SO(3)xR3 spline, k=6
    0.0303282,  # SO(3)xR3 GP
  ]

  fig, ax = plt.subplots(1,1, figsize=(6,2.5))
  ax.bar(range(6), sampling_times, 1.0, color=colors, edgecolor='k', bottom=[0 for i in range(len(sampling_times))])
  ax.set_ylabel(r'Sampling Time (\si{s})')
  ax.set_xticklabels([''] + labels, rotation=15, ha='right')
  ax.set_title(r'Post-solve Trajectory Sampling Time')

  if(save_figs):
    plt.savefig("sampling_time.png", dpi=300, bbox_inches="tight")

  plt.show()


if __name__ == "__main__":
  main()