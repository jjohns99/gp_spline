import numpy as np
import matplotlib.pyplot as plt
import os

def main():
  save_figs = False

  plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.serif": "Times"
  })

  linestyles = ['-.', '--', ':', (0, (3, 5, 1, 5, 1, 5)), (0, (5, 10)), (0, (3, 5, 1, 5))]
  markers = ['x', '.', '+', '1', '2', '|']

  mc_directory = "../data/linear_sim/mc/"
  mc_sweep_directory = "../data/linear_sim/mc_sweep/"

  file_names = os.listdir(mc_directory)

  # these contain a tuple with (params, data)
  gp_mc_data = []
  spline_mc_data = []

  for file_name in file_names:
    # save gp data
    if file_name[0] == "g":
      gp_params_raw = file_name[:-4].split('_')[1:]
      gp_mc_data.append(({"traj_type": int(gp_params_raw[0]), 
                          "model_type": int(gp_params_raw[1]), 
                          "n": int(gp_params_raw[2]),
                          "mp_used": bool(int(gp_params_raw[3]))},
                         np.loadtxt(mc_directory + file_name, dtype=float)))

    # save spline data
    elif file_name[0] == "s":
      spline_params_raw = file_name[:-4].split('_')[1:]
      spline_mc_data.append(({"traj_type": int(spline_params_raw[0]), 
                              "model_type": int(spline_params_raw[1]), 
                              "k": int(spline_params_raw[2]),
                              "dt": float(spline_params_raw[3]),
                              "mp_used": bool(int(spline_params_raw[4])),
                              "mp_dt": float(spline_params_raw[5])},
                             np.loadtxt(mc_directory + file_name, dtype=float)))


  # plot spline motion prior frequency vs. rmse for wnoj motion priors
  knot_period = 0.1
  spl_mp_dt_vals = [10.0, 3.0, 1.0, 0.7, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.03]

  fig, ax = plt.subplots(2,1, figsize=(6,3))
  # x_rmse_dict = {3: [], 4: [], 5: [], 6: []}
  # v_rmse_dict = {3: [], 4: [], 5: [], 6: []}
  x_rmse_dict = {4: [], 5: [], 6: []}
  v_rmse_dict = {4: [], 5: [], 6: []}
  for spline_data in spline_mc_data:
    # only look at runs that used wnoj model, dt = knot_period, and had motion priors
    if spline_data[0]["traj_type"] == 1 and spline_data[0]["dt"] == knot_period and spline_data[0]["mp_used"] and spline_data[0]["k"] in x_rmse_dict.keys() and spline_data[0]["mp_dt"] in spl_mp_dt_vals:
      x_rmse_dict[spline_data[0]["k"]].append((spline_data[0]["dt"] / spline_data[0]["mp_dt"], np.median(spline_data[1][:,0])))
      v_rmse_dict[spline_data[0]["k"]].append((spline_data[0]["dt"] / spline_data[0]["mp_dt"], np.median(spline_data[1][:,1])))

  ls_cp = linestyles.copy()
  for k, d in x_rmse_dict.items():
    d.sort(key=lambda x: x[0])
    ax[0].plot([t for t,_ in d], [val for _,val in d], marker='x', linestyle=ls_cp.pop(0), label="k="+str(k))

  ls_cp = linestyles.copy()
  for k, d in v_rmse_dict.items():
    d.sort(key=lambda x: x[0])
    ax[1].plot([t for t,_ in d], [val for _,val in d], marker='x', linestyle=ls_cp.pop(0), label="k="+str(k))
  ax[0].legend(loc="upper right")
  ax[0].set_xticklabels([])
  ax[0].set_ylabel("Position RMSE (m)")
  ax[1].set_ylabel("Velocity RMSE (m/s)")
  ax[1].set_xlabel(r'$\delta t / \delta t^\prime$')
  fig.suptitle("Error vs. Spline Motion Prior Frequency (WNOJ)")

  if save_figs:
    plt.savefig("spline_mp_freq_wnoj.png", dpi=300, bbox_inches="tight")

  # plot spline motion prior frequency vs. rmse for wnoa motion priors
  fig, ax = plt.subplots(2,1, figsize=(6,3))
  # x_rmse_dict = {3: [], 4: [], 5: [], 6: []}
  # v_rmse_dict = {3: [], 4: [], 5: [], 6: []}
  x_rmse_dict = {4: [], 5: [], 6: []}
  v_rmse_dict = {4: [], 5: [], 6: []}
  for spline_data in spline_mc_data:
    # only look at runs that used wnoa model, dt = knot_period, and had motion priors
    if spline_data[0]["traj_type"] == 0 and spline_data[0]["dt"] == knot_period and spline_data[0]["mp_used"] and spline_data[0]["k"] in x_rmse_dict.keys() and spline_data[0]["mp_dt"] in spl_mp_dt_vals:
      x_rmse_dict[spline_data[0]["k"]].append((spline_data[0]["dt"] / spline_data[0]["mp_dt"], np.median(spline_data[1][:,0])))
      v_rmse_dict[spline_data[0]["k"]].append((spline_data[0]["dt"] / spline_data[0]["mp_dt"], np.median(spline_data[1][:,1])))


  ls_cp = linestyles.copy()
  for k, d in x_rmse_dict.items():
    d.sort(key=lambda x: x[0])
    ax[0].plot([t for t,_ in d], [val for _,val in d], linestyle=ls_cp.pop(0), marker='x', label="k="+str(k))
  
  ls_cp = linestyles.copy()
  for k, d in v_rmse_dict.items():
    d.sort(key=lambda x: x[0])
    ax[1].plot([t for t,_ in d], [val for _,val in d], linestyle=ls_cp.pop(0), marker='x', label="k="+str(k))
  ax[0].legend(loc="upper right")
  ax[0].set_xticklabels([])
  ax[0].set_ylabel("Position RMSE (m)")
  ax[1].set_ylabel("Velocity RMSE (m/s)")
  ax[1].set_xlabel(r'$\delta t / \delta t^\diamond$')
  ax[0].set_title("Error vs. Spline Motion Prior Frequency (WNOA)")

  if save_figs:
    plt.savefig("spline_mp_freq_wnoa.png", dpi=300, bbox_inches="tight")


  # # plot rmse vs num params for wnoj
  # spl_dt_vals = [7.0, 5.0, 3.5, 2.0, 1.0, 0.5, 0.1, 0.05, 0.01]
  # gp_n_vals = [2000, 1000, 600, 300, 150, 30, 15, 3]

  # fig, ax = plt.subplots(3,1, figsize=(6,5))

  # # spl_nomp_x_rmse_dict = {3: [], 4: [], 5: [], 6: []}
  # # spl_nomp_v_rmse_dict = {3: [], 4: [], 5: [], 6: []}
  # # spl_nomp_solve_time_dict = {3: [], 4: [], 5: [], 6: []}
  # # spl_mp_x_rmse_dict = {3: [], 4: [], 5: [], 6: []}
  # # spl_mp_v_rmse_dict = {3: [], 4: [], 5: [], 6: []}
  # # spl_mp_solve_time_dict = {3: [], 4: [], 5: [], 6: []}
  # spl_nomp_x_rmse_dict = {4: [], 6: []}
  # spl_nomp_v_rmse_dict = {4: [], 6: []}
  # spl_nomp_solve_time_dict = {4: [], 6: []}
  # spl_mp_x_rmse_dict = {4: [], 6: []}
  # spl_mp_v_rmse_dict = {4: [], 6: []}
  # spl_mp_solve_time_dict = {4: [], 6: []}

  # gp_nomp_x_rmse = []
  # gp_nomp_v_rmse = []
  # gp_nomp_solve_time = []
  # gp_mp_x_rmse = []
  # gp_mp_v_rmse = []
  # gp_mp_solve_time = []

  # for spline_data in spline_mc_data:
  #   # only look at runs that used wnoj model and dt is in spl_dt_vals
  #   if spline_data[0]["traj_type"] == 1 and spline_data[0]["dt"] in spl_dt_vals and spline_data[0]["k"] in spl_nomp_x_rmse_dict.keys():
  #     if not spline_data[0]["mp_used"]:
  #       spl_nomp_x_rmse_dict[spline_data[0]["k"]].append((spline_data[1][0,3], np.median(spline_data[1][:,0])))
  #       spl_nomp_v_rmse_dict[spline_data[0]["k"]].append((spline_data[1][0,3], np.median(spline_data[1][:,1])))
  #       spl_nomp_solve_time_dict[spline_data[0]["k"]].append((spline_data[1][0,3], np.median(spline_data[1][:,2])))
  #     else:
  #       if abs(spline_data[0]["dt"] * 3.0 - spline_data[0]["mp_dt"]) < 1e-6:
  #         spl_mp_x_rmse_dict[spline_data[0]["k"]].append((spline_data[1][0,3], np.median(spline_data[1][:,0])))
  #         spl_mp_v_rmse_dict[spline_data[0]["k"]].append((spline_data[1][0,3], np.median(spline_data[1][:,1])))
  #         spl_mp_solve_time_dict[spline_data[0]["k"]].append((spline_data[1][0,3], np.median(spline_data[1][:,2])))

  # for gp_data in gp_mc_data:
  #   if gp_data[0]["traj_type"] == 1 and gp_data[0]["n"] in gp_n_vals:
  #     if not gp_data[0]["mp_used"]:
  #       gp_nomp_x_rmse.append((gp_data[1][0,3], np.median(gp_data[1][:,0])))
  #       gp_nomp_v_rmse.append((gp_data[1][0,3], np.median(gp_data[1][:,1])))
  #       gp_nomp_solve_time.append((gp_data[1][0,3], np.median(gp_data[1][:,2])))
  #     else:
  #       gp_mp_x_rmse.append((gp_data[1][0,3], np.median(gp_data[1][:,0])))
  #       gp_mp_v_rmse.append((gp_data[1][0,3], np.median(gp_data[1][:,1])))
  #       gp_mp_solve_time.append((gp_data[1][0,3], np.median(gp_data[1][:,2])))

  # gp_nomp_x_rmse.sort(key=lambda x: x[0])
  # gp_nomp_v_rmse.sort(key=lambda x: x[0])
  # gp_nomp_solve_time.sort(key=lambda x: x[0])
  # gp_mp_x_rmse.sort(key=lambda x: x[0])
  # gp_mp_v_rmse.sort(key=lambda x: x[0])
  # gp_mp_solve_time.sort(key=lambda x: x[0])

  # ls_cp = linestyles.copy()
  # mk_cp = markers.copy()
  # ax[0].plot([t for t,_ in gp_nomp_x_rmse], [val for _,val in gp_nomp_x_rmse], linestyle=ls_cp.pop(0), marker=mk_cp.pop(0), label="GP No MP")
  # ax[0].plot([t for t,_ in gp_mp_x_rmse], [val for _,val in gp_mp_x_rmse], linestyle=ls_cp.pop(0), marker=mk_cp.pop(0), label="GP MP")
  # for k, d in spl_nomp_x_rmse_dict.items():
  #   d.sort(key=lambda x: x[0])
  #   ax[0].plot([t for t,_ in d], [val for _,val in d], linestyle=ls_cp.pop(0), marker=mk_cp.pop(0), label="Spline No MP, k="+str(k))
  # for k, d in spl_mp_x_rmse_dict.items():
  #   d.sort(key=lambda x: x[0])
  #   ax[0].plot([t for t,_ in d], [val for _,val in d], linestyle=ls_cp.pop(0), marker=mk_cp.pop(0), label="Spline MP, k="+str(k))

  # ls_cp = linestyles.copy()
  # mk_cp = markers.copy()
  # ax[1].plot([t for t,_ in gp_nomp_v_rmse], [val for _,val in gp_nomp_v_rmse], linestyle=ls_cp.pop(0), marker=mk_cp.pop(0), label="GP No MP")
  # ax[1].plot([t for t,_ in gp_mp_v_rmse], [val for _,val in gp_mp_v_rmse], linestyle=ls_cp.pop(0), marker=mk_cp.pop(0), label="GP MP")
  # for k, d in spl_nomp_v_rmse_dict.items():
  #   d.sort(key=lambda x: x[0])
  #   ax[1].plot([t for t,_ in d], [val for _,val in d], linestyle=ls_cp.pop(0), marker=mk_cp.pop(0), label="Spline No MP, k="+str(k))
  # for k, d in spl_mp_v_rmse_dict.items():
  #   d.sort(key=lambda x: x[0])
  #   ax[1].plot([t for t,_ in d], [val for _,val in d], linestyle=ls_cp.pop(0), marker=mk_cp.pop(0), label="Spline MP, k="+str(k))

  # ls_cp = linestyles.copy()
  # mk_cp = markers.copy()
  # ax[2].plot([t for t,_ in gp_nomp_solve_time], [val for _,val in gp_nomp_solve_time], linestyle=ls_cp.pop(0), marker=mk_cp.pop(0), label="GP No MP")
  # ax[2].plot([t for t,_ in gp_mp_solve_time], [val for _,val in gp_mp_solve_time], linestyle=ls_cp.pop(0), marker=mk_cp.pop(0), label="GP MP")
  # for k, d in spl_nomp_solve_time_dict.items():
  #   d.sort(key=lambda x: x[0])
  #   ax[2].plot([t for t,_ in d], [val for _,val in d], linestyle=ls_cp.pop(0), marker=mk_cp.pop(0), label="Spline No MP, k="+str(k))
  # for k, d in spl_mp_solve_time_dict.items():
  #   d.sort(key=lambda x: x[0])
  #   ax[2].plot([t for t,_ in d], [val for _,val in d], linestyle=ls_cp.pop(0), marker=mk_cp.pop(0), label="Spline MP, k="+str(k))

  # ax[0].legend(loc="upper center", fontsize=6, bbox_to_anchor=(0.25,1.0))
  # ax[2].set_xlabel("Number of Parameters")
  # ax[0].set_ylabel("Position RMSE (m)")
  # ax[1].set_ylabel("Velocity RMSE (m/s)")
  # ax[2].set_ylabel("Solve Time (s)")
  # ax[0].set_title("Error vs. Number of Estimation Parameters (WNOJ)")

  # ax[0].set_xscale("log")
  # ax[1].set_xscale("log")
  # ax[2].set_xscale("log")
  # ax[0].set_yscale("log")
  # ax[1].set_yscale("log")
  # ax[0].set_xticklabels([])
  # ax[1].set_xticklabels([])

  # if save_figs:
  #   plt.savefig("error_v_params_wnoj.png", dpi=300, bbox_inches="tight")


  # # plot rmse vs num params for sinusoid (with wnoj priors)
  # spl_dt_vals = [2.0, 1.0, 0.5, 0.1, 0.05, 0.01]
  # gp_n_vals = [600, 300, 150, 30, 15, 3]

  # fig, ax = plt.subplots(3,1, figsize=(6,5))

  # # spl_nomp_x_rmse_dict = {3: [], 4: [], 5: [], 6: []}
  # # spl_nomp_v_rmse_dict = {3: [], 4: [], 5: [], 6: []}
  # # spl_nomp_solve_time_dict = {3: [], 4: [], 5: [], 6: []}
  # # spl_mp_x_rmse_dict = {3: [], 4: [], 5: [], 6: []}
  # # spl_mp_v_rmse_dict = {3: [], 4: [], 5: [], 6: []}
  # # spl_mp_solve_time_dict = {3: [], 4: [], 5: [], 6: []}
  # spl_nomp_x_rmse_dict = {4: [], 6: []}
  # spl_nomp_v_rmse_dict = {4: [], 6: []}
  # spl_nomp_solve_time_dict = {4: [], 6: []}
  # spl_mp_x_rmse_dict = {4: [], 6: []}
  # spl_mp_v_rmse_dict = {4: [], 6: []}
  # spl_mp_solve_time_dict = {4: [], 6: []}

  # gp_nomp_x_rmse = []
  # gp_nomp_v_rmse = []
  # gp_nomp_solve_time = []
  # gp_mp_x_rmse = []
  # gp_mp_v_rmse = []
  # gp_mp_solve_time = []

  # for spline_data in spline_mc_data:
  #   # only look at runs that used wnoj model and dt is in spl_dt_vals
  #   if spline_data[0]["traj_type"] == 2 and spline_data[0]["dt"] in spl_dt_vals and spline_data[0]["k"] in spl_nomp_x_rmse_dict.keys():
  #     if not spline_data[0]["mp_used"]:
  #       spl_nomp_x_rmse_dict[spline_data[0]["k"]].append((spline_data[1][0,3], np.median(spline_data[1][:,0])))
  #       spl_nomp_v_rmse_dict[spline_data[0]["k"]].append((spline_data[1][0,3], np.median(spline_data[1][:,1])))
  #       spl_nomp_solve_time_dict[spline_data[0]["k"]].append((spline_data[1][0,3], np.median(spline_data[1][:,2])))
  #     else:
  #       if abs(spline_data[0]["dt"] * 3.0 - spline_data[0]["mp_dt"]) < 1e-6 and spline_data[0]["model_type"] == 1:
  #         spl_mp_x_rmse_dict[spline_data[0]["k"]].append((spline_data[1][0,3], np.median(spline_data[1][:,0])))
  #         spl_mp_v_rmse_dict[spline_data[0]["k"]].append((spline_data[1][0,3], np.median(spline_data[1][:,1])))
  #         spl_mp_solve_time_dict[spline_data[0]["k"]].append((spline_data[1][0,3], np.median(spline_data[1][:,2])))

  # for gp_data in gp_mc_data:
  #   if gp_data[0]["traj_type"] == 2 and gp_data[0]["n"] in gp_n_vals:
  #     if not gp_data[0]["mp_used"]:
  #       gp_nomp_x_rmse.append((gp_data[1][0,3], np.median(gp_data[1][:,0])))
  #       gp_nomp_v_rmse.append((gp_data[1][0,3], np.median(gp_data[1][:,1])))
  #       gp_nomp_solve_time.append((gp_data[1][0,3], np.median(gp_data[1][:,2])))
  #     else:
  #       if gp_data[0]["model_type"] == 1:
  #         gp_mp_x_rmse.append((gp_data[1][0,3], np.median(gp_data[1][:,0])))
  #         gp_mp_v_rmse.append((gp_data[1][0,3], np.median(gp_data[1][:,1])))
  #         gp_mp_solve_time.append((gp_data[1][0,3], np.median(gp_data[1][:,2])))

  # gp_nomp_x_rmse.sort(key=lambda x: x[0])
  # gp_nomp_v_rmse.sort(key=lambda x: x[0])
  # gp_nomp_solve_time.sort(key=lambda x: x[0])
  # gp_mp_x_rmse.sort(key=lambda x: x[0])
  # gp_mp_v_rmse.sort(key=lambda x: x[0])
  # gp_mp_solve_time.sort(key=lambda x: x[0])

  # ls_cp = linestyles.copy()
  # mk_cp = markers.copy()
  # ax[0].plot([t for t,_ in gp_nomp_x_rmse], [val for _,val in gp_nomp_x_rmse], linestyle=ls_cp.pop(0), marker=mk_cp.pop(0), label="GP No MP")
  # ax[0].plot([t for t,_ in gp_mp_x_rmse], [val for _,val in gp_mp_x_rmse], linestyle=ls_cp.pop(0), marker=mk_cp.pop(0), label="GP MP")
  # for k, d in spl_nomp_x_rmse_dict.items():
  #   d.sort(key=lambda x: x[0])
  #   ax[0].plot([t for t,_ in d], [val for _,val in d], linestyle=ls_cp.pop(0), marker=mk_cp.pop(0), label="Spline No MP, k="+str(k))
  # for k, d in spl_mp_x_rmse_dict.items():
  #   d.sort(key=lambda x: x[0])
  #   ax[0].plot([t for t,_ in d], [val for _,val in d], linestyle=ls_cp.pop(0), marker=mk_cp.pop(0), label="Spline MP, k="+str(k))

  # ls_cp = linestyles.copy()
  # mk_cp = markers.copy()
  # ax[1].plot([t for t,_ in gp_nomp_v_rmse], [val for _,val in gp_nomp_v_rmse], linestyle=ls_cp.pop(0), marker=mk_cp.pop(0), label="GP No MP")
  # ax[1].plot([t for t,_ in gp_mp_v_rmse], [val for _,val in gp_mp_v_rmse], linestyle=ls_cp.pop(0), marker=mk_cp.pop(0), label="GP MP")
  # for k, d in spl_nomp_v_rmse_dict.items():
  #   d.sort(key=lambda x: x[0])
  #   ax[1].plot([t for t,_ in d], [val for _,val in d], linestyle=ls_cp.pop(0), marker=mk_cp.pop(0), label="Spline No MP, k="+str(k))
  # for k, d in spl_mp_v_rmse_dict.items():
  #   d.sort(key=lambda x: x[0])
  #   ax[1].plot([t for t,_ in d], [val for _,val in d], linestyle=ls_cp.pop(0), marker=mk_cp.pop(0), label="Spline MP, k="+str(k))

  # ls_cp = linestyles.copy()
  # mk_cp = markers.copy()
  # ax[2].plot([t for t,_ in gp_nomp_solve_time], [val for _,val in gp_nomp_solve_time], linestyle=ls_cp.pop(0), marker=mk_cp.pop(0), label="GP No MP")  
  # ax[2].plot([t for t,_ in gp_mp_solve_time], [val for _,val in gp_mp_solve_time], linestyle=ls_cp.pop(0), marker=mk_cp.pop(0), label="GP MP")
  # for k, d in spl_nomp_solve_time_dict.items():
  #   d.sort(key=lambda x: x[0])
  #   ax[2].plot([t for t,_ in d], [val for _,val in d], linestyle=ls_cp.pop(0), marker=mk_cp.pop(0), label="Spline No MP, k="+str(k))
  # for k, d in spl_mp_solve_time_dict.items():
  #   d.sort(key=lambda x: x[0])
  #   ax[2].plot([t for t,_ in d], [val for _,val in d], linestyle=ls_cp.pop(0), marker=mk_cp.pop(0), label="Spline MP, k="+str(k))

  # ax[0].legend(loc="upper center", fontsize=6, bbox_to_anchor=(0.35,1.0))
  # ax[2].set_xlabel("Number of Parameters")
  # ax[0].set_ylabel("Position RMSE (m)")
  # ax[1].set_ylabel("Velocity RMSE (m/s)")
  # ax[2].set_ylabel("Solve Time (s)")
  # ax[0].set_title("Error vs. Number of Estimation Parameters (Sinusoid)")

  # ax[0].set_xscale("log")
  # ax[1].set_xscale("log")
  # ax[2].set_xscale("log")
  # ax[0].set_yscale("log")
  # ax[1].set_yscale("log")
  # ax[0].set_xticklabels([])
  # ax[1].set_xticklabels([])

  # if save_figs:
  #   plt.savefig("error_v_params_sin.png", dpi=300, bbox_inches="tight")

  # plot rmse for wnoj traj
  wnoj_mc_sweep_datas = []
  wnoj_mc_sweep_datas.append(np.loadtxt(mc_sweep_directory + "gp_1_1_0_sweep.txt", dtype=float))
  wnoj_mc_sweep_datas.append(np.loadtxt(mc_sweep_directory + "gp_1_1_1_sweep.txt", dtype=float))
  wnoj_mc_sweep_datas.append(np.loadtxt(mc_sweep_directory + "spline_1_1_4_0_sweep.txt", dtype=float))
  wnoj_mc_sweep_datas.append(np.loadtxt(mc_sweep_directory + "spline_1_1_4_1_sweep.txt", dtype=float))
  wnoj_mc_sweep_datas.append(np.loadtxt(mc_sweep_directory + "spline_1_1_6_0_sweep.txt", dtype=float))
  wnoj_mc_sweep_datas.append(np.loadtxt(mc_sweep_directory + "spline_1_1_6_1_sweep.txt", dtype=float))

  # remove duplicate parameter values, keeping the one that used the highest value of dt or n
  # these are sorted by dt/n when logged, so the highest value is that with the highest row index
  for i, data in enumerate(wnoj_mc_sweep_datas):
    rows_to_keep = (data[:-1,3] - data[1:,3]).astype(bool).tolist() + [True]
    wnoj_mc_sweep_datas[i] = data[rows_to_keep,:]

  labels = ["GP No MP", "GP MP", "Spline No MP, k=4", "Spline MP, k=4", "Spline No MP, k=6", "Spline MP, k=6"]

  fig, ax = plt.subplots(3,1)
  [ax[0].plot(data[:,3], data[:,0], linestyle=ls, label=label) for data, ls, label in zip(wnoj_mc_sweep_datas, linestyles, labels)]
  [ax[1].plot(data[:,3], data[:,1], linestyle=ls, label=label) for data, ls, label in zip(wnoj_mc_sweep_datas, linestyles, labels)]
  [ax[2].plot(data[:,3], data[:,2], linestyle=ls, label=label) for data, ls, label in zip(wnoj_mc_sweep_datas, linestyles, labels)]

  ax[0].legend(loc="upper center", fontsize=6, bbox_to_anchor=(0.25,1.0))
  ax[2].set_xlabel("Number of Parameters")
  ax[0].set_ylabel("Position RMSE (m)")
  ax[1].set_ylabel("Velocity RMSE (m/s)")
  ax[2].set_ylabel("Solve Time (s)")
  ax[0].set_title("Error vs. Number of Estimation Parameters (WNOJ)")

  ax[0].set_xscale("log")
  ax[1].set_xscale("log")
  ax[2].set_xscale("log")
  ax[0].set_yscale("log")
  ax[1].set_yscale("log")
  ax[0].set_xticklabels([])
  ax[1].set_xticklabels([])

  if save_figs:
    plt.savefig("error_v_params_wnoj.png", dpi=300, bbox_inches="tight")

  # plot rmse for sin traj
  sin_mc_sweep_datas = []
  sin_mc_sweep_datas.append(np.loadtxt(mc_sweep_directory + "gp_2_1_0_sweep.txt", dtype=float))
  sin_mc_sweep_datas.append(np.loadtxt(mc_sweep_directory + "gp_2_1_1_sweep.txt", dtype=float))
  sin_mc_sweep_datas.append(np.loadtxt(mc_sweep_directory + "spline_2_1_4_0_sweep.txt", dtype=float))
  sin_mc_sweep_datas.append(np.loadtxt(mc_sweep_directory + "spline_2_1_4_1_sweep.txt", dtype=float))
  sin_mc_sweep_datas.append(np.loadtxt(mc_sweep_directory + "spline_2_1_6_0_sweep.txt", dtype=float))
  sin_mc_sweep_datas.append(np.loadtxt(mc_sweep_directory + "spline_2_1_6_1_sweep.txt", dtype=float))

  # remove duplicate parameter values, keeping the one that used the highest value of dt or n
  # these are sorted by dt/n when logged, so the highest value is that with the highest row index
  for i, data in enumerate(sin_mc_sweep_datas):
    rows_to_keep = (data[:-1,3] - data[1:,3]).astype(bool).tolist() + [True]
    sin_mc_sweep_datas[i] = data[rows_to_keep,:]

  labels = ["GP No MP", "GP MP", "Spline No MP, k=4", "Spline MP, k=4", "Spline No MP, k=6", "Spline MP, k=6"]

  fig, ax = plt.subplots(3,1)
  [ax[0].plot(data[:,3], data[:,0], linestyle=ls, label=label) for data, ls, label in zip(sin_mc_sweep_datas, linestyles, labels)]
  [ax[1].plot(data[:,3], data[:,1], linestyle=ls, label=label) for data, ls, label in zip(sin_mc_sweep_datas, linestyles, labels)]
  [ax[2].plot(data[:,3], data[:,2], linestyle=ls, label=label) for data, ls, label in zip(sin_mc_sweep_datas, linestyles, labels)]

  ax[0].legend(loc="upper center", fontsize=6, bbox_to_anchor=(0.35,1.0))
  ax[2].set_xlabel("Number of Parameters")
  ax[0].set_ylabel("Position RMSE (m)")
  ax[1].set_ylabel("Velocity RMSE (m/s)")
  ax[2].set_ylabel("Solve Time (s)")
  ax[0].set_title("Error vs. Number of Estimation Parameters (Sinusoid)")

  ax[0].set_xscale("log")
  ax[1].set_xscale("log")
  ax[2].set_xscale("log")
  ax[0].set_yscale("log")
  ax[1].set_yscale("log")
  ax[0].set_xticklabels([])
  ax[1].set_xticklabels([])

  if save_figs:
    plt.savefig("error_v_params_sin.png", dpi=300, bbox_inches="tight")

  # plot total position and motion prior cost at end of opt
  mc_sweep_directory = "../data/linear_sim/exp/mc_sweep/"
  wnoj_spline_sweep_cost = np.loadtxt(mc_sweep_directory + "spline_1_1_4_1_sweep.txt", dtype=float)
  wnoj_gp_sweep_cost = np.loadtxt(mc_sweep_directory + "gp_1_1_1_sweep.txt", dtype=float)

  fig, ax = plt.subplots(2,1)
  ax[0].plot(wnoj_spline_sweep_cost[:,3], wnoj_spline_sweep_cost[:,5], label="spline")
  ax[0].plot(wnoj_gp_sweep_cost[:,3], wnoj_gp_sweep_cost[:,5], label="gp")
  ax[1].plot(wnoj_spline_sweep_cost[:,3], wnoj_spline_sweep_cost[:,6], label="spline")
  ax[1].plot(wnoj_gp_sweep_cost[:,3], wnoj_gp_sweep_cost[:,6], label="gp")
  ax[0].legend()
  ax[0].set_ylabel("pos")
  ax[1].set_ylabel("mp")
  ax[0].set_xscale("log")
  ax[1].set_xscale("log")

  plt.show()  


if __name__ == "__main__":
  main()