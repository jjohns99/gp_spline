#pragma once

#include <vector>
#include <cmath>
#include <string>
#include <Eigen/Dense>
#include "lie_groups/se3.hpp"
#include "gp/lie_gp.hpp"

namespace hw
{

struct HwParams
{
  // sensor parameters
  Eigen::Matrix3d cam_mat;
  lie_groups::SE3d T_imu_cam;
  double timeshift_cam_imu;
  bool use_constant_pose_covariance;
  double sigma_april_trans;
  double sigma_april_rot;
  double sigma_pix;
  double sigma_gyro;
  double sigma_accel;
  double g;
  double apriltag_side_length;

  // data parameters
  std::string data_path;
  std::vector<double> data_start_times;
  std::vector<double> data_end_times;
  std::vector<int> apriltag_ids;

  // residuals to use
  bool use_imu;
  bool use_mp;

  // logging
  bool log_results;
  std::string log_directory;
  bool print_est_prog;
  int max_num_iterations;
  double sample_period;

  // spline estimation parameters
  int spline_k;
  double spline_dt;
  double spline_son_mp_dt;
  double spline_rn_mp_dt;
  int spline_son_mp_type;
  int spline_rn_mp_type;

  // gp estimation parameters
  double q_gp_rot;
  double q_gp_pos;
  int n_gp;
  int gp_son_type;

  // mocap to imu calibration
  lie_groups::SE3d T_imu_mocap;
  double timeshift_mocap_cam;
  bool calibrate_mocap_tx;
  bool low_pass_filter_mocap_orientation;
  double mocap_lpf_alpha;
};

} // namespace hw