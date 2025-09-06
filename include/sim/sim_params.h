#pragma once

#include <vector>
#include <cmath>
#include <string>
#include <Eigen/Dense>
#include "lie_groups/se3.hpp"
#include "gp/lie_gp.hpp"

namespace sim
{

// enum EstimatorType
// {
//   SE3SPLINE = 0,
//   SO3XR3SPLINE = 1,
//   SE3GP = 2,
//   SO3XR3GP = 3
// };

struct SimParams
{
  // simulated trajectory parameters
  int sim_traj_type;

  // wnoj trajectory params
  lie_groups::SE3d T_init;
  Eigen::Matrix<double, 6, 1> twist_init;
  Eigen::Matrix<double, 6, 1> twist_dot_init;

  // sinusoidal trajectory params
  double x_amplitude;
  double x_period;
  double x_offset;
  double x_phase;
  double y_amplitude;
  double y_period;
  double y_offset;
  double y_phase;
  double z_amplitude;
  double z_period;
  double z_offset;
  double z_phase;
  double roll_amplitude;
  double roll_period;
  double roll_offset;
  double roll_phase;
  double pitch_amplitude;
  double pitch_period;
  double pitch_offset;
  double pitch_phase;
  double yaw_slope;
  double yaw_offset;

  // IMU parameters
  double imu_frequency;
  double gyro_stdev;
  double accel_stdev;
  double gyro_walk_stdev;
  double accel_walk_stdev;
  double g;

  // camera parameters
  double camera_frequency;
  double pix_stdev;
  double pos_stdev;
  double rot_stdev;
  bool sim_constant_cov;
  bool est_constant_cov;
  Eigen::Matrix3d cam_mat;
  lie_groups::SE3d T_imu_c;

  // camera field of view
  double max_range;
  double max_azimuth;
  double max_elevation;
  bool assert_facing_camera;
  double apriltag_side_length;
  
  // sim parameters
  double sim_end_time;
  double sim_period;

  // residuals to use
  bool use_imu;
  bool use_mp;

  // opt params
  int max_num_iterations;

  // monte carlo
  int num_monte_carlo_runs;
  bool seed_random;

  // logging
  bool log_results;
  std::string log_directory;
  bool print_est_prog;

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
};

} // namespace sim