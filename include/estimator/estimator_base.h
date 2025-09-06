#pragma once

#include <vector>
#include <memory>
#include <cmath>
#include <limits>
#include <Eigen/Dense>
#include <ceres/ceres.h>

#include "lie_groups/se3.hpp"

namespace estimator
{

struct EstimatorParams
{
  double g = 9.81;
  lie_groups::SE3d T_b_c;

  // these are only used by the spline estimators
  bool use_bias_splines = false; // use splines for imu biases. assume constant if false
  int bias_spline_k;
  double bias_spline_dt;

  // initial guess for imu biases
  Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
  Eigen::Vector3d accel_bias = Eigen::Vector3d::Zero();

  // parameters to determine apriltag pose covariance
  Eigen::Matrix3d cam_mat = Eigen::Matrix3d::Identity();
  double apriltag_side_length;
};

// base estimator class to be inherited by SE(3) gp, SO(3)xR3, and spline estimators
class EstimatorBase
{
private:
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Matrix6d = Eigen::Matrix<double, 6, 6>;

public:
  EstimatorBase(EstimatorParams est_params);
  void add_imu_measurements(std::shared_ptr<std::vector<std::pair<double, Vector6d>>> z_imu, Matrix6d sqrt_W);
  void add_lm_pose_measurements(std::shared_ptr<std::vector<std::pair<double, std::vector<std::pair<int, lie_groups::SE3d>>>>> z_land, Eigen::Matrix2d cam_feat_W);
  void add_lm_pose_measurements(std::shared_ptr<std::vector<std::pair<double, std::vector<std::pair<int, lie_groups::SE3d>>>>> z_land, Matrix6d pose_sqrt_W);
  void set_landmark_truth(std::vector<lie_groups::SE3d> const& lm);
  void get_gyro_bias(Eigen::Vector3d* vec);
  void get_accel_bias(Eigen::Vector3d* vec);
  std::pair<double, double> get_start_and_end_time();

  ceres::Solver::Summary solve(bool print_calib_progress = true, bool use_motion_priors = false, int max_num_iterations = 100);

protected:
  void set_start_and_end_time();
  void set_apriltag_cov_params();
  void get_apriltag_pose_sqrt_W(lie_groups::SE3d const& T_l_c, Matrix6d& sqrt_W_lm);
  virtual void setup_opt(ceres::Problem& problem, bool use_motion_priors = false) = 0;

  EstimatorParams est_params_;

  // ceres::Problem doesn't take ownership of eval callbacks,
  // so we have to delete it
  std::unique_ptr<ceres::EvaluationCallback> lm_eval_callback_;

  double start_time_;
  double end_time_;

  Matrix6d sqrt_W_imu_;

  // parameters for determining apriltag pose covariance
  Eigen::Matrix2d cam_feature_W_;
  std::vector<Eigen::Vector3d> apriltag_corners_;
  Eigen::Matrix<double, 8, 8> W_cam_;

  // constant pose sqrt inv covariance (only used if provided, otherwise use camera covariance)
  Matrix6d pose_sqrt_W_;
  bool use_constant_pose_cov_ = false;

  std::shared_ptr<std::vector<std::pair<double, Vector6d>>> z_imu_;
  std::shared_ptr<std::vector<std::pair<double, std::vector<std::pair<int, lie_groups::SE3d>>>>> z_land_;

  // use these if use_bias_splines = false
  Eigen::Map<Eigen::Vector3d> gyro_bias_;
  Eigen::Map<Eigen::Vector3d> accel_bias_;

  std::vector<lie_groups::SE3d> landmark_truth_;
};

} // namespace estimator