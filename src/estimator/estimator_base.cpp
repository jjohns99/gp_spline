#include "estimator/estimator_base.h"

namespace estimator
{

EstimatorBase::EstimatorBase(EstimatorParams est_params) : est_params_{est_params}, gyro_bias_{nullptr}, accel_bias_{nullptr}, z_land_{nullptr}, z_imu_{nullptr}
{}

void EstimatorBase::add_imu_measurements(std::shared_ptr<std::vector<std::pair<double, Vector6d>>> z_imu, Matrix6d sqrt_W)
{
  z_imu_ = z_imu;
  sqrt_W_imu_ = sqrt_W;
}

void EstimatorBase::add_lm_pose_measurements(std::shared_ptr<std::vector<std::pair<double, std::vector<std::pair<int, lie_groups::SE3d>>>>> z_land, Eigen::Matrix2d cam_feat_W)
{
  z_land_ = z_land;
  cam_feature_W_ = cam_feat_W;
  set_apriltag_cov_params();
}

void EstimatorBase::add_lm_pose_measurements(std::shared_ptr<std::vector<std::pair<double, std::vector<std::pair<int, lie_groups::SE3d>>>>> z_land, Matrix6d pose_sqrt_W)
{
  z_land_ = z_land;
  pose_sqrt_W_ = pose_sqrt_W;
  use_constant_pose_cov_ = true;
}

void EstimatorBase::set_landmark_truth(std::vector<lie_groups::SE3d> const& lm)
{
  landmark_truth_ = lm;
}

void EstimatorBase::get_gyro_bias(Eigen::Vector3d* vec)
{
  *vec = gyro_bias_;
}

void EstimatorBase::get_accel_bias(Eigen::Vector3d* vec)
{
  *vec = accel_bias_;
}

ceres::Solver::Summary EstimatorBase::solve(bool print_calib_progress, bool use_motion_priors, int max_num_iterations)
{
  ceres::Problem problem;
  setup_opt(problem, use_motion_priors);

  ceres::Solver::Options solver_options;
  solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  solver_options.minimizer_progress_to_stdout = print_calib_progress;
  solver_options.num_threads = 1;
  solver_options.max_num_iterations = max_num_iterations;
  solver_options.check_gradients = false;
  solver_options.update_state_every_iteration = true;

  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);    
  if(print_calib_progress) std::cout << summary.FullReport();
  return summary;
}

void EstimatorBase::set_start_and_end_time()
{
  // this assumes that the measurement vectors are ordered by time
  double t0_imu = (z_imu_ != nullptr) ? z_imu_->at(0).first : std::numeric_limits<double>::infinity();
  double t0_land = (z_land_ != nullptr) ? z_land_->at(0).first : std::numeric_limits<double>::infinity();
  double tf_imu = (z_imu_ != nullptr) ? z_imu_->back().first : -std::numeric_limits<double>::infinity();
  double tf_land = (z_land_ != nullptr) ? z_land_->back().first : -std::numeric_limits<double>::infinity();
  
  start_time_ = std::min({t0_imu, t0_land});
  end_time_ = std::max({tf_imu, tf_land});
}

std::pair<double, double> EstimatorBase::get_start_and_end_time()
{
  return std::make_pair(start_time_, end_time_);
}

void EstimatorBase::set_apriltag_cov_params()
{
  W_cam_.setZero();
  for(int i = 0; i < 4; ++i) W_cam_.block<2,2>(2*i, 2*i) = cam_feature_W_;

  apriltag_corners_.push_back(est_params_.apriltag_side_length/2.0 * (Eigen::Vector3d() << -1.0, 1.0, 0.0).finished());
  apriltag_corners_.push_back(est_params_.apriltag_side_length/2.0 * (Eigen::Vector3d() << 1.0, 1.0, 0.0).finished());
  apriltag_corners_.push_back(est_params_.apriltag_side_length/2.0 * (Eigen::Vector3d() << 1.0, -1.0, 0.0).finished());
  apriltag_corners_.push_back(est_params_.apriltag_side_length/2.0 * (Eigen::Vector3d() << -1.0, -1.0, 0.0).finished());
}

void EstimatorBase::get_apriltag_pose_sqrt_W(lie_groups::SE3d const& T_l_c, Matrix6d& sqrt_W_lm)
{
  Eigen::Matrix<double, 8, 6> J_apriltag_est;
  for(int i = 0; i < apriltag_corners_.size(); ++i)
  {
    Eigen::Vector3d corner_c = T_l_c * apriltag_corners_[i];
    Eigen::Matrix<double, 2, 3> dpi_dcorner = (Eigen::Matrix<double, 2, 3>() << 1.0/corner_c(2), 0.0, -corner_c(0)/std::pow(corner_c(2), 2),
                                                                                0.0, 1.0/corner_c(2), -corner_c(1)/std::pow(corner_c(2), 2)).finished();
    Eigen::Matrix<double, 3, 6> dcorner_dT;
    dcorner_dT.block<3,3>(0,0).setIdentity();
    dcorner_dT.block<3,3>(0,3) = -lie_groups::SO3d::hat(corner_c);
    J_apriltag_est.block<2,6>(2*i, 0) = est_params_.cam_mat.block<2,2>(0,0) * dpi_dcorner * dcorner_dT;
  }
  Matrix6d lm_pose_W = J_apriltag_est.transpose() * W_cam_ * J_apriltag_est;
  Eigen::LLT<Matrix6d> chol(lm_pose_W);
  assert(chol.info() == Eigen::Success); // will fail if cov isn't positive definite
  
  sqrt_W_lm = chol.matrixL().transpose();
}

} // namespace estimator