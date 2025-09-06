#include "estimator/gp_so3xr3_estimator.h"

namespace estimator
{

GPSO3xR3Estimator::GPSO3xR3Estimator(EstimatorParams est_params) : EstimatorBase{est_params}
{}

void GPSO3xR3Estimator::init_gp(Matrix6d Q, int n, gp::ModelType son_type)
{
  set_start_and_end_time();
  Q_ = Q;
  n_ = n;
  son_type_ = son_type;

  R_.reset(new std::vector<SOn>());
  p_.reset(new std::vector<Rn>());
  om_.reset(new std::vector<Rn>());
  v_.reset(new std::vector<Rn>());
  if(son_type_ == gp::ModelType::JERK) omdot_.reset(new std::vector<Rn>());
  a_.reset(new std::vector<Rn>());

  assert(z_land_ != nullptr);
  std::vector<double> t_vec;
  for(int i = 0; i < z_land_->size(); ++i)
  {
    if(i % n_) continue;

    R_->emplace_back(new double[4]{0.0, 0.0, 0.0, 1.0});
    p_->emplace_back(new double[3]{0.0, 0.0, 0.0});
    om_->emplace_back(new double[3]{0.0, 0.0, 0.0});
    v_->emplace_back(new double[3]{0.0, 0.0, 0.0});
    a_->emplace_back(new double[3]{0.0, 0.0, 0.0});
    if(son_type_ == gp::ModelType::JERK) omdot_->emplace_back(new double[3]{0.0, 0.0, 0.0});
    t_vec.push_back(z_land_->at(i).first);
  }
  // add extra parameter at very last pose measurement, if necessary
  if(t_vec.back() < z_land_->back().first)
  {
    R_->emplace_back(new double[4]{0.0, 0.0, 0.0, 1.0});
    p_->emplace_back(new double[3]{0.0, 0.0, 0.0});
    om_->emplace_back(new double[3]{0.0, 0.0, 0.0});
    v_->emplace_back(new double[3]{0.0, 0.0, 0.0});
    a_->emplace_back(new double[3]{0.0, 0.0, 0.0});
    if(son_type_ == gp::ModelType::JERK) omdot_->emplace_back(new double[3]{0.0, 0.0, 0.0});
    t_vec.push_back(z_land_->back().first);
  }

  gp_.reset(new GP(gp::ModelType::JERK, son_type_, Q_));
  gp_->init_est_params(t_vec, p_, v_, R_, om_, a_, (son_type_ == gp::ModelType::JERK) ? omdot_ : nullptr);
}

void GPSO3xR3Estimator::setup_opt(ceres::Problem& problem, bool use_motion_priors)
{
  // can only set problem options in constructor, unfortunately
  if(z_land_ != nullptr)
  {
    pre_eval_map_.clear();

    std::vector<double> t_pre_eval;
    int ind = 0;
    for(int i = 0; i < z_land_->size(); ++i)
    {
      if(!(i % n_) || !gp_->is_time_valid(z_land_->at(i).first)) continue;
      t_pre_eval.push_back(z_land_->at(i).first);
      pre_eval_map_[i] = ind++;
    }
    
    ceres::Problem::Options problem_options;
    lm_eval_callback_.reset(new PoseEvalCallback<GP>(t_pre_eval, gp_));
    problem_options.evaluation_callback = lm_eval_callback_.get();

    problem = ceres::Problem(problem_options);
  }

  for(auto& param : *R_)
  {
    problem.AddParameterBlock(param.data(), 4, new SOnParameterization<typename SOn::NonmapT>());
  }

  // add IMU residuals to problem
  if(z_imu_ != nullptr)
  {
    new (&gyro_bias_) Eigen::Map<Eigen::Vector3d>(new double[3]{est_params_.gyro_bias(0), est_params_.gyro_bias(1), est_params_.gyro_bias(2)});
    new (&accel_bias_) Eigen::Map<Eigen::Vector3d>(new double[3]{est_params_.accel_bias(0), est_params_.accel_bias(1), est_params_.accel_bias(2)});
    problem.AddParameterBlock(gyro_bias_.data(), 3);
    problem.AddParameterBlock(accel_bias_.data(), 3);

    for(auto const& z_imu : *z_imu_)
    {
      if(!gp_->is_time_valid(z_imu.first)) continue;

      int i = gp_->get_i(z_imu.first);
      ceres::CostFunction* r_imu = new GPImuResidualSO3xR3(z_imu.second, z_imu.first, est_params_.g, sqrt_W_imu_, gp_);
      std::vector<double*> params;
      for(int j = 0; j <= 1; ++j) params.push_back(p_->at(i + j).data());
      for(int j = 0; j <= 1; ++j) params.push_back(v_->at(i + j).data());
      for(int j = 0; j <= 1; ++j) params.push_back(a_->at(i + j).data());
      for(int j = 0; j <= 1; ++j) params.push_back(R_->at(i + j).data());
      for(int j = 0; j <= 1; ++j) params.push_back(om_->at(i + j).data());
      if(son_type_ == gp::ModelType::JERK)
      {
        for(int j = 0; j <= 1; ++j) params.push_back(omdot_->at(i + j).data());
      }
      params.push_back(gyro_bias_.data());
      params.push_back(accel_bias_.data());

      problem.AddResidualBlock(r_imu, nullptr, params);
    }
  }

  // add landmark residuals to problem
  if(z_land_ != nullptr)
  {
    for(int ind = 0; ind < z_land_->size(); ++ind)
    {
      double t = z_land_->at(ind).first;
      if(!gp_->is_time_valid(t)) continue;

      // check if this image corresponds to an estimation time
      if(!(ind % n_) || (ind == (z_land_->size() - 1)))
      {
        int i = ind / n_;
        std::vector<double*> params;
        params.push_back(p_->at(i).data());
        params.push_back(R_->at(i).data());

        for(auto const& z_land : z_land_->at(ind).second)
        {
          Matrix6d lm_sqrt_W;
          if(!use_constant_pose_cov_) get_apriltag_pose_sqrt_W(z_land.second, lm_sqrt_W);
          else lm_sqrt_W = pose_sqrt_W_;

          ceres::CostFunction* r_land = new ApriltagResidualSO3xR3(z_land.second, landmark_truth_[z_land.first], lm_sqrt_W, est_params_.T_b_c.inverse());
          problem.AddResidualBlock(r_land, nullptr, params);
        }
      }
      else
      {
        int i = gp_->get_i(t);
        std::vector<double*> params;
        for(int j = 0; j <= 1; ++j) params.push_back(p_->at(i + j).data());
        for(int j = 0; j <= 1; ++j) params.push_back(v_->at(i + j).data());
        for(int j = 0; j <= 1; ++j) params.push_back(a_->at(i + j).data());
        for(int j = 0; j <= 1; ++j) params.push_back(R_->at(i + j).data());
        for(int j = 0; j <= 1; ++j) params.push_back(om_->at(i + j).data());
        if(son_type_ == gp::ModelType::JERK)
        {
          for(int j = 0; j <= 1; ++j) params.push_back(omdot_->at(i + j).data());
        }

        for(auto const& z_land : z_land_->at(ind).second)
        {
          Matrix6d lm_sqrt_W;
          if(!use_constant_pose_cov_) get_apriltag_pose_sqrt_W(z_land.second, lm_sqrt_W);
          else lm_sqrt_W = pose_sqrt_W_;
          int pre_eval_id = pre_eval_map_.find(ind)->second;
          ceres::CostFunction* r_land = new GPApriltagResidualSO3xR3(z_land.second, landmark_truth_[z_land.first], pre_eval_id, lm_sqrt_W,
                  gp_, est_params_.T_b_c.inverse());
          problem.AddResidualBlock(r_land, nullptr, params);
        }
      }
    }
  }

  // add motion prior residuals to problem
  if(use_motion_priors)
  {
    for(int i = 0; i < R_->size()-1; ++i)
    {
      std::vector<double*> params;
      for(int j = 0; j <= 1; ++j) params.push_back(R_->at(i + j).data());
      for(int j = 0; j <= 1; ++j) params.push_back(om_->at(i + j).data());
      if(son_type_ == gp::ModelType::JERK) 
      {
        for(int j = 0; j <= 1; ++j) params.push_back(omdot_->at(i + j).data());

        Eigen::Matrix<double, 9, 9> Q_i_inv = gp_->get_son_gp()->get_Q_inv_i_i1(i);
        Eigen::LLT<Eigen::Matrix<double, 9, 9>> chol(Q_i_inv);
        Eigen::Matrix<double, 9, 9> sqrt_W_dyn = chol.matrixL().transpose();

        ceres::CostFunction* r_son_dyn = new GPDynamicsResidual<SOnGP, 9, 4, 3>(sqrt_W_dyn, gp_->get_son_gp(), i);
        problem.AddResidualBlock(r_son_dyn, nullptr, params);
      }
      else
      {
        Eigen::Matrix<double, 6, 6> Q_i_inv = gp_->get_son_gp()->get_Q_inv_i_i1(i);
        Eigen::LLT<Eigen::Matrix<double, 6, 6>> chol(Q_i_inv);
        Eigen::Matrix<double, 6, 6> sqrt_W_dyn = chol.matrixL().transpose();

        ceres::CostFunction* r_son_dyn = new GPDynamicsResidual<SOnGP, 6, 4, 3>(sqrt_W_dyn, gp_->get_son_gp(), i);
        problem.AddResidualBlock(r_son_dyn, nullptr, params);
      }
    }

    for(int i = 0; i < p_->size()-1; ++i)
    {
      std::vector<double*> params;
      for(int j = 0; j <= 1; ++j) params.push_back(p_->at(i + j).data());
      for(int j = 0; j <= 1; ++j) params.push_back(v_->at(i + j).data());
      for(int j = 0; j <= 1; ++j) params.push_back(a_->at(i + j).data());

      Eigen::Matrix<double, 9, 9> Q_i_inv = gp_->get_rn_gp()->get_Q_inv_i_i1(i);
      Eigen::LLT<Eigen::Matrix<double, 9, 9>> chol(Q_i_inv);
      Eigen::Matrix<double, 9, 9> sqrt_W_dyn = chol.matrixL().transpose();

      ceres::CostFunction* r_rn_dyn = new GPDynamicsResidual<RnGP, 9, 3, 3>(sqrt_W_dyn, gp_->get_rn_gp(), i);
      problem.AddResidualBlock(r_rn_dyn, nullptr, params);
    }
  }
}

std::shared_ptr<gp::SOnxRnGP<lie_groups::Map<lie_groups::SO3d>, Eigen::Map<Eigen::Vector3d>, Eigen::Map<Eigen::Vector3d>>> GPSO3xR3Estimator::get_gp()
{
  return gp_;
}

} // namespace estimator