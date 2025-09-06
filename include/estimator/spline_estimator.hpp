#pragma once

#include <vector>
#include <memory>
#include <cmath>
#include <limits>
#include <Eigen/Dense>
#include <ceres/ceres.h>

#include "lie_groups/so3.hpp"
#include "lie_groups/se3.hpp"
#include "spline/rn_spline.hpp"
#include "spline/sonxrn_spline.hpp"
#include "residuals/pose_eval_callback.hpp"
#include "residuals/dummy_callback.h"
#include "residuals/spline_apriltag_residual.hpp"
#include "residuals/spline_imu_residual.hpp"
#include "residuals/spline_dynamics_residual.hpp"
#include "residuals/sonxrn_spline_dynamics_residual.hpp"
#include "residuals/mocap_spline_residual.hpp"
#include "parameterizations/sen_parameterization.hpp"
#include "estimator/estimator_base.h"
#include "estimator/rn_spline_motion_prior.hpp"
#include "estimator/lie_spline_motion_prior.hpp"

namespace estimator
{

// have to template specialize for so3xr3 because motion priors are different

// SplineT must either be LieSpline<Map<SE3>> or LieSpline<Map<SO3xR3>>
template <typename SplineT>
class SplineEstimatorBase : public EstimatorBase
{
private:
  using SE3 = typename SplineT::GroupT;
  using BiasSplineT = spline::RnSpline<Eigen::Map<Eigen::Vector3d>, 3>;
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Matrix6d = Eigen::Matrix<double, 6, 6>;

public:
  SplineEstimatorBase(EstimatorParams est_params) : EstimatorBase{est_params}, motion_priors_set_{false}
  {}

  // must be called after all measurements have been added
  void init_spline(double dt, int k, std::vector<Eigen::Matrix<double, 7, 1>> const* init_guess = nullptr)
  {
    k_ = k;
    set_start_and_end_time();

    ctrl_pts_.reset(new std::vector<SE3>);
    if(init_guess == nullptr)
    {
      int num_cp = static_cast<int>(std::ceil((end_time_ - start_time_)/dt)) + k;
      for(int i = 0; i < num_cp; ++i)
        ctrl_pts_->emplace_back(new double[7]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0});
    }
    else
    {
      for(int i = 0; i < init_guess->size(); ++i)
        ctrl_pts_->emplace_back(new double[7]{init_guess->at(i)(0), init_guess->at(i)(1), init_guess->at(i)(2),
                                init_guess->at(i)(3), init_guess->at(i)(4), init_guess->at(i)(5), init_guess->at(i)(6)});
    }

    spline_.reset(new SplineT(k));
    spline_->init_ctrl_pts(ctrl_pts_, start_time_, dt);

    // initialize imu bias splines, if desired
    if(est_params_.use_bias_splines)
    {
      gyro_bias_ctrl_pts_.reset(new std::vector<Eigen::Map<Eigen::Vector3d>>);
      accel_bias_ctrl_pts_.reset(new std::vector<Eigen::Map<Eigen::Vector3d>>);
      int num_cp = static_cast<int>(std::ceil((end_time_ - start_time_)/est_params_.bias_spline_dt)) + est_params_.bias_spline_k;
      for(int i = 0; i < num_cp; ++i)
      {
        gyro_bias_ctrl_pts_->emplace_back(new double[3]{est_params_.gyro_bias(0), est_params_.gyro_bias(1), est_params_.gyro_bias(2)});
        accel_bias_ctrl_pts_->emplace_back(new double[3]{est_params_.accel_bias(0), est_params_.accel_bias(1), est_params_.accel_bias(2)});
      }

      gyro_bias_spline_.reset(new BiasSplineT(est_params_.bias_spline_k));
      accel_bias_spline_.reset(new BiasSplineT(est_params_.bias_spline_k));
      gyro_bias_spline_->init_ctrl_pts(gyro_bias_ctrl_pts_, start_time_, est_params_.bias_spline_dt);
      accel_bias_spline_->init_ctrl_pts(accel_bias_ctrl_pts_, start_time_, est_params_.bias_spline_dt);
    }
  }

  std::shared_ptr<SplineT> get_spline()
  {
    return spline_;
  }

  void get_gyro_bias(std::shared_ptr<BiasSplineT>& spl)
  {
    assert(gyro_bias_spline_ != nullptr);
    spl = gyro_bias_spline_;
  }

  void get_accel_bias(std::shared_ptr<BiasSplineT>& spl)
  {
    assert(accel_bias_spline_ != nullptr);
    spl = accel_bias_spline_;
  }

protected:
  virtual void setup_opt(ceres::Problem& problem, bool use_motion_priors = false) override
  {
    // can only set problem options in constructor, unfortunately
    if(z_land_ != nullptr)
    {
      std::vector<double> t_pre_eval;
      for(auto const& z : *z_land_) t_pre_eval.push_back(z.first);
      
      ceres::Problem::Options problem_options;
      lm_eval_callback_.reset(new PoseEvalCallback<SplineT>(t_pre_eval, spline_));
      problem_options.evaluation_callback = lm_eval_callback_.get();

      problem = ceres::Problem(problem_options);
    }
    else
    {
      ceres::Problem::Options problem_options;
      lm_eval_callback_.reset(new DummyCallback());
      problem_options.evaluation_callback = lm_eval_callback_.get();

      problem = ceres::Problem(problem_options);
    }

    for(auto& ctrl_pt : *ctrl_pts_)
    {
      problem.AddParameterBlock(ctrl_pt.data(), 7, new SEnParameterization<typename SE3::NonmapT>());
    }

    // add IMU residuals to problem
    if(z_imu_ != nullptr)
    {
      if(!est_params_.use_bias_splines)
      {
        new (&gyro_bias_) Eigen::Map<Eigen::Vector3d>(new double[3]{est_params_.gyro_bias(0), est_params_.gyro_bias(1), est_params_.gyro_bias(2)});
        new (&accel_bias_) Eigen::Map<Eigen::Vector3d>(new double[3]{est_params_.accel_bias(0), est_params_.accel_bias(1), est_params_.accel_bias(2)});
        problem.AddParameterBlock(gyro_bias_.data(), 3);
        problem.AddParameterBlock(accel_bias_.data(), 3);
      }

      for(auto const& z_imu : *z_imu_)
      {
        if(!spline_->get_time_range().valid(z_imu.first)) continue;

        int i = spline_->get_i(z_imu.first);
        ceres::CostFunction* r_imu = new SplineImuResidual<SplineT>(z_imu.second, z_imu.first, est_params_.g, sqrt_W_imu_, spline_, 
                est_params_.use_bias_splines ? gyro_bias_spline_ : nullptr, est_params_.use_bias_splines ? accel_bias_spline_ : nullptr);
        std::vector<double*> params;
        for(int j = 0; j < k_; ++j) params.push_back(ctrl_pts_->at(i + 1 + j).data());
        if(est_params_.use_bias_splines)
        {
          int i_bias = gyro_bias_spline_->get_i(z_imu.first);
          for(int j = 0; j < est_params_.bias_spline_k; ++j) params.push_back(gyro_bias_ctrl_pts_->at(i_bias + 1 + j).data());
          for(int j = 0; j < est_params_.bias_spline_k; ++j) params.push_back(accel_bias_ctrl_pts_->at(i_bias + 1 + j).data());
        }
        else
        {
          params.push_back(gyro_bias_.data());
          params.push_back(accel_bias_.data());
        }

        problem.AddResidualBlock(r_imu, nullptr, params);
      }
    }

    // add landmark residuals to problem
    if(z_land_ != nullptr)
    {
      for(int ind = 0; ind < z_land_->size(); ++ind)
      {
        double t = z_land_->at(ind).first;
        int i = spline_->get_i(t);
        std::vector<double*> params;
        for(int j = 0; j < k_; ++j) params.push_back(ctrl_pts_->at(i + 1 + j).data());

        for(auto const& z_land : z_land_->at(ind).second)
        {
          Matrix6d lm_sqrt_W;
          if(!use_constant_pose_cov_) get_apriltag_pose_sqrt_W(z_land.second, lm_sqrt_W);
          else lm_sqrt_W = pose_sqrt_W_;
          ceres::CostFunction* r_land = new SplineApriltagResidual<SplineT>(z_land.second, landmark_truth_[z_land.first], ind, lm_sqrt_W,
                  spline_, est_params_.T_b_c.inverse());
          problem.AddResidualBlock(r_land, nullptr, params);
        }
      }
    }

    // add motion priors
    if(use_motion_priors && motion_priors_set_) add_motion_priors(problem);
  }

  virtual void add_motion_priors(ceres::Problem& problem) = 0;

  std::shared_ptr<SplineT> spline_;
  std::shared_ptr<std::vector<SE3>> ctrl_pts_;

  int k_;

  // use these if use_bias_splines = true
  std::shared_ptr<BiasSplineT> gyro_bias_spline_;
  std::shared_ptr<BiasSplineT> accel_bias_spline_;
  std::shared_ptr<std::vector<Eigen::Map<Eigen::Vector3d>>> gyro_bias_ctrl_pts_;
  std::shared_ptr<std::vector<Eigen::Map<Eigen::Vector3d>>> accel_bias_ctrl_pts_;

  // motion prior params
  Matrix6d Qc_;
  bool motion_priors_set_;
};


template <typename SplineT>
class SplineEstimator : public SplineEstimatorBase<SplineT>
{
private:
  using Matrix6d = Eigen::Matrix<double, 6, 6>;

public:
  SplineEstimator(EstimatorParams est_params) : SplineEstimatorBase<SplineT>(est_params)
  {}

  void set_up_motion_priors(ModelType mp_type, double mp_dt, Matrix6d Qc)
  {
    mp_type_ = mp_type;
    mp_dt_ = mp_dt;
    this->Qc_ = Qc;

    this->motion_priors_set_ = true;
  }

  // calibrate transformation from mocap frame to spline frame
  // (it is assumed that spline tx and mocap tx have same base)
  std::pair<double, lie_groups::SE3d> calibrate_mocap_transformation(std::shared_ptr<std::vector<std::pair<double, lie_groups::SE3d>>> z_mocap,
    lie_groups::SE3d const& T_guess, double time_offset_guess)
  {
    lie_groups::Map<lie_groups::SE3d> T_bm(new double[7]{T_guess.translation()(0), T_guess.translation()(1), T_guess.translation()(2), 
      T_guess.rotation().mem().x(), T_guess.rotation().mem().y(), T_guess.rotation().mem().z(), T_guess.rotation().mem().w()});
    double* mocap_time_offset = new double{time_offset_guess};

    ceres::Problem problem;
    problem.AddParameterBlock(T_bm.data(), 7, new SEnParameterization<lie_groups::SE3d>());
    problem.AddParameterBlock(mocap_time_offset, 1);
    problem.SetParameterLowerBound(mocap_time_offset, 0, -this->spline_->get_dt() / 2.0);
    problem.SetParameterUpperBound(mocap_time_offset, 0, this->spline_->get_dt() / 2.0);

    for(auto const& z : *z_mocap)
    {
      if(!this->spline_->get_time_range().valid(z.first)) continue;

      ceres::SizedCostFunction<6, 7, 1>* r_mocap = new MocapSplineResidual(z.second, z.first, this->spline_);

      std::vector<double*> params;
      params.push_back(T_bm.data());
      params.push_back(mocap_time_offset);

      problem.AddResidualBlock(r_mocap, nullptr, params);
    }

    ceres::Solver::Options solver_options;
    solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    solver_options.minimizer_progress_to_stdout = true;
    solver_options.num_threads = 1;
    solver_options.max_num_iterations = 200;
    solver_options.check_gradients = false;

    std::cout << "Calibrating transformation from spline frame to mocap frame and time offset\n";
    ceres::Solver::Summary summary;
    ceres::Solve(solver_options, &problem, &summary);
    std::cout << summary.FullReport();

    std::cout << "T_bm: " << T_bm.matrix() << "\n";
    std::cout << "Time offset: " << *mocap_time_offset << "\n";

    return std::make_pair(*mocap_time_offset, T_bm);
}

private:

  // this will only work for se3 splines. Will need something different for so3xr3
  virtual void add_motion_priors(ceres::Problem& problem) override
  {
    mp_.reset(new LieSplineMotionPrior<SplineT>(mp_type_, this->Qc_));

    double t1 = this->start_time_;
    for(double t2 = t1 + mp_dt_; t2 <= this->end_time_; t2 += mp_dt_)
    {
      add_motion_prior(t1, t2, problem);
      t1 = t2;
    }
    // need to add one more right at the end time
    if(t1 < this->end_time_) add_motion_prior(t1, this->end_time_, problem);
  }

  void add_motion_prior(double t1, double t2, ceres::Problem& problem)
  {
    ceres::CostFunction* r;
    if(mp_type_ == ModelType::JERK)
    {
      Eigen::Matrix<double, 18, 18> Q_inv = mp_->template get_Q_inv<18>(t1, t2);
      Eigen::LLT<Eigen::Matrix<double, 18, 18>> chol(Q_inv);
      r = new estimator::SplineDynamicsResidual<SplineT, LieSplineMotionPrior<SplineT>, 18, 7, 6>(t1, t2, chol.matrixL().transpose(), mp_, this->spline_);
    }
    else
    {
      Eigen::Matrix<double, 12, 12> Q_inv = mp_->template get_Q_inv<12>(t1, t2);
      Eigen::LLT<Eigen::Matrix<double, 12, 12>> chol(Q_inv);
      r = new estimator::SplineDynamicsResidual<SplineT, LieSplineMotionPrior<SplineT>, 12, 7, 6>(t1, t2, chol.matrixL().transpose(), mp_, this->spline_);
    }

    std::vector<double*> params;
    int i1 = this->spline_->get_i(t1);
    for(int j = 0; j < this->k_; ++j) params.push_back(this->ctrl_pts_->at(i1 + j + 1).data());
    int i2 = this->spline_->get_i(t2);
    for(int j = 0; j < this->k_; ++j) if(i2 + j - i1 >= this->k_) params.push_back(this->ctrl_pts_->at(i2 + j + 1).data());

    problem.AddResidualBlock(r, nullptr, params);
  }

  std::shared_ptr<LieSplineMotionPrior<SplineT>> mp_;
  ModelType mp_type_;
  double mp_dt_;
};

// specialized version for so3xr3 splines
template <typename G>
class SplineEstimator<spline::SOnxRnSpline<G>> : public SplineEstimatorBase<spline::SOnxRnSpline<G>>
{
private:
  using Matrix6d = Eigen::Matrix<double, 6, 6>;
  using SOnSplineT = spline::LieSpline<typename G::SOnT>;
  using RnSplineT = spline::RnSpline<typename G::RnT, 3>;

public:
  SplineEstimator(EstimatorParams est_params) : SplineEstimatorBase<spline::SOnxRnSpline<G>>(est_params)
  {}

  void set_up_motion_priors(ModelType son_type, ModelType rn_type, double son_mp_dt, double rn_mp_dt, Matrix6d Qc)
  {
    son_type_ = son_type;
    rn_type_ = rn_type;
    son_mp_dt_ = son_mp_dt;
    rn_mp_dt_ = rn_mp_dt;
    this->Qc_ = Qc;

    this->motion_priors_set_ = true;
  }

private:
  virtual void add_motion_priors(ceres::Problem& problem)
  {
    son_mp_.reset(new LieSplineMotionPrior<SOnSplineT>(son_type_, this->Qc_.template block<3,3>(3,3)));
    rn_mp_.reset(new RnSplineMotionPrior<RnSplineT, 3>(rn_type_, this->Qc_.template block<3,3>(0,0)));

    // first add son priors
    double t1 = this->start_time_;
    for(double t2 = t1 + son_mp_dt_; t2 <= this->end_time_; t2 += son_mp_dt_)
    {
      add_son_motion_prior(t1, t2, problem);
      t1 = t2;
    }
    // need to add one more right at the end time
    if(t1 < this->end_time_) add_son_motion_prior(t1, this->end_time_, problem);

    // now add rn priors
    t1 = this->start_time_;
    for(double t2 = t1 + rn_mp_dt_; t2 <= this->end_time_; t2 += rn_mp_dt_)
    {
      add_rn_motion_prior(t1, t2, problem);
      t1 = t2;
    }
    // need to add one more right at the end time
    if(t1 < this->end_time_) add_rn_motion_prior(t1, this->end_time_, problem);
  }

  void add_son_motion_prior(double t1, double t2, ceres::Problem& problem)
  {
    ceres::CostFunction* r_son;
    if(son_type_ == ModelType::JERK)
    {
      Eigen::Matrix<double, 9, 9> Q_inv = son_mp_->template get_Q_inv<9>(t1, t2);
      Eigen::LLT<Eigen::Matrix<double, 9, 9>> chol(Q_inv);
      r_son = new estimator::SOnxRnSplineDynamicsResidual<SOnSplineT, LieSplineMotionPrior<SOnSplineT>, 9, 7, 3>(t1, t2, chol.matrixL().transpose(), son_mp_, this->spline_->get_son_spline(), true);
    }
    else
    {
      Eigen::Matrix<double, 6, 6> Q_inv = son_mp_->template get_Q_inv<6>(t1, t2);
      Eigen::LLT<Eigen::Matrix<double, 6, 6>> chol(Q_inv);
      r_son = new estimator::SOnxRnSplineDynamicsResidual<SOnSplineT, LieSplineMotionPrior<SOnSplineT>, 6, 7, 3>(t1, t2, chol.matrixL().transpose(), son_mp_, this->spline_->get_son_spline(), true);
    }

    std::vector<double*> params;
    int i1 = this->spline_->get_i(t1);
    for(int j = 0; j < this->k_; ++j) params.push_back(this->ctrl_pts_->at(i1 + j + 1).data());
    int i2 = this->spline_->get_i(t2);
    for(int j = 0; j < this->k_; ++j) if(i2 + j - i1 >= this->k_) params.push_back(this->ctrl_pts_->at(i2 + j + 1).data());

    problem.AddResidualBlock(r_son, nullptr, params);
  }

  void add_rn_motion_prior(double t1, double t2, ceres::Problem& problem)
  {
    ceres::CostFunction* r_rn;
    if(rn_type_ == ModelType::JERK)
    {
      Eigen::Matrix<double, 9, 9> Q_inv = rn_mp_->template get_Q_inv<9>(t1, t2);
      Eigen::LLT<Eigen::Matrix<double, 9, 9>> chol(Q_inv);
      r_rn = new estimator::SOnxRnSplineDynamicsResidual<RnSplineT, RnSplineMotionPrior<RnSplineT, 3>, 9, 7, 3>(t1, t2, chol.matrixL().transpose(), rn_mp_, this->spline_->get_rn_spline(), false);
    }
    else
    {
      Eigen::Matrix<double, 6, 6> Q_inv = rn_mp_->template get_Q_inv<6>(t1, t2);
      Eigen::LLT<Eigen::Matrix<double, 6, 6>> chol(Q_inv);
      r_rn = new estimator::SOnxRnSplineDynamicsResidual<RnSplineT, RnSplineMotionPrior<RnSplineT, 3>, 6, 7, 3>(t1, t2, chol.matrixL().transpose(), rn_mp_, this->spline_->get_rn_spline(), false);
    }

    std::vector<double*> params;
    int i1 = this->spline_->get_i(t1);
    for(int j = 0; j < this->k_; ++j) params.push_back(this->ctrl_pts_->at(i1 + j + 1).data());
    int i2 = this->spline_->get_i(t2);
    for(int j = 0; j < this->k_; ++j) if(i2 + j - i1 >= this->k_) params.push_back(this->ctrl_pts_->at(i2 + j + 1).data());

    problem.AddResidualBlock(r_rn, nullptr, params);
  }

  std::shared_ptr<LieSplineMotionPrior<SOnSplineT>> son_mp_;
  std::shared_ptr<RnSplineMotionPrior<RnSplineT, 3>> rn_mp_;
  ModelType rn_type_;
  ModelType son_type_;
  double rn_mp_dt_;
  double son_mp_dt_;
};

} // namespace estimator