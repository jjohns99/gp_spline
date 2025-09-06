#pragma once

#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <ceres/ceres.h>

#include "spline/rn_spline.hpp"
#include "linear_example/spline_pos_residual.hpp"
#include "residuals/dummy_callback.h"
#include "estimator/rn_spline_motion_prior.hpp"
#include "residuals/spline_dynamics_residual.hpp"

namespace linear_example
{

// double or triple integrator model in DoF dimensions with position measurements,
// estimated using a spline in R^DoF
template <int DoF>
class SplineLinEst
{
private:
  using VectorDoF = Eigen::Matrix<double, DoF, 1>;
  using MatrixDoF = Eigen::Matrix<double, DoF, DoF>;
  using SplineT = spline::RnSpline<Eigen::Map<VectorDoF>, DoF>;
  using MP = estimator::RnSplineMotionPrior<SplineT, DoF>;

public:
  SplineLinEst(int k, double dt, bool use_motion_priors = false, double motion_prior_dt = 0.0, estimator::ModelType motion_type = estimator::ModelType::ACC, MatrixDoF Qc = MatrixDoF::Identity()) : 
    k_{k}, dt_{dt}, use_motion_priors_{use_motion_priors}, motion_prior_dt_{motion_prior_dt}, motion_type_{motion_type}, Qc_{Qc}
  {}

  void add_pos_measurements(std::shared_ptr<std::vector<std::pair<double, VectorDoF>>> z_pos, MatrixDoF pos_sqrt_W)
  {
    z_pos_ = z_pos;
    pos_sqrt_W_ = pos_sqrt_W;
  }

  ceres::Solver::Summary solve(bool print_est_progress)
  {
    // instantiate spline
    spline_.reset(new SplineT(k_));

    double start_time = z_pos_->at(0).first;
    double end_time = z_pos_->back().first;

    // initialize control points
    ctrl_pts_.reset(new std::vector<Eigen::Map<VectorDoF>>);
    int num_cp = static_cast<int>(std::ceil((end_time - start_time)/dt_)) + k_;
    for(int i = 0; i < num_cp; ++i) ctrl_pts_->emplace_back(new double[DoF]{});
    spline_->init_ctrl_pts(ctrl_pts_, start_time, dt_);

    // create problem
    ceres::Problem::Options problem_options;
    dummy_callback_.reset(new estimator::DummyCallback()); // add dummy callback to force ceres to write to original data while iterating
    problem_options.evaluation_callback = dummy_callback_.get();
    problem_ = ceres::Problem(problem_options);

    // add measurements to problem
    for(auto const& z : *z_pos_)
    {
      ceres::CostFunction* r = new SplinePosResidual<DoF>(z.first, z.second, pos_sqrt_W_, spline_);

      int i = spline_->get_i(z.first);
      std::vector<double*> params;
      for(int j = 0; j < k_; ++j) params.push_back(ctrl_pts_->at(i + j + 1).data());
      
      pos_resid_ids_.push_back(problem_.AddResidualBlock(r, nullptr, params));
    }

    // add motion priors to problem
    if(use_motion_priors_)
    {
      mp_.reset(new MP(motion_type_, Qc_));

      double t1 = start_time;
      for(double t2 = t1 + motion_prior_dt_; t2 <= end_time; t2 += motion_prior_dt_)
      {
        add_motion_prior(t1, t2);
        t1 = t2;
      }
      // need to add one more right at the end time
      add_motion_prior(t1, end_time);
    }

    // set solver options
    ceres::Solver::Options solver_options;
    solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    solver_options.initial_trust_region_radius = 1e30;
    solver_options.max_trust_region_radius = 1e30;
    solver_options.minimizer_progress_to_stdout = print_est_progress;
    solver_options.num_threads = 1;
    solver_options.max_num_iterations = 1;
    solver_options.update_state_every_iteration = true; // may not need dummy callback if we have this

    // solve. this should only need one iteration (problem is linear least squares)
    ceres::Solver::Summary summary;
    ceres::Solve(solver_options, &problem_, &summary);    
    if(print_est_progress) std::cout << summary.FullReport();
    return summary;
  }

  void add_motion_prior(double t1, double t2)
  {
    ceres::CostFunction* r;
    if(motion_type_ == estimator::ModelType::ACC)
    {
      // Eigen::Matrix<double, 2*DoF, 2*DoF> Phi = mp.template get_Phi<2*DoF>(t1, t2);
      Eigen::Matrix<double, 2*DoF, 2*DoF> Q_inv = mp_->template get_Q_inv<2*DoF>(t1, t2);
      Eigen::LLT<Eigen::Matrix<double, 2*DoF, 2*DoF>> chol(Q_inv);
      r = new estimator::SplineDynamicsResidual<SplineT, MP, 2*DoF, DoF, DoF>(t1, t2, chol.matrixL().transpose(), mp_, spline_);
    }
    else
    { 
      // Eigen::Matrix<double, 3*DoF, 3*DoF> Phi = mp.template get_Phi<3*DoF>(t1, t2);
      Eigen::Matrix<double, 3*DoF, 3*DoF> Q_inv = mp_->template get_Q_inv<3*DoF>(t1, t2);
      Eigen::LLT<Eigen::Matrix<double, 3*DoF, 3*DoF>> chol(Q_inv);
      r = new estimator::SplineDynamicsResidual<SplineT, MP, 3*DoF, DoF, DoF>(t1, t2, chol.matrixL().transpose(), mp_, spline_);
    }

    std::vector<double*> params;
    int i1 = spline_->get_i(t1);
    for(int j = 0; j < k_; ++j) params.push_back(ctrl_pts_->at(i1 + j + 1).data());
    int i2 = spline_->get_i(t2);
    for(int j = 0; j < k_; ++j) if(i2 + j - i1 >= k_) params.push_back(ctrl_pts_->at(i2 + j + 1).data());

    mp_resid_ids_.push_back(problem_.AddResidualBlock(r, nullptr, params));
  }

  std::shared_ptr<SplineT> get_spline()
  {
    return spline_;
  }

  bool get_pos_resid_cost(double &cost)
  {
    if(!pos_resid_ids_.size()) return false;
    ceres::Problem::EvaluateOptions eval_options;
    eval_options.residual_blocks = pos_resid_ids_;
    cost = 0.0;
    problem_.Evaluate(eval_options, &cost, nullptr, nullptr, nullptr);
    return true;
  }

  bool get_mp_resid_cost(double &cost)
  {
    if(!mp_resid_ids_.size()) return false;
    ceres::Problem::EvaluateOptions eval_options;
    eval_options.residual_blocks = mp_resid_ids_;
    cost = 0.0;
    problem_.Evaluate(eval_options, &cost, nullptr, nullptr, nullptr);
    return true;
  }

private:
  std::shared_ptr<SplineT> spline_;
  std::shared_ptr<std::vector<Eigen::Map<VectorDoF>>> ctrl_pts_;

  std::shared_ptr<MP> mp_;

  std::shared_ptr<std::vector<std::pair<double, VectorDoF>>> z_pos_;
  MatrixDoF pos_sqrt_W_;

  int k_;
  double dt_;

  std::shared_ptr<estimator::DummyCallback> dummy_callback_;

  bool use_motion_priors_;
  double motion_prior_dt_;
  estimator::ModelType motion_type_;
  MatrixDoF Qc_;

  ceres::Problem problem_;

  // save residual ids for evalutation after solving
  std::vector<ceres::ResidualBlockId> pos_resid_ids_;
  std::vector<ceres::ResidualBlockId> mp_resid_ids_;
};

} // namespace example