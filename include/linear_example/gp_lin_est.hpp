#pragma once

#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <ceres/ceres.h>

#include "gp/rn_gp.hpp"
#include "linear_example/gp_pos_residual.hpp"
#include "linear_example/pos_residual.hpp"
#include "residuals/gp_dynamics_residual.hpp"
#include "residuals/dummy_callback.h"

namespace linear_example
{

// double or triple integrator model in DoF dimensions with position measurements,
// estimated using a GP in R^DoF
template <int DoF>
class GPLinEst
{
private:
  using VectorDoF = Eigen::Matrix<double, DoF, 1>;
  using MatrixDoF = Eigen::Matrix<double, DoF, DoF>;

public:
  GPLinEst(MatrixDoF Q, gp::ModelType type, int n, bool use_motion_priors = true) : Q_{Q}, type_{type}, n_{n}, use_mp_{use_motion_priors}
  {}

  void add_pos_measurements(std::shared_ptr<std::vector<std::pair<double, VectorDoF>>> z_pos, MatrixDoF pos_sqrt_W)
  {
    z_pos_ = z_pos;
    pos_sqrt_W_ = pos_sqrt_W;
  }

  ceres::Solver::Summary solve(bool print_est_progress)
  {
    // instantiate gp
    gp_.reset(new gp::RnGP<Eigen::Map<VectorDoF>, DoF>(type_, Q_));

    // instantiate estimation parameters
    p_.reset(new std::vector<Eigen::Map<VectorDoF>>);
    v_.reset(new std::vector<Eigen::Map<VectorDoF>>);
    if(type_ == gp::ModelType::JERK) a_.reset(new std::vector<Eigen::Map<VectorDoF>>);
    std::vector<double> t_est;
    for(int i = 0; i < z_pos_->size(); ++i)
    {
      if(!(i % n_))
      {
        t_est.push_back(z_pos_->at(i).first);
        p_->emplace_back(new double[DoF]{});
        v_->emplace_back(new double[DoF]{});
        if(type_ == gp::ModelType::JERK) a_->emplace_back(new double[DoF]{});
      }
    }
    gp_->init_est_params(t_est, p_, v_, (type_ == gp::ModelType::JERK ? a_ : nullptr));
    
    // create problem
    ceres::Problem::Options problem_options;
    dummy_callback_.reset(new estimator::DummyCallback()); // add dummy callback to force ceres to write to original data while iterating
    problem_options.evaluation_callback = dummy_callback_.get();
    problem_ = ceres::Problem(problem_options);

    // add position measurements to problem
    for(int i = 0; i < z_pos_->size(); ++i)
    {
      if(!gp_->is_time_valid(z_pos_->at(i).first)) continue;

      if(!(i % n_))
      {
        ceres::CostFunction* r = new PosResidual<DoF>(z_pos_->at(i).first, z_pos_->at(i).second, pos_sqrt_W_);
        std::vector<double*> params;
        params.push_back(p_->at(i/n_).data());

        pos_resid_ids_.push_back(problem_.AddResidualBlock(r, nullptr, params));
      }
      else
      {
        ceres::CostFunction* r = new GPPosResidual<DoF>(z_pos_->at(i).first, z_pos_->at(i).second, pos_sqrt_W_, gp_);

        int i_gp = gp_->get_i(z_pos_->at(i).first);
        std::vector<double*> params;
        for(int j = 0; j < 2; ++j) params.push_back(p_->at(i_gp+j).data());
        for(int j = 0; j < 2; ++j) params.push_back(v_->at(i_gp+j).data());
        if(type_ == gp::ModelType::JERK) for(int j = 0; j < 2; ++j) params.push_back(a_->at(i_gp+j).data());

        pos_resid_ids_.push_back(problem_.AddResidualBlock(r, nullptr, params));
      }
    }

    // add motion priors to problem
    if(use_mp_)
    {
      for(int i = 0; i < p_->size()-1; ++i)
      {
        ceres::CostFunction* r;
        if(type_ == gp::ModelType::ACC)
        {
          Eigen::Matrix<double, 2*DoF, 2*DoF> Q_i_inv = gp_->get_Q_inv_i_i1(i);
          Eigen::LLT<Eigen::Matrix<double, 2*DoF, 2*DoF>> chol(Q_i_inv);
          Eigen::Matrix<double, 2*DoF, 2*DoF> sqrt_W_dyn = chol.matrixL().transpose();

          r = new estimator::GPDynamicsResidual<gp::RnGP<Eigen::Map<VectorDoF>, DoF>, 2*DoF, DoF, DoF>(sqrt_W_dyn, gp_, i);
        }
        else
        {
          Eigen::Matrix<double, 3*DoF, 3*DoF> Q_i_inv = gp_->get_Q_inv_i_i1(i);
          Eigen::LLT<Eigen::Matrix<double, 3*DoF, 3*DoF>> chol(Q_i_inv);
          Eigen::Matrix<double, 3*DoF, 3*DoF> sqrt_W_dyn = chol.matrixL().transpose();

          r = new estimator::GPDynamicsResidual<gp::RnGP<Eigen::Map<VectorDoF>, DoF>, 3*DoF, DoF, DoF>(sqrt_W_dyn, gp_, i);
        }

        std::vector<double*> params;
        for(int j = 0; j < 2; ++j) params.push_back(p_->at(i+j).data());
        for(int j = 0; j < 2; ++j) params.push_back(v_->at(i+j).data());
        if(type_ == gp::ModelType::JERK) for(int j = 0; j < 2; ++j) params.push_back(a_->at(i+j).data());

        mp_resid_ids_.push_back(problem_.AddResidualBlock(r, nullptr, params));
      }
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

  std::shared_ptr<gp::RnGP<Eigen::Map<VectorDoF>, DoF>> get_gp()
  {
    return gp_;
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
  std::shared_ptr<gp::RnGP<Eigen::Map<VectorDoF>, DoF>> gp_;
  std::shared_ptr<std::vector<Eigen::Map<VectorDoF>>> p_;
  std::shared_ptr<std::vector<Eigen::Map<VectorDoF>>> v_;
  std::shared_ptr<std::vector<Eigen::Map<VectorDoF>>> a_;
  
  std::shared_ptr<std::vector<std::pair<double, VectorDoF>>> z_pos_;
  MatrixDoF pos_sqrt_W_;

  MatrixDoF Q_;
  gp::ModelType type_;
  int n_; // every nth measurement will be an estimation time
  bool use_mp_; // choose whether to use motion priors

  std::shared_ptr<estimator::DummyCallback> dummy_callback_;

  ceres::Problem problem_;

  // save residual ids for evalutation after solving
  std::vector<ceres::ResidualBlockId> pos_resid_ids_;
  std::vector<ceres::ResidualBlockId> mp_resid_ids_;
};

} // linear_example