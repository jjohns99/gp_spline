#include <vector>
#include <memory>
#include <cmath>
#include <limits>
#include <Eigen/Dense>
#include <ceres/ceres.h>

#include "utils/multivariate_gaussian.hpp"
#include "lie_groups/so2.hpp"
#include "lie_groups/se2.hpp"
#include "lie_groups/so3.hpp"
#include "lie_groups/se3.hpp"
#include "spline/lie_spline.hpp"
#include "parameterizations/sen_parameterization.hpp"
#include "residuals/spline_dynamics_residual.hpp"
#include "residuals/dummy_callback.h"
#include "estimator/lie_spline_motion_prior.hpp"

template <typename G>
class GResidual : public ceres::CostFunction
{
private:
  static const int DoF = G::DoF;
  using VectorDoF = Eigen::Matrix<double, DoF, 1>;
  using MatrixDoF = Eigen::Matrix<double, DoF, DoF>; 
  using SplineT = spline::LieSpline<lie_groups::Map<G>>;

public:
  GResidual(double t, G z, MatrixDoF sqrt_W, std::shared_ptr<SplineT> spl) :
    t_{t}, z_{z}, sqrt_W_{sqrt_W}, spline_{spl}
  {
    set_num_residuals(DoF);

    std::vector<int32_t> param_sizes;
    for(int i = 0; i < spline_->get_order(); ++i) param_sizes.push_back(G::MEM);
    *mutable_parameter_block_sizes() = param_sizes;
  }

  virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
  {
    Eigen::Map<VectorDoF> r(residuals);

    G g;
    std::vector<MatrixDoF> jacs;
    spline_->eval(t_, &g, nullptr, nullptr, (jacobians == nullptr) ? nullptr : &jacs);
    r = sqrt_W_ * (g * z_.inverse()).Log();

    if(jacobians != nullptr)
    {
      for(int i = 0; i < jacs.size(); ++i)
      {
        Eigen::Map<Eigen::Matrix<double, DoF, G::MEM, Eigen::RowMajor>> J(jacobians[i]);
        J.setZero();
        J.template block<DoF, DoF>(0,0) = sqrt_W_ * jacs[i];
      }
    }
    return true;
  }

private:
  double t_;
  G z_;
  MatrixDoF sqrt_W_;
  std::shared_ptr<SplineT> spline_;
};

// optimizer with G measurements and motion priors
template <typename G>
class LieSplineOptimizer
{
private:
  using SplineT = spline::LieSpline<lie_groups::Map<G>>;
  using MPT = estimator::LieSplineMotionPrior<SplineT>;
  static const int DoF = G::DoF;
  using VectorDoF = Eigen::Matrix<double, DoF, 1>;
  using MatrixDoF = Eigen::Matrix<double, DoF, DoF>;

public:
  LieSplineOptimizer(std::vector<std::pair<double, G>> meas, MatrixDoF cov) : measurements_{meas},
    start_time_{meas[0].first}, end_time_{meas.back().first}, h_{1e-6}
  {
    Eigen::LLT<MatrixDoF> chol(cov.inverse());
    meas_sqrt_W_ = chol.matrixL().transpose();
  }

  void init_spline(double dt, int k, std::shared_ptr<std::vector<lie_groups::Map<G>>> ctrl_pts)
  {
    k_ = k;
    ctrl_pts_ = ctrl_pts;

    spline_.reset(new SplineT(k));
    spline_->init_ctrl_pts(ctrl_pts_, start_time_, dt);
  }

  void set_up_motion_priors(estimator::ModelType mp_type, double mp_dt, MatrixDoF Qc)
  {
    mp_type_ = mp_type;
    mp_dt_ = mp_dt;
    Qc_ = Qc;

    mp_.reset(new MPT(mp_type_, Qc_));
  }

  ceres::Solver::Summary solve(bool use_priors = true, bool check_mp_jacs = false)
  {
    ceres::Problem::Options problem_options;
    dummy_cb_.reset(new estimator::DummyCallback());
    problem_options.evaluation_callback = dummy_cb_.get();

    ceres::Problem problem(problem_options);

    for(auto& ctrl_pt : *ctrl_pts_)
    {
      problem.AddParameterBlock(ctrl_pt.data(), G::MEM, new SEnParameterization<typename G::NonmapT>());
    }

    for(int ind = 0; ind < measurements_.size(); ++ind)
    {
      double t = measurements_[ind].first;
      int i = spline_->get_i(t);
      std::vector<double*> params;
      for(int j = 0; j < k_; ++j) params.push_back(ctrl_pts_->at(i + 1 + j).data());

      ceres::CostFunction* r_g = new GResidual<G>(measurements_[ind].first, measurements_[ind].second, meas_sqrt_W_, spline_);
      problem.AddResidualBlock(r_g, nullptr, params);
    }

    if(use_priors)
    {
      mp_resids_.clear();
      double t1 = start_time_;
      for(double t2 = t1 + mp_dt_; t2 <= end_time_; t2 += mp_dt_)
      {
        add_motion_prior(t1, t2, problem);
        t1 = t2;
      }
      // need to add one more right at the end time
      if(t1 < end_time_) add_motion_prior(t1, end_time_, problem);

      // check jacobians of these motion priors at current control point values
      if(check_mp_jacs) 
      {
        if(mp_type_ == estimator::ModelType::ACC) check_motion_prior_jacs<2*DoF>(problem);
        else check_motion_prior_jacs<3*DoF>(problem);
      }
    }

    ceres::Solver::Options solver_options;
    solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    solver_options.minimizer_progress_to_stdout = true;
    solver_options.num_threads = 1;
    solver_options.max_num_iterations = 200;
    solver_options.check_gradients = false;
    solver_options.update_state_every_iteration = true;
    solver_options.initial_trust_region_radius = 1e4;

    ceres::Solver::Summary summary;
    ceres::Solve(solver_options, &problem, &summary);    
    std::cout << summary.FullReport();
    return summary;
  }

private:
  void add_motion_prior(double t1, double t2, ceres::Problem& problem)
  {
    ceres::CostFunction* r;
    if(mp_type_ == estimator::ModelType::JERK)
    {
      Eigen::Matrix<double, 3*DoF, 3*DoF> Q_inv = mp_->template get_Q_inv<3*DoF>(t1, t2);
      Eigen::LLT<Eigen::Matrix<double, 3*DoF, 3*DoF>> chol(Q_inv);
      r = new estimator::SplineDynamicsResidual<SplineT, MPT, 3*DoF, G::MEM, DoF>(t1, t2, chol.matrixL().transpose(), mp_, spline_);
    }
    else
    {
      Eigen::Matrix<double, 2*DoF, 2*DoF> Q_inv = mp_->template get_Q_inv<2*DoF>(t1, t2);
      Eigen::LLT<Eigen::Matrix<double, 2*DoF, 2*DoF>> chol(Q_inv);
      r = new estimator::SplineDynamicsResidual<SplineT, MPT, 2*DoF, G::MEM, DoF>(t1, t2, chol.matrixL().transpose(), mp_, spline_);
    }

    std::vector<double*> params;
    int i1 = spline_->get_i(t1);
    for(int j = 0; j < k_; ++j) params.push_back(ctrl_pts_->at(i1 + j + 1).data());
    int i2 = spline_->get_i(t2);
    for(int j = 0; j < k_; ++j) if(i2 + j - i1 >= k_) params.push_back(ctrl_pts_->at(i2 + j + 1).data());

    mp_resids_.push_back(problem.AddResidualBlock(r, nullptr, params));
  }

  template <int S>
  void check_motion_prior_jacs(ceres::Problem& problem)
  {
    for(auto const& id : mp_resids_)
    {
      std::vector<double*> params;
      problem.GetParameterBlocksForResidualBlock(id, &params);
      int num_params = params.size();

      std::vector<Eigen::Map<Eigen::Matrix<double, S, DoF, (DoF == 1 ? Eigen::ColMajor : Eigen::RowMajor)>>> eval_jacs;
      double** eval_jac_data = new double*[num_params];
      for(int i = 0; i < num_params; ++i) 
      {
        eval_jac_data[i] = new double[S*DoF]{0.0};
        eval_jacs.emplace_back(eval_jac_data[i]);
      }

      double r_unpert_data[S];
      Eigen::Map<Eigen::Matrix<double, S, 1>> r_unpert(r_unpert_data);
      double cost;
      problem.EvaluateResidualBlock(id, false, &cost, r_unpert_data, eval_jac_data);

      std::vector<Eigen::Matrix<double, S, DoF>> fin_diff_jacs;
      for(int i = 0; i < num_params; ++i)
      {
        Eigen::Matrix<double, S, DoF> fin_diff_jac;
        fin_diff_jac.setZero();

        for(int d = 0; d < DoF; ++d)
        {
          double param_copy[DoF];
          std::copy(params[i], params[i]+DoF, param_copy);

          double pert[DoF]{0.0};
          pert[d] = h_;
        
          // perturb the control point data
          const ceres::LocalParameterization* param_updater = problem.GetParameterization(params[i]);
          double param_pert_data[DoF]{0.0};
          param_updater->Plus(params[i], pert, param_pert_data);
          std::copy(param_pert_data, param_pert_data+DoF, params[i]);

          // approximate directional derivative
          double r_pert_data[S];
          problem.EvaluateResidualBlock(id, false, &cost, r_pert_data, nullptr);
          Eigen::Map<Eigen::Matrix<double, S, 1>> r_pert(r_pert_data);
          fin_diff_jac.col(d) = (r_pert - r_unpert) / h_;

          // unperturb the control point data
          std::copy(param_copy, param_copy+DoF, params[i]);
        }

        fin_diff_jacs.push_back(fin_diff_jac);
      }

      // check that fin_diff_jacs is close to eval_jacs
      for(int i = 0; i < num_params; ++i)
      {
        std::cout << "Eval jac:\n" << eval_jacs[i] << "\n\n";
        std::cout << "Fin diff jac:\n" << fin_diff_jacs[i] << "\n\n";
      }

      for(int i = 0; i < num_params; ++i) delete[] eval_jac_data[i];
      delete[] eval_jac_data;
    }
  }

  std::vector<std::pair<double, G>> measurements_;
  MatrixDoF meas_sqrt_W_;
  double start_time_;
  double end_time_;

  std::shared_ptr<std::vector<lie_groups::Map<G>>> ctrl_pts_;
  std::shared_ptr<SplineT> spline_;
  int k_;

  std::shared_ptr<MPT> mp_;
  estimator::ModelType mp_type_;
  double mp_dt_;
  MatrixDoF Qc_;

  std::shared_ptr<estimator::DummyCallback> dummy_cb_;

  std::vector<ceres::ResidualBlockId> mp_resids_;
  double h_;
};

template <typename G>
void simulate_trajectory(double meas_stdev, double traj_stdev, std::vector<double>& t_true, std::vector<G>& g_true, std::vector<Eigen::Matrix<double, G::DoF, 1>>& v_true, std::vector<std::pair<double, G>>& meas)
{
  using VectorDoF = Eigen::Matrix<double, G::DoF, 1>;
  using MatrixDoF = Eigen::Matrix<double, G::DoF, G::DoF>;

  utils::MultivariateGaussian<G::DoF> traj_noise(VectorDoF::Zero(), std::pow(traj_stdev, 2.0) * MatrixDoF::Identity());
  utils::MultivariateGaussian<G::DoF> meas_noise(VectorDoF::Zero(), std::pow(meas_stdev, 2.0) * MatrixDoF::Identity());

  G g_init = G();
  VectorDoF v_init = VectorDoF::Zero();

  double sim_dt = 0.01;
  double meas_dt = 0.1;
  bool start = true;
  double last_meas_time = -10.0;
  for(double sim_time = 0.0; sim_time < 10.0; sim_time += sim_dt)
  {
    G g_prev;
    VectorDoF v_prev;
    if(start)
    {
      g_prev = g_init;
      v_prev = v_init;
      start = false;
    }
    else
    {
      g_prev = g_true.back();
      v_prev = v_true.back();
    }
    
    v_true.push_back(v_prev + sim_dt*traj_noise.sample());
    g_true.push_back(G::Exp(sim_dt * v_true.back()) * g_prev);
    t_true.push_back(sim_time);

    if(sim_time - last_meas_time > meas_dt)
    {
      meas.push_back(std::make_pair(sim_time, G::Exp(meas_noise.sample()) * g_true.back()));
      last_meas_time = sim_time - 1e-6;
    }
  }
}

int main()
{
  // so2
  double so2_meas_stdev = 0.1;
  double so2_traj_stdev = 0.2;
  int so2_k = 4;
  double so2_dt = 0.1;

  std::vector<double> so2_t_traj;
  std::vector<lie_groups::SO2d> so2_traj;
  std::vector<Eigen::Matrix<double, 1, 1>> so2_v_traj;
  std::vector<std::pair<double, lie_groups::SO2d>> so2_meas;
  simulate_trajectory<lie_groups::SO2d>(so2_meas_stdev, so2_traj_stdev, so2_t_traj, so2_traj, so2_v_traj, so2_meas);

  LieSplineOptimizer<lie_groups::SO2d> so2_est(so2_meas, std::pow(so2_meas_stdev, 2.0) * Eigen::Matrix<double, 1, 1>::Identity());
  so2_est.set_up_motion_priors(estimator::ModelType::ACC, 2*so2_dt, std::pow(so2_traj_stdev, 2.0) * Eigen::Matrix<double, 1, 1>::Identity());

  std::shared_ptr<std::vector<lie_groups::Map<lie_groups::SO2d>>> so2_ctrl_pts(new std::vector<lie_groups::Map<lie_groups::SO2d>>());
  int num_so2_cp = static_cast<int>(std::ceil((so2_meas.back().first - so2_meas[0].first)/so2_dt)) + so2_k;
  for(int i = 0; i < num_so2_cp; ++i)
    so2_ctrl_pts->emplace_back(new double[1]{0.0});
  so2_est.init_spline(so2_dt, so2_k, so2_ctrl_pts);

  std::cout << "Solving for SO(2) trajectory \n\n";
  so2_est.solve(true);
  // so2_est.solve(true);

  // so3
  double so3_meas_stdev = 0.1;
  double so3_traj_stdev = 0.2;
  int so3_k = 4;
  double so3_dt = 0.1;

  std::vector<double> so3_t_traj;
  std::vector<lie_groups::SO3d> so3_traj;
  std::vector<Eigen::Matrix<double, 3, 1>> so3_v_traj;
  std::vector<std::pair<double, lie_groups::SO3d>> so3_meas;
  simulate_trajectory<lie_groups::SO3d>(so3_meas_stdev, so3_traj_stdev, so3_t_traj, so3_traj, so3_v_traj, so3_meas);

  LieSplineOptimizer<lie_groups::SO3d> so3_est(so3_meas, std::pow(so3_meas_stdev, 2.0) * Eigen::Matrix3d::Identity());
  so3_est.set_up_motion_priors(estimator::ModelType::ACC, 2*so3_dt, std::pow(so3_traj_stdev, 2.0) * Eigen::Matrix3d::Identity());

  std::shared_ptr<std::vector<lie_groups::Map<lie_groups::SO3d>>> so3_ctrl_pts(new std::vector<lie_groups::Map<lie_groups::SO3d>>());
  int num_so3_cp = static_cast<int>(std::ceil((so3_meas.back().first - so3_meas[0].first)/so3_dt)) + so3_k;
  for(int i = 0; i < num_so3_cp; ++i)
    so3_ctrl_pts->emplace_back(new double[4]{0.0, 0.0, 0.0, 1.0});
  so3_est.init_spline(so3_dt, so3_k, so3_ctrl_pts);

  std::cout << "Solving for SO(3) trajectory \n\n";
  so3_est.solve(true);
  // so3_est.solve(true);

  // se2
  double se2_meas_stdev = 0.1;
  double se2_traj_stdev = 0.2;
  int se2_k = 4;
  double se2_dt = 0.1;

  std::vector<double> se2_t_traj;
  std::vector<lie_groups::SE2d> se2_traj;
  std::vector<Eigen::Matrix<double, 3, 1>> se2_v_traj;
  std::vector<std::pair<double, lie_groups::SE2d>> se2_meas;
  simulate_trajectory<lie_groups::SE2d>(se2_meas_stdev, se2_traj_stdev, se2_t_traj, se2_traj, se2_v_traj, se2_meas);

  LieSplineOptimizer<lie_groups::SE2d> se2_est(se2_meas, std::pow(se2_meas_stdev, 2.0) * Eigen::Matrix3d::Identity());
  se2_est.set_up_motion_priors(estimator::ModelType::ACC, 2*se2_dt, std::pow(se2_traj_stdev, 2.0) * Eigen::Matrix3d::Identity());

  std::shared_ptr<std::vector<lie_groups::Map<lie_groups::SE2d>>> se2_ctrl_pts(new std::vector<lie_groups::Map<lie_groups::SE2d>>());
  int num_se2_cp = static_cast<int>(std::ceil((se2_meas.back().first - se2_meas[0].first)/se2_dt)) + se2_k;
  for(int i = 0; i < num_se2_cp; ++i)
    se2_ctrl_pts->emplace_back(new double[3]{0.0, 0.0, 0.0});
  se2_est.init_spline(se2_dt, se2_k, se2_ctrl_pts);

  std::cout << "Solving for SE(2) trajectory \n\n";
  se2_est.solve(true);
  // se2_est.solve(true);

  // se3
  double se3_meas_stdev = 0.1;
  double se3_traj_stdev = 0.2;
  int se3_k = 4;
  double se3_dt = 0.1;

  std::vector<double> se3_t_traj;
  std::vector<lie_groups::SE3d> se3_traj;
  std::vector<Eigen::Matrix<double, 6, 1>> se3_v_traj;
  std::vector<std::pair<double, lie_groups::SE3d>> se3_meas;
  simulate_trajectory<lie_groups::SE3d>(se3_meas_stdev, se3_traj_stdev, se3_t_traj, se3_traj, se3_v_traj, se3_meas);

  LieSplineOptimizer<lie_groups::SE3d> se3_est(se3_meas, std::pow(se3_meas_stdev, 2.0) * Eigen::Matrix<double, 6, 6>::Identity());
  se3_est.set_up_motion_priors(estimator::ModelType::ACC, 2*se3_dt, std::pow(se3_traj_stdev, 2.0) * Eigen::Matrix<double, 6, 6>::Identity());

  std::shared_ptr<std::vector<lie_groups::Map<lie_groups::SE3d>>> se3_ctrl_pts(new std::vector<lie_groups::Map<lie_groups::SE3d>>());
  int num_se3_cp = static_cast<int>(std::ceil((se3_meas.back().first - se3_meas[0].first)/se3_dt)) + se3_k;
  for(int i = 0; i < num_se3_cp; ++i)
    se3_ctrl_pts->emplace_back(new double[7]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0});
  se3_est.init_spline(se3_dt, se3_k, se3_ctrl_pts);

  std::cout << "Solving for SE(3) trajectory \n\n";
  se3_est.solve(true);
  // se3_est.solve(true);
}