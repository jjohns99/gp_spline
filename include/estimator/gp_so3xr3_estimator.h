#pragma once

#include <vector>
#include <memory>
#include <cmath>
#include <limits>
#include <map>
#include <Eigen/Dense>
#include <ceres/ceres.h>

#include "lie_groups/se3.hpp"
#include "gp/lie_gp.hpp"
#include "estimator/estimator_base.h"
#include "residuals/pose_eval_callback.hpp"
#include "residuals/apriltag_residual.hpp"
#include "residuals/gp_apriltag_residual.hpp"
#include "residuals/gp_imu_residual.hpp"
#include "residuals/gp_dynamics_residual.hpp"
#include "parameterizations/son_parameterization.hpp"

namespace estimator
{

class GPSO3xR3Estimator : public EstimatorBase
{
private:
  using SOn = lie_groups::Map<lie_groups::SO3d>;
  using Rn = Eigen::Map<Eigen::Vector3d>;
  using GP = gp::SOnxRnGP<SOn, Rn, Rn>;
  using Matrix6d = Eigen::Matrix<double, 6, 6>;

  using SOnGP = gp::LieGP<SOn, Rn>;
  using RnGP = gp::RnGP<Rn, 3>;

public:
  GPSO3xR3Estimator(EstimatorParams est_params);

  // n means that every nth camera image will be used as an estimation time
  void init_gp(Matrix6d Q, int n = 1, gp::ModelType son_type = gp::ModelType::ACC);

  std::shared_ptr<GP> get_gp();

private:
  virtual void setup_opt(ceres::Problem& problem, bool use_motion_priors = false) override;

  std::shared_ptr<GP> gp_;

  std::shared_ptr<std::vector<SOn>> R_;
  std::shared_ptr<std::vector<Rn>> p_;
  std::shared_ptr<std::vector<Rn>> om_;
  std::shared_ptr<std::vector<Rn>> v_;
  std::shared_ptr<std::vector<Rn>> omdot_;
  std::shared_ptr<std::vector<Rn>> a_;

  Matrix6d Q_;
  gp::ModelType son_type_;

  int n_; // every nth image will contain an estimation time
  std::map<int, int> pre_eval_map_; // map from z_land_ index to pre eval index
};

} // namespace estimator