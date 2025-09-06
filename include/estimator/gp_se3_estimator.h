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
#include "parameterizations/sen_parameterization.hpp"

namespace estimator
{

class GPSE3Estimator : public EstimatorBase
{
private:
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Matrix6d = Eigen::Matrix<double, 6, 6>;
  using G = lie_groups::Map<lie_groups::SE3d>;
  using T = Eigen::Map<Vector6d>;
  using GP = gp::LieGP<G, T>;

public:
  GPSE3Estimator(EstimatorParams est_params);

  // n means that every nth camera image will be used as an estimation time
  void init_gp(Matrix6d Q, int n = 1);

  std::shared_ptr<GP> get_gp();

private:
  virtual void setup_opt(ceres::Problem& problem, bool use_motion_priors = false) override;

  std::shared_ptr<GP> gp_;
  std::shared_ptr<std::vector<G>> T_;
  std::shared_ptr<std::vector<T>> v_;
  std::shared_ptr<std::vector<T>> a_;

  Matrix6d Q_;

  int n_; // every nth image will contain an estimation time
  std::map<int, int> pre_eval_map_; // map from z_land_ index to pre eval index
};

} // namespace estimator