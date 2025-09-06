#pragma once

#include <memory>
#include <vector>
#include <Eigen/Dense>
#include <ceres/ceres.h>

#include "lie_groups/se3.hpp"
#include "spline/lie_spline.hpp"

namespace estimator
{

class MocapSplineResidual : public ceres::SizedCostFunction<6, 7, 1>
{
private:
  using SplineT = spline::LieSpline<lie_groups::Map<lie_groups::SE3d>>;
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Matrix6d = Eigen::Matrix<double, 6, 6>;

public:
  MocapSplineResidual(lie_groups::SE3d z_mocap, double t, std::shared_ptr<SplineT> spl) :
    z_mocap_{z_mocap}, t_{t}, spline_{spl}
  {}

  virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
  {
    Eigen::Map<Vector6d> r(residuals);
    lie_groups::Map<const lie_groups::SE3d> T_bm(parameters[0]);
    double time_offset = parameters[1][0];

    lie_groups::SE3d T_ib;
    Vector6d vel;
    spline_->eval(t_ + time_offset, &T_ib, (jacobians == nullptr) ? nullptr : &vel);

    r = (T_bm * T_ib * z_mocap_.inverse()).Log();

    if(jacobians != nullptr)
    {
      Matrix6d dz_dT = lie_groups::SE3d::Jl_inv(r);
      if(jacobians[0] != nullptr)
      {
        Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J_pose(jacobians[0]);
        J_pose.setZero();
        J_pose.block<6,6>(0,0) = dz_dT;
      }

      if(jacobians[1] != nullptr)
      {
        Eigen::Map<Vector6d> J_time(jacobians[1]);
        J_time = dz_dT * vel;
      }
    }

    return true;
  }

private:
  std::shared_ptr<SplineT> spline_;
  double t_;
  lie_groups::SE3d z_mocap_;

};

} // namespace estimator