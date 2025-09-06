#pragma once

#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <ceres/ceres.h>

#include "lie_groups/so3.hpp"
#include "lie_groups/se3.hpp"

namespace estimator
{

// this residual should be used for discrete-time estimation or for gp estimation at estimation times
class ApriltagResidualSE3 : public ceres::CostFunction
{
private:
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Matrix6d = Eigen::Matrix<double, 6, 6>;

public:
  ApriltagResidualSE3(lie_groups::SE3d const& z_pose, lie_groups::SE3d const& T_i_l, 
    Matrix6d sqrt_W, lie_groups::SE3d const& T_c_b) :
    z_pose_{z_pose}, T_i_l_{T_i_l}, sqrt_W_{sqrt_W}, T_c_b_{T_c_b}
  {
    set_num_residuals(6);

    std::vector<int32_t> param_block_sizes;
    param_block_sizes.push_back(7);

    *mutable_parameter_block_sizes() = param_block_sizes;
  }

  virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
  {
    Eigen::Map<Vector6d> r(residuals);

    lie_groups::Map<const lie_groups::SE3d> T_i_b(parameters[0]);

    // use right invariant error so that noise ends up in camera frame
    lie_groups::SE3d T_l_c = (T_i_l_ * T_i_b.inverse() * T_c_b_).inverse();
    Vector6d err = (T_l_c * z_pose_.inverse()).Log();
    r = sqrt_W_ * err;

    if(jacobians != nullptr)
    {
      Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J(jacobians[0]);
      J.col(6).setZero();
      J.block<6,6>(0,0) = sqrt_W_ * lie_groups::SE3d::Jl_inv(err) * T_c_b_.inverse().Ad();
    }

    return true;
  }

private:
  lie_groups::SE3d z_pose_;
  lie_groups::SE3d T_i_l_;
  lie_groups::SE3d T_c_b_;

  Matrix6d sqrt_W_;
};

// this residual should be used for discrete-time estimation or for gp estimation at estimation times
class ApriltagResidualSO3xR3 : public ceres::CostFunction
{
private:
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Matrix6d = Eigen::Matrix<double, 6, 6>;

public:
  ApriltagResidualSO3xR3(lie_groups::SE3d const& z_pose, lie_groups::SE3d const& T_i_l, 
    Matrix6d sqrt_W, lie_groups::SE3d const& T_c_b) :
    z_pose_{z_pose}, T_i_l_{T_i_l}, sqrt_W_{sqrt_W}, T_c_b_{T_c_b}
  {
    set_num_residuals(6);

    std::vector<int32_t> param_block_sizes;
    param_block_sizes.push_back(3);
    param_block_sizes.push_back(4);

    *mutable_parameter_block_sizes() = param_block_sizes;
  }

  virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
  {
    Eigen::Map<Vector6d> r(residuals);

    Eigen::Map<const Eigen::Vector3d> p(parameters[0]);
    lie_groups::Map<const lie_groups::SO3d> R(parameters[1]);
    lie_groups::SE3d T_i_b = lie_groups::SE3d(R.mem(), -(R * p));

    // use right invariant error so that noise ends up in camera frame
    lie_groups::SE3d T_l_c = (T_i_l_ * T_i_b.inverse() * T_c_b_).inverse();
    Vector6d err = (T_l_c * z_pose_.inverse()).Log();
    r = sqrt_W_ * err;

    if(jacobians != nullptr)
    {
      Matrix6d dr_dT = sqrt_W_ * lie_groups::SE3d::Jl_inv(err) * T_c_b_.inverse().Ad();
      Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> Jp(jacobians[0]);
      Jp = dr_dT.block<6,3>(0,0) * (-(R.matrix()));
      
      Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>> JR(jacobians[1]);
      JR.col(3).setZero();
      JR.block<6,3>(0,0) = dr_dT.block<6,3>(0,3);
    }

    return true;
  }

private:
  lie_groups::SE3d z_pose_;
  lie_groups::SE3d T_i_l_;
  lie_groups::SE3d T_c_b_;

  Matrix6d sqrt_W_;
};

} // namespace estimator