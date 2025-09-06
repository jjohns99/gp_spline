#pragma once

#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <ceres/ceres.h>

#include "lie_groups/so3.hpp"
#include "lie_groups/se3.hpp"
#include "gp/lie_gp.hpp"
#include "gp/sonxrn_gp.hpp"

namespace estimator
{

// this residual should only be used for interpolated measurement times (i.e. not estimation times)
class GPApriltagResidualSE3 : public ceres::CostFunction
{
private:
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Matrix6d = Eigen::Matrix<double, 6, 6>;
  using GP = gp::LieGP<lie_groups::Map<lie_groups::SE3d>, Eigen::Map<Vector6d>>;

public:
  GPApriltagResidualSE3(lie_groups::SE3d const& z_pose, lie_groups::SE3d const& T_i_l, int pre_eval_id, 
    Matrix6d sqrt_W, std::shared_ptr<GP> gp, lie_groups::SE3d const& T_c_b) :
    z_pose_{z_pose}, T_i_l_{T_i_l}, pre_eval_id_{pre_eval_id}, sqrt_W_{sqrt_W}, gp_{gp}, T_c_b_{T_c_b}
  {
    set_num_residuals(6);
    std::vector<int32_t> param_block_sizes;
    // estimation parameters
    for(int i = 0; i < ((gp_->get_type() == gp::ModelType::ACC) ? 4 : 6); ++i)
    {
      if(i < 2) param_block_sizes.push_back(7); // poses have 7 parameters
      else param_block_sizes.push_back(6);
    }

    *mutable_parameter_block_sizes() = param_block_sizes;
  }

  virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
  {
    Eigen::Map<Vector6d> r(residuals);

    lie_groups::SE3d T_i_b;
    std::vector<Matrix6d> T_jacs;
    gp_->get_pre_eval(pre_eval_id_, &T_i_b, nullptr, nullptr, (jacobians == nullptr) ? nullptr : &T_jacs);

    // use right invariant error so that noise ends up in camera frame
    lie_groups::SE3d T_l_c = (T_i_l_ * T_i_b.inverse() * T_c_b_).inverse();
    Vector6d err = (T_l_c * z_pose_.inverse()).Log();
    r = sqrt_W_ * err;

    if(jacobians != nullptr)
    {
      Matrix6d dr_dTlc = lie_groups::SE3d::Jl_inv(err);
      Matrix6d dr_dTib = dr_dTlc * T_c_b_.inverse().Ad();
      for(int i = 0; i < T_jacs.size(); ++i)
      {
        if(i < 2)
        {
          Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J(jacobians[i]);
          J.col(6).setZero();
          J.block<6,6>(0,0) = sqrt_W_ * dr_dTib * T_jacs[i];
        }
        else
        {
          Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> J(jacobians[i]);
          J = sqrt_W_ * dr_dTib * T_jacs[i];
        }
      }
    }

    return true;
  }

private:
  lie_groups::SE3d z_pose_;
  lie_groups::SE3d T_i_l_;
  lie_groups::SE3d T_c_b_;

  int pre_eval_id_;
  Matrix6d sqrt_W_;

  std::shared_ptr<GP> gp_;
};

// this residual should only be used for interpolated measurement times (i.e. not estimation times)
class GPApriltagResidualSO3xR3 : public ceres::CostFunction
{
private:
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Matrix6d = Eigen::Matrix<double, 6, 6>;
  using GP = gp::SOnxRnGP<lie_groups::Map<lie_groups::SO3d>, Eigen::Map<Eigen::Vector3d>, Eigen::Map<Eigen::Vector3d>>;

public:
  GPApriltagResidualSO3xR3(lie_groups::SE3d const& z_pose, lie_groups::SE3d const& T_i_l, int pre_eval_id, 
    Matrix6d sqrt_W, std::shared_ptr<GP> gp, lie_groups::SE3d const& T_c_b) :
    z_pose_{z_pose}, T_i_l_{T_i_l}, pre_eval_id_{pre_eval_id}, sqrt_W_{sqrt_W}, gp_{gp}, T_c_b_{T_c_b}
  {
    set_num_residuals(6);
    std::vector<int32_t> param_block_sizes;
    // estimation parameters
    for(int i = 0; i < ((gp_->get_son_type() == gp::ModelType::ACC) ? 10 : 12); ++i)
    {
      if(i == 6 || i == 7) param_block_sizes.push_back(4); // rotations have 4 parameters
      else param_block_sizes.push_back(3);
    }

    *mutable_parameter_block_sizes() = param_block_sizes;
  }

  virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
  {
    Eigen::Map<Vector6d> r(residuals);

    lie_groups::SO3d R;
    Eigen::Vector3d p;
    std::vector<Eigen::Matrix3d> R_jacs, p_jacs;
    gp_->get_pre_eval(pre_eval_id_, &p, nullptr, nullptr, &R, nullptr, nullptr, 
      ((jacobians == nullptr) ? nullptr : &p_jacs), nullptr, nullptr, ((jacobians == nullptr) ? nullptr : &R_jacs));

    lie_groups::SE3d T_i_b = lie_groups::SE3d(R, -(R * p));
    // use right invariant error so that noise ends up in camera frame
    lie_groups::SE3d T_l_c = (T_i_l_ * T_i_b.inverse() * T_c_b_).inverse();
    Vector6d err = (T_l_c * z_pose_.inverse()).Log();
    r = sqrt_W_ * err;

    if(jacobians != nullptr)
    {
      Matrix6d dr_dTlc = lie_groups::SE3d::Jl_inv(err);
      Matrix6d dr_dTib = dr_dTlc * T_c_b_.inverse().Ad();

      for(int i = 0; i < p_jacs.size(); ++i)
      {
        if(jacobians[i] != nullptr)
        {
          Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J(jacobians[i]);
          J = sqrt_W_ * dr_dTib.block<6,3>(0,0) * (-(R.matrix())) * p_jacs[i];
        }
      }
      for(int i = 0; i < R_jacs.size(); ++i)
      {
        int j = i + p_jacs.size();
        if(jacobians[j] != nullptr)
        {
          if(i < 2)
          {
            Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>> J(jacobians[j]);
            J.col(3).setZero();
            J.block<6,3>(0,0) = sqrt_W_ * dr_dTib.block<6,3>(0,3) * R_jacs[i];
          }
          else
          {
            Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J(jacobians[j]);
            J = sqrt_W_ * dr_dTib.block<6,3>(0,3) * R_jacs[i];
          }
        }
      }
    }

    return true;
  }

private:
  lie_groups::SE3d z_pose_;
  lie_groups::SE3d T_i_l_;
  lie_groups::SE3d T_c_b_;

  int pre_eval_id_;
  Matrix6d sqrt_W_;

  std::shared_ptr<GP> gp_;
};

} // namespace estimator