#pragma once

#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <ceres/ceres.h>

#include "lie_groups/so3.hpp"
#include "lie_groups/se3.hpp"
#include "lie_groups/sonxrn.hpp"
#include "spline/sonxrn_spline.hpp"

#include <iostream>

namespace estimator
{

// in this case the spline represents the transformation from the inertial frame to the imu frame
template <typename SplineT>
class SplineApriltagResidualBase : public ceres::CostFunction
{
private:
  using Matrix6d = Eigen::Matrix<double, 6, 6>;

public:
  SplineApriltagResidualBase(lie_groups::SE3d const& z_pose, lie_groups::SE3d const& T_i_l, int pre_eval_id, 
    Matrix6d sqrt_W, std::shared_ptr<SplineT> spl, lie_groups::SE3d const& T_c_b) :
    z_pose_{z_pose}, T_i_l_{T_i_l}, pre_eval_id_{pre_eval_id}, sqrt_W_{sqrt_W}, spline_{spl}, T_c_b_{T_c_b}
  {
    set_num_residuals(6);
    std::vector<int32_t> param_block_sizes;
    for(int i = 0; i < spline_->get_order(); ++i) param_block_sizes.push_back(7);

    *mutable_parameter_block_sizes() = param_block_sizes;
  }

protected:
  std::shared_ptr<SplineT> spline_;
  const lie_groups::SE3d z_pose_; // tx from lth landmark frame to camera frame
  const lie_groups::SE3d T_i_l_;
  int pre_eval_id_;
  const Matrix6d sqrt_W_;
  const lie_groups::SE3d T_c_b_; // this is precalibrated

};


template <typename SplineT>
class SplineApriltagResidual : public SplineApriltagResidualBase<SplineT>
{
private:
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Matrix6d = Eigen::Matrix<double, 6, 6>;

public:
  SplineApriltagResidual(lie_groups::SE3d const& z_pose, lie_groups::SE3d const& T_i_l, int pre_eval_id, 
    Matrix6d sqrt_W, std::shared_ptr<SplineT> spl, lie_groups::SE3d const& T_c_b) :
    SplineApriltagResidualBase<SplineT>{z_pose, T_i_l, pre_eval_id, sqrt_W, spl, T_c_b}
  {}

  virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
  {
    Eigen::Map<Vector6d> r(residuals);

    // the evaluation callback precomputes jacobians to avoid calling spline->get_jac() for every residual
    typename SplineT::NonmapT T_i_b;
    std::vector<Matrix6d> spl_d0_jacs;
    this->spline_->get_pre_eval(this->pre_eval_id_, &T_i_b, nullptr, nullptr, (jacobians == nullptr) ? nullptr : &spl_d0_jacs);
    typename SplineT::NonmapT T_b_i = T_i_b.inverse();

    // use right invariant error so that noise ends up in camera frame
    lie_groups::SE3d T_l_c = (this->T_i_l_ * T_b_i * this->T_c_b_).inverse();
    Vector6d err = (T_l_c * this->z_pose_.inverse()).Log();
    r = this->sqrt_W_ * err;
    
    if(jacobians != nullptr)
    {
      Matrix6d dr_dTlc = lie_groups::SE3d::Jl_inv(err);
      Matrix6d dr_dTib = dr_dTlc * this->T_c_b_.inverse().Ad();
      for(int i = 0; i < this->spline_->get_order(); ++i)
      {
        if(jacobians[i] != nullptr)
        {
          Eigen::Matrix<double, 6, 6> dr_dT = dr_dTib * spl_d0_jacs[i];
          Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J(jacobians[i]);
          J.setZero();
          J.block<6,6>(0,0) = this->sqrt_W_ * dr_dT;
        }
      }
    }

    return true;
  }
};


// Need explicit specialization for SO3xR3
template <typename G>
class SplineApriltagResidual<spline::SOnxRnSpline<G>> : public SplineApriltagResidualBase<spline::SOnxRnSpline<G>>
{
private:
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Matrix6d = Eigen::Matrix<double, 6, 6>;

public:
  SplineApriltagResidual(lie_groups::SE3d const& z_pose, lie_groups::SE3d const& T_i_l, int pre_eval_id, 
    Matrix6d sqrt_W, std::shared_ptr<spline::SOnxRnSpline<G>> spl, lie_groups::SE3d const& T_c_b) :
    SplineApriltagResidualBase<spline::SOnxRnSpline<G>>{z_pose, T_i_l, pre_eval_id, sqrt_W, spl, T_c_b}
  {}

  virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
  {
    Eigen::Map<Vector6d> r(residuals);

    // the evaluation callback precomputes jacobians to avoid calling spline->get_jac() for every residual
    typename G::NonmapT R_i_bxp;
    std::vector<Matrix6d> spl_d0_jacs;
    this->spline_->get_pre_eval(this->pre_eval_id_, &R_i_bxp, nullptr, nullptr, (jacobians == nullptr) ? nullptr : &spl_d0_jacs);
    lie_groups::SO3d R_b_i = R_i_bxp.rotation().inverse();

    lie_groups::SE3d T_b_i = lie_groups::SE3d(R_b_i, R_i_bxp.translation());

    // use right invariant error so that noise ends up in camera frame
    lie_groups::SE3d T_l_c = (this->T_i_l_ * T_b_i * this->T_c_b_).inverse();
    Vector6d err = (T_l_c * this->z_pose_.inverse()).Log();
    r = this->sqrt_W_ * err;
    
    if(jacobians != nullptr)
    {
      Matrix6d dr_dTlc = lie_groups::SE3d::Jl_inv(err);
      Matrix6d dr_dTbi = -dr_dTlc * (T_b_i * this->T_c_b_).inverse().Ad();
      for(int i = 0; i < this->spline_->get_order(); ++i)
      {
        if(jacobians[i] != nullptr)
        {
          Eigen::Matrix<double, 6, 6> dTbi_dT{Eigen::Matrix<double, 6, 6>::Zero()};
          dTbi_dT.block<3,3>(0,0) = spl_d0_jacs[i].block<3,3>(0,0);
          dTbi_dT.block<6,3>(0,3) = -(Eigen::Matrix<double, 6, 3>() << lie_groups::SO3d::hat(R_i_bxp.translation()), 
              Eigen::Matrix3d::Identity()).finished() * R_b_i.matrix() * spl_d0_jacs[i].block<3,3>(3,3);

          Eigen::Matrix<double, 6, 6> dr_dT = dr_dTbi * dTbi_dT;
          Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J(jacobians[i]);
          J.setZero();
          J.block<6,6>(0,0) = this->sqrt_W_ * dr_dT;
        }
      }
    }

    return true;
  }
};

} // namespace estimator