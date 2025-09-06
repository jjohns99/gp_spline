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

class GPImuResidualSE3 : public ceres::CostFunction
{
private:
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Matrix6d = Eigen::Matrix<double, 6, 6>;
  using GP = gp::LieGP<lie_groups::Map<lie_groups::SE3d>, Eigen::Map<Vector6d>>;

public:
  GPImuResidualSE3(Vector6d z_imu, double t, double g, Matrix6d sqrt_W, std::shared_ptr<GP> gp) :
     z_imu_{z_imu}, t_{t}, g_{g}, sqrt_W_{sqrt_W}, gp_{gp}
  {
    set_num_residuals(6);
    std::vector<int32_t> param_block_sizes;
    // estimation parameters
    for(int i = 0; i < ((gp_->get_type() == gp::ModelType::ACC) ? 4 : 6); ++i)
    {
      if(i < 2) param_block_sizes.push_back(7); // poses have 7 parameters
      else param_block_sizes.push_back(6);
    }

    for(int i = 0; i < 2; ++i) param_block_sizes.push_back(3); // gyro and accel biases

    *mutable_parameter_block_sizes() = param_block_sizes;
  }

  virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
  {
    Eigen::Map<Vector6d> r(residuals);
    int gyro_bias_param_ind = (gp_->get_type() == gp::ModelType::ACC) ? 4 : 6;

    lie_groups::SE3d T;
    Vector6d v, a;
    std::vector<Matrix6d> T_jacs, v_jacs, a_jacs;
    gp_->eval(t_, &T, &v, &a, jacobians == nullptr ? nullptr : &T_jacs, jacobians == nullptr ? nullptr : &v_jacs, jacobians == nullptr ? nullptr : &a_jacs);

    Eigen::Vector3d omega = -v.tail<3>();
    Eigen::Vector3d acc = -a.head<3>();

    Eigen::Vector3d b_gyro = Eigen::Map<const Eigen::Vector3d>(parameters[gyro_bias_param_ind]);
    Eigen::Vector3d b_accel = Eigen::Map<const Eigen::Vector3d>(parameters[gyro_bias_param_ind + 1]);

    Eigen::Vector3d z_gyro = omega + b_gyro;
    Eigen::Vector3d z_accel = acc - g_ * T.rotation().matrix().col(2) + b_accel;

    r = sqrt_W_ * ((Vector6d() << z_gyro, z_accel).finished() - z_imu_);

    if(jacobians != nullptr)
    {
      // estimation parameter jacobians
      for(int i = 0; i < v_jacs.size(); ++i)
      {
        if(jacobians[i] != nullptr)
        {
          Eigen::Matrix<double, 6, GP::DoF> dz_dx;
          dz_dx.template block<3,GP::DoF>(0,0) = -v_jacs[i].template block<3,GP::DoF>(3,0);
          dz_dx.template block<3,GP::DoF>(3,0) = -a_jacs[i].template block<3,GP::DoF>(0,0) + g_ * lie_groups::SO3d::hat(T.rotation().matrix().col(2)) * T_jacs[i].template block<3,GP::DoF>(3,0);

          // Pose jacobians are 6x7. Need to do these separately
          if(i < 2)
          {
            Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J(jacobians[i]);
            J.col(6).setZero();
            J.block<6,6>(0,0) = sqrt_W_ * dz_dx;
          }
          else
          {
            Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> J(jacobians[i]);
            J = sqrt_W_ * dz_dx;
          }
        }
      }

      // bias jacobians
      if(jacobians[gyro_bias_param_ind] != nullptr)
      {
        Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J_gyro(jacobians[gyro_bias_param_ind]);
        J_gyro.block<3,3>(0,0) = sqrt_W_.block<3,3>(0,0);
        J_gyro.block<3,3>(3,0).setZero();
      }
      if(jacobians[gyro_bias_param_ind+1] != nullptr)
      {
        Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J_acc(jacobians[gyro_bias_param_ind+1]);
        J_acc.block<3,3>(0,0).setZero();
        J_acc.block<3,3>(3,0) = sqrt_W_.block<3,3>(3,3);
      }
    }

    return true;
  }

private:
  Vector6d z_imu_;
  Matrix6d sqrt_W_;
  double g_;
  double t_;

  std::shared_ptr<GP> gp_;
};


class GPImuResidualSO3xR3 : public ceres::CostFunction
{
private:
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Matrix6d = Eigen::Matrix<double, 6, 6>;
  using GP = gp::SOnxRnGP<lie_groups::Map<lie_groups::SO3d>, Eigen::Map<Eigen::Vector3d>, Eigen::Map<Eigen::Vector3d>>;

public:
  GPImuResidualSO3xR3(Vector6d z_imu, double t, double g, Matrix6d sqrt_W, std::shared_ptr<GP> gp) :
     z_imu_{z_imu}, t_{t}, g_{g}, sqrt_W_{sqrt_W}, gp_{gp}
  {
    set_num_residuals(6);
    std::vector<int32_t> param_block_sizes;
    // estimation parameters
    for(int i = 0; i < ((gp_->get_son_type() == gp::ModelType::ACC) ? 10 : 12); ++i)
    {
      if(i == 6 || i == 7) param_block_sizes.push_back(4); // rotations have 4 parameters
      else param_block_sizes.push_back(3);
    }

    for(int i = 0; i < 2; ++i) param_block_sizes.push_back(3); // gyro and accel biases

    *mutable_parameter_block_sizes() = param_block_sizes;
  }

  virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
  {
    Eigen::Map<Vector6d> r(residuals);
    int gyro_bias_param_ind = (gp_->get_son_type() == gp::ModelType::ACC) ? 10 : 12;

    lie_groups::SO3d R;
    Eigen::Vector3d v, a, om; // this omega is on the left of R, will need to negate
    std::vector<Eigen::Matrix3d> R_jacs, om_jacs, v_jacs, a_jacs;
    gp_->eval(t_, nullptr, &v, &a, &R, &om, nullptr, nullptr, jacobians == nullptr ? nullptr : &v_jacs, 
      jacobians == nullptr ? nullptr : &a_jacs, jacobians == nullptr ? nullptr : &R_jacs, jacobians == nullptr ? nullptr : &om_jacs);

    Eigen::Vector3d b_gyro = Eigen::Map<const Eigen::Vector3d>(parameters[gyro_bias_param_ind]);
    Eigen::Vector3d b_accel = Eigen::Map<const Eigen::Vector3d>(parameters[gyro_bias_param_ind + 1]);

    Eigen::Vector3d z_gyro = -om + b_gyro;
    Eigen::Vector3d z_accel = R * a - g_ * R.matrix().col(2) + b_accel
                              + lie_groups::SO3d::hat(om) * (R * v);

    r = sqrt_W_ * ((Vector6d() << z_gyro, z_accel).finished() - z_imu_);

    if(jacobians != nullptr)
    {
      Eigen::Matrix3d dza_dR = -lie_groups::SO3d::hat(R * a) + g_ * lie_groups::SO3d::hat(R.matrix().col(2))
                               - lie_groups::SO3d::hat(om) * lie_groups::SO3d::hat(R * v);

      // rn gp jacobians
      for(int i = 0; i < a_jacs.size(); ++i)
      {
        if(jacobians[i] != nullptr)
        {
          Eigen::Matrix<double, 6, 3> dz_dx;
          dz_dx.block<3,3>(0,0).setZero();
          dz_dx.block<3,3>(3,0) = R.matrix() * a_jacs[i] 
                                  + lie_groups::SO3d::hat(om) * R.matrix() * v_jacs[i];

          Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J(jacobians[i]);
          J = sqrt_W_ * dz_dx;
        }
      }

      // son gp jacobians
      for(int i = 0; i < R_jacs.size(); ++i)
      {
        int j = i + a_jacs.size();
        if(jacobians[j] != nullptr)
        {
          Eigen::Matrix<double, 6, 3> dz_dx;
          dz_dx.block<3,3>(0,0) = -om_jacs[i];
          dz_dx.block<3,3>(3,0) = dza_dR * R_jacs[i]
                                  - lie_groups::SO3d::hat(R * v) * om_jacs[i];

          if(i < 2) // rotation jacobians are 6x4
          {
            Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>> J(jacobians[j]);
            J.col(3).setZero();
            J.block<6,3>(0,0) = sqrt_W_ * dz_dx;
          }
          else
          {
            Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J(jacobians[j]);
            J.block<6,3>(0,0) = sqrt_W_ * dz_dx;
          }
        }
      }

      // bias jacobians
      if(jacobians[gyro_bias_param_ind] != nullptr)
      {
        Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J_gyro(jacobians[gyro_bias_param_ind]);
        J_gyro.block<3,3>(0,0) = sqrt_W_.block<3,3>(0,0);
        J_gyro.block<3,3>(3,0).setZero();
      }
      if(jacobians[gyro_bias_param_ind+1] != nullptr)
      {
        Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J_acc(jacobians[gyro_bias_param_ind+1]);
        J_acc.block<3,3>(0,0).setZero();
        J_acc.block<3,3>(3,0) = sqrt_W_.block<3,3>(3,3);
      }
    }

    return true;
  }

private:
  Vector6d z_imu_;
  Matrix6d sqrt_W_;
  double g_;
  double t_;

  std::shared_ptr<GP> gp_;
};

} // namespace estimator