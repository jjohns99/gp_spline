#pragma once

#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <ceres/ceres.h>

#include "lie_groups/so3.hpp"
#include "spline/sonxrn_spline.hpp"

namespace estimator
{

template <typename SplineT>
class SplineImuResidualBase : public ceres::CostFunction
{
private:
  using BiasSplineT = spline::RnSpline<Eigen::Map<Eigen::Vector3d>, 3>;
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Matrix6d = Eigen::Matrix<double, 6, 6>;

public:
  SplineImuResidualBase(Vector6d z_imu, double t, double g, Matrix6d sqrt_W, std::shared_ptr<SplineT> spl, 
    std::shared_ptr<BiasSplineT> gb_spl = nullptr, std::shared_ptr<BiasSplineT> ab_spl = nullptr) :
    z_imu_{z_imu}, t_{t}, g_{g}, sqrt_W_{sqrt_W}, spline_{spl}, gyro_bias_spline_{gb_spl},
    accel_bias_spline_{ab_spl}, use_bias_splines_{gb_spl != nullptr}
  {
    set_num_residuals(6);
    std::vector<int32_t> param_block_sizes;
    for(int i = 0; i < spline_->get_order(); ++i) param_block_sizes.push_back(7);

    // initialize gyro and accel biases
    if(use_bias_splines_)
    {
      for(int i = 0; i < gyro_bias_spline_->get_order(); ++i) param_block_sizes.push_back(3);
      for(int i = 0; i < accel_bias_spline_->get_order(); ++i) param_block_sizes.push_back(3);
    }
    else
    {
      for(int i = 0; i < 2; ++i) param_block_sizes.push_back(3); // gyro and accel biases
    }

    *mutable_parameter_block_sizes() = param_block_sizes;
  }

protected:
  std::shared_ptr<SplineT> spline_;
  const Vector6d z_imu_;
  double t_;
  double g_;
  const Matrix6d sqrt_W_;

  // imu bias splines, if supplied. Otherwise assume biases are constant
  std::shared_ptr<BiasSplineT> gyro_bias_spline_;
  std::shared_ptr<BiasSplineT> accel_bias_spline_;
  bool use_bias_splines_;

};

template <typename SplineT>
class SplineImuResidual : public SplineImuResidualBase<SplineT>
{
private:
  using BiasSplineT = spline::RnSpline<Eigen::Map<Eigen::Vector3d>, 3>;
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Matrix6d = Eigen::Matrix<double, 6, 6>;

public:
  SplineImuResidual(Vector6d z_imu, double t, double g, Matrix6d sqrt_W, std::shared_ptr<SplineT> spl, 
    std::shared_ptr<BiasSplineT> gb_spl = nullptr, std::shared_ptr<BiasSplineT> ab_spl = nullptr) :
    SplineImuResidualBase<SplineT>{z_imu, t, g, sqrt_W, spl, gb_spl, ab_spl}
  {}

  virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
  {
    Eigen::Map<Vector6d> r(residuals);
    int gyro_bias_param_ind = this->spline_->get_order();
    int accel_bias_param_ind = !this->use_bias_splines_ ? gyro_bias_param_ind + 1 : gyro_bias_param_ind + this->gyro_bias_spline_->get_order();

    lie_groups::SE3d T;
    Vector6d v, a;
    std::vector<Matrix6d> spl_d0_jacs, spl_d1_jacs, spl_d2_jacs;
    this->spline_->eval(this->t_, &T, &v, &a, (jacobians == nullptr) ? nullptr : &spl_d0_jacs, (jacobians == nullptr) ? nullptr : &spl_d1_jacs, 
      (jacobians == nullptr) ? nullptr : &spl_d2_jacs);

    Eigen::Vector3d omega = -v.tail<3>();
    Eigen::Vector3d vel = -v.head<3>();
    Eigen::Vector3d acc = -a.head<3>();

    // get biases
    Eigen::Vector3d b_gyro, b_accel;
    std::vector<Eigen::Matrix3d> b_gyro_d0_jacs, b_accel_d0_jacs;
    if(this->use_bias_splines_)
    {
      this->gyro_bias_spline_->eval(this->t_, &b_gyro, nullptr, nullptr, (jacobians == nullptr) ? nullptr : & b_gyro_d0_jacs);
      this->accel_bias_spline_->eval(this->t_, &b_accel, nullptr, nullptr, (jacobians == nullptr) ? nullptr : & b_accel_d0_jacs);
    }
    else
    {
      b_gyro = Eigen::Map<const Eigen::Vector3d>(parameters[gyro_bias_param_ind]);
      b_accel = Eigen::Map<const Eigen::Vector3d>(parameters[accel_bias_param_ind]);
    }
    
    Eigen::Vector3d z_gyro = omega + b_gyro;
    Eigen::Vector3d z_accel = acc - this->g_ * T.rotation().matrix().col(2) + b_accel;
                              // - lie_groups::SO3d::hat(omega) * vel;

    r = this->sqrt_W_ * ((Vector6d() << z_gyro, z_accel).finished() - this->z_imu_);

    if(jacobians != nullptr)
    {
      // control point jacobians
      for(int i = 0; i < this->spline_->get_order(); ++i)
      {
        if(jacobians[i] != nullptr)
        {
          Eigen::Matrix<double, 6, 6> dz_dT;
          dz_dT.block<3,6>(0,0) = -spl_d1_jacs[i].block<3,6>(3,0);
          dz_dT.block<3,6>(3,0) = -spl_d2_jacs[i].block<3,6>(0,0) + this->g_ * lie_groups::SO3d::hat(T.rotation().matrix().col(2)) * spl_d0_jacs[i].block<3,6>(3,0);
                                  // - (Eigen::Matrix<double, 3, 6>() << lie_groups::SO3d::hat(omega), -lie_groups::SO3d::hat(vel)).finished() * spl_d1_jacs[i];

          Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J(jacobians[i]);
          J.setZero();
          J.block<6,6>(0,0) = this->sqrt_W_ * dz_dT;
        }
      }

      // biases
      if(this->use_bias_splines_)
      {
        for(int i = 0; i < this->gyro_bias_spline_->get_order(); ++i)
        {
          if(jacobians[gyro_bias_param_ind + i] != nullptr)
          {
            Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J_gyro(jacobians[gyro_bias_param_ind + i]);
            J_gyro.block<3,3>(0,0) = (this->sqrt_W_.template block<3,3>(0,0)) * b_gyro_d0_jacs[i];
            J_gyro.block<3,3>(3,0).setZero();
          }
        }
        for(int i = 0; i < this->accel_bias_spline_->get_order(); ++i)
        {
          if(jacobians[accel_bias_param_ind + i] != nullptr)
          {
            Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J_accel(jacobians[accel_bias_param_ind + i]);
            J_accel.block<3,3>(0,0).setZero();
            J_accel.block<3,3>(3,0) = (this->sqrt_W_.template block<3,3>(3,3)) * b_accel_d0_jacs[i];
          }
        }
      }
      else
      {
        if(jacobians[gyro_bias_param_ind] != nullptr)
        {
          Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J_gyro(jacobians[gyro_bias_param_ind]);
          J_gyro.block<3,3>(0,0) = this->sqrt_W_.template block<3,3>(0,0);
          J_gyro.block<3,3>(3,0).setZero();
        }
        if(jacobians[accel_bias_param_ind] != nullptr)
        {
          Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J_accel(jacobians[accel_bias_param_ind]);
          J_accel.block<3,3>(0,0).setZero();
          J_accel.block<3,3>(3,0) = this->sqrt_W_.template block<3,3>(3,3);
        }
      }
    }

    return true;
  }
};


// Need explicit specialization for SO3xR3
template <typename G>
class SplineImuResidual<spline::SOnxRnSpline<G>> : public SplineImuResidualBase<spline::SOnxRnSpline<G>>
{
private:
  using BiasSplineT = spline::RnSpline<Eigen::Map<Eigen::Vector3d>, 3>;
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Matrix6d = Eigen::Matrix<double, 6, 6>;

public:
  SplineImuResidual(Vector6d z_imu, double t, double g, Matrix6d sqrt_W, std::shared_ptr<spline::SOnxRnSpline<G>> spl,
    std::shared_ptr<BiasSplineT> gb_spl = nullptr, std::shared_ptr<BiasSplineT> ab_spl = nullptr) :
    SplineImuResidualBase<spline::SOnxRnSpline<G>>{z_imu, t, g, sqrt_W, spl, gb_spl, ab_spl}
  {}

  virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
  {
    Eigen::Map<Vector6d> r(residuals);
    int gyro_bias_param_ind = this->spline_->get_order();
    int accel_bias_param_ind = !this->use_bias_splines_ ? gyro_bias_param_ind + 1 : gyro_bias_param_ind + this->gyro_bias_spline_->get_order();

    lie_groups::SO3d R_i_b;
    Eigen::Vector3d om, v, a;
    std::vector<Eigen::Matrix3d> spl_R_jacs, spl_om_jacs, spl_v_jacs, spl_a_jacs;
    this->spline_->eval_son(this->t_, &R_i_b, &om, nullptr, (jacobians == nullptr) ? nullptr : &spl_R_jacs,
      (jacobians == nullptr) ? nullptr : &spl_om_jacs);
    this->spline_->eval_rn(this->t_, nullptr, &v, &a, nullptr, (jacobians == nullptr) ? nullptr : &spl_v_jacs, 
      (jacobians == nullptr) ? nullptr : &spl_a_jacs);
    Eigen::Vector3d omega = -om;

    // get biases
    Eigen::Vector3d b_gyro, b_accel;
    std::vector<Eigen::Matrix3d> b_gyro_d0_jacs, b_accel_d0_jacs;
    if(this->use_bias_splines_)
    {
      this->gyro_bias_spline_->eval(this->t_, &b_gyro, nullptr, nullptr, (jacobians == nullptr) ? nullptr : & b_gyro_d0_jacs);
      this->accel_bias_spline_->eval(this->t_, &b_accel, nullptr, nullptr, (jacobians == nullptr) ? nullptr : & b_accel_d0_jacs);
    }
    else
    {
      b_gyro = Eigen::Map<const Eigen::Vector3d>(parameters[gyro_bias_param_ind]);
      b_accel = Eigen::Map<const Eigen::Vector3d>(parameters[accel_bias_param_ind]);
    }
    
    Eigen::Vector3d z_gyro = omega + b_gyro;
    Eigen::Vector3d z_accel = R_i_b * a - lie_groups::SO3d::hat(omega) * (R_i_b * v) - this->g_ * R_i_b.matrix().col(2) + b_accel;
    // Eigen::Vector3d z_accel = R_i_b * a - this->g_ * R_i_b.matrix().col(2) + b_accel;

    r = this->sqrt_W_ * ((Vector6d() << z_gyro, z_accel).finished() - this->z_imu_);

    if(jacobians != nullptr)
    {
      // control point jacobians
      for(int i = 0; i < this->spline_->get_order(); ++i)
      {
        if(jacobians[i] != nullptr)
        {
          Eigen::Matrix<double, 6, 6> dz_dT;
          dz_dT.block<3,3>(0,0).setZero();
          dz_dT.block<3,3>(0,3) = -spl_om_jacs[i];
          dz_dT.block<3,3>(3,0) = -lie_groups::SO3d::hat(omega) * R_i_b.matrix() * spl_v_jacs[i] + R_i_b.matrix() * spl_a_jacs[i];
          dz_dT.block<3,3>(3,3) = (-lie_groups::SO3d::hat(R_i_b * a) + lie_groups::SO3d::hat(omega) * lie_groups::SO3d::hat(R_i_b * v) + this->g_ * lie_groups::SO3d::hat(R_i_b.matrix().col(2))) * spl_R_jacs[i]
                                  - lie_groups::SO3d::hat(R_i_b * v) * spl_om_jacs[i];
          // dz_dT.block<3,3>(3,0) = R_i_b.matrix() * spl_a_jacs[i];
          // dz_dT.block<3,3>(3,3) = (-lie_groups::SO3d::hat(R_i_b * a) + this->g_ * lie_groups::SO3d::hat(R_i_b.matrix().col(2))) * spl_R_jacs[i];

          Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J(jacobians[i]);
          J.setZero();
          J.block<6,6>(0,0) = this->sqrt_W_ * dz_dT;
        }
      }

      // biases
      if(this->use_bias_splines_)
      {
        for(int i = 0; i < this->gyro_bias_spline_->get_order(); ++i)
        {
          if(jacobians[gyro_bias_param_ind + i] != nullptr)
          {
            Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J_gyro(jacobians[gyro_bias_param_ind + i]);
            J_gyro.block<3,3>(0,0) = (this->sqrt_W_.template block<3,3>(0,0)) * b_gyro_d0_jacs[i];
            J_gyro.block<3,3>(3,0).setZero();
          }
        }
        for(int i = 0; i < this->accel_bias_spline_->get_order(); ++i)
        {
          if(jacobians[accel_bias_param_ind + i] != nullptr)
          {
            Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J_accel(jacobians[accel_bias_param_ind + i]);
            J_accel.block<3,3>(0,0).setZero();
            J_accel.block<3,3>(3,0) = (this->sqrt_W_.template block<3,3>(3,3)) * b_accel_d0_jacs[i];
          }
        }
      }
      else
      {
        if(jacobians[gyro_bias_param_ind] != nullptr)
        {
          Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J_gyro(jacobians[gyro_bias_param_ind]);
          J_gyro.block<3,3>(0,0) = this->sqrt_W_.template block<3,3>(0,0);
          J_gyro.block<3,3>(3,0).setZero();
        }
        if(jacobians[accel_bias_param_ind] != nullptr)
        {
          Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J_accel(jacobians[accel_bias_param_ind]);
          J_accel.block<3,3>(0,0).setZero();
          J_accel.block<3,3>(3,0) = this->sqrt_W_.template block<3,3>(3,3);
        }
      }
    }

    return true;
  }
};

} // namespace estimator