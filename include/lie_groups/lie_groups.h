#pragma once

#include <Eigen/Dense>

namespace lie_groups
{

const double SMALL_ANGLE = 1e-10;

namespace internal
{

// this struct will hold typenames and dimensions
template <typename T>
struct traits;

} //namespace internal

template <typename T, typename TBase, typename P, int SIZE, int DoF>
class LieGroup
{
public:
  // need nonconst version of matrix() in order to not recompute
  // the matrix everytime the function is called. See SO2 or SO3
  // to see why
  virtual Eigen::Matrix<P, SIZE, SIZE> matrix() = 0;
  virtual Eigen::Matrix<P, SIZE, SIZE> matrix() const = 0;
  virtual T inverse() const = 0;
  static T Exp(Eigen::Matrix<P, DoF, 1> tau){ return TBase::Exp(tau); };
  virtual Eigen::Matrix<P, DoF, 1> Log() const = 0;
  virtual Eigen::Matrix<P, DoF, DoF> Ad() = 0;
  static Eigen::Matrix<P, SIZE, SIZE> hat(Eigen::Matrix<P, DoF, 1> tau){ return TBase::hat(tau); };
  static Eigen::Matrix<P, DoF, 1> vee(Eigen::Matrix<P, SIZE, SIZE> tau_hat){ return TBase::vee(tau_hat); };
  static Eigen::Matrix<P, DoF, DoF> ad(Eigen::Matrix<P, DoF, 1> tau){ return TBase::ad(tau); };
  static Eigen::Matrix<P, DoF, DoF> Jr(Eigen::Matrix<P, DoF, 1> tau) { return TBase::Jr(tau); };
  static Eigen::Matrix<P, DoF, DoF> Jl(Eigen::Matrix<P, DoF, 1> tau){ return TBase::Jl(tau); };
  static Eigen::Matrix<P, DoF, DoF> Jr_inv(Eigen::Matrix<P, DoF, 1> tau){ return TBase::Jr_inv(tau); };
  static Eigen::Matrix<P, DoF, DoF> Jl_inv(Eigen::Matrix<P, DoF, 1> tau){ return TBase::Jr_inv(tau); };
  static T random() {return TBase::random(); };

  // No simple way to inherit this
  // virtual T operator*(const OtherT &M) const = 0;
};

// this will allow us to create Eigen-like maps
// for our lie groups
template <typename T>
class Map;

} //namespace lie_groups