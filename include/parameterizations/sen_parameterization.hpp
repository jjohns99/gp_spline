#pragma once

#include <ceres/ceres.h>
#include <Eigen/Dense>

// works for any lie_group SEn type, including SOnxRn types
template <typename G>
class SEnParameterization : public ceres::LocalParameterization
{
public:
  virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const
  {
    lie_groups::Map<const G> T(x);
    Eigen::Map<const typename G::TangentT> tau(delta);
    lie_groups::Map<G> T_delta(x_plus_delta);
    T_delta = G::Exp(tau) * T;

    return true;
  }

  virtual bool ComputeJacobian(const double* x, double* jacobian) const
  {
    Eigen::Map<Eigen::Matrix<double, G::MEM, G::DoF, Eigen::RowMajor>> J(jacobian);
    J.setZero();
    J.template block<G::DoF, G::DoF>(0,0).setIdentity();
    return true;
  }

  virtual int GlobalSize() const
  {
    return G::MEM;
  }

  virtual int LocalSize() const
  {
    return G::DoF;
  }

};