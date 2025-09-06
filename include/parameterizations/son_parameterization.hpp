#pragma once

#include <ceres/ceres.h>
#include <Eigen/Dense>

// works for any lie_group SOn type (must use G::NonmapT as G)
template <typename G>
class SOnParameterization : public ceres::LocalParameterization
{
public:
  virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const
  {
    lie_groups::Map<const G> R(x);
    Eigen::Map<const typename G::TangentT> phi(delta);
    lie_groups::Map<G> R_plus_delta(x_plus_delta);
    R_plus_delta = G::Exp(phi) * R;
    
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