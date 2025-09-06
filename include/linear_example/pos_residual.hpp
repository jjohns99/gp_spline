#pragma once

#include <memory>
#include <Eigen/Dense>
#include <ceres/ceres.h>

namespace linear_example
{

template <int DoF>
class PosResidual : public ceres::SizedCostFunction<DoF, DoF>
{
private:
  using VectorDoF = Eigen::Matrix<double, DoF, 1>;
  using MatrixDoF = Eigen::Matrix<double, DoF, DoF>;

public:
  PosResidual(double t, VectorDoF z, MatrixDoF sqrt_W) :
    t_{t}, z_{z}, sqrt_W_{sqrt_W}
  {}

  virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
  {
    Eigen::Map<const VectorDoF> p(parameters[0]);

    Eigen::Map<VectorDoF> r(residuals);
    r = sqrt_W_ * (p - z_);

    if(jacobians != nullptr)
    {
      Eigen::Map<Eigen::Matrix<double, DoF, DoF, Eigen::RowMajor>> J(jacobians[0]);
      J = sqrt_W_;
    }

    return true;
  }

private:
  double t_;
  VectorDoF z_;
  MatrixDoF sqrt_W_;
};

} // namespace linear_example