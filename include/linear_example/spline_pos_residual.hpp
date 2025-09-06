#pragma once

#include <memory>
#include <vector>
#include <Eigen/Dense>
#include <ceres/ceres.h>

#include "spline/rn_spline.hpp"

namespace linear_example
{

template <int DoF>
class SplinePosResidual : public ceres::CostFunction
{
private:
  using VectorDoF = Eigen::Matrix<double, DoF, 1>;
  using MatrixDoF = Eigen::Matrix<double, DoF, DoF>;
  using SplineT = spline::RnSpline<Eigen::Map<VectorDoF>, DoF>;

public:
  SplinePosResidual(double t, VectorDoF z, MatrixDoF sqrt_W, std::shared_ptr<SplineT> spl) :
    t_{t}, z_{z}, sqrt_W_{sqrt_W}, spline_{spl}
  {
    set_num_residuals(DoF);
    std::vector<int32_t> param_block_sizes;
    for(int i = 0; i < spline_->get_order(); ++i)
      param_block_sizes.push_back(DoF);

    *mutable_parameter_block_sizes() = param_block_sizes;
  }

  virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
  {
    VectorDoF p;
    std::vector<MatrixDoF> p_jacs;
    spline_->eval(t_, &p, nullptr, nullptr, (jacobians == nullptr) ? nullptr : &p_jacs);

    Eigen::Map<VectorDoF> r(residuals);
    r = sqrt_W_ * (p - z_);

    if(jacobians != nullptr)
    {
      for(int i = 0; i < p_jacs.size(); ++i)
      {
        Eigen::Map<Eigen::Matrix<double, DoF, DoF, Eigen::RowMajor>> J(jacobians[i]);
        J = sqrt_W_ * p_jacs[i];
      }
    }

    return true;
  }

private:
  double t_;
  VectorDoF z_;
  MatrixDoF sqrt_W_;
  std::shared_ptr<SplineT> spline_;
};

} // namespace linear_example