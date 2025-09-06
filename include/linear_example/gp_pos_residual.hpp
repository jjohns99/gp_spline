#pragma once

#include <memory>
#include <vector>
#include <Eigen/Dense>
#include <ceres/ceres.h>

#include "gp/rn_gp.hpp"

namespace linear_example
{

template <int DoF>
class GPPosResidual : public ceres::CostFunction
{
private:
  using VectorDoF = Eigen::Matrix<double, DoF, 1>;
  using MatrixDoF = Eigen::Matrix<double, DoF, DoF>;
  using GP = gp::RnGP<Eigen::Map<VectorDoF>, DoF>;

public:
  GPPosResidual(double t, VectorDoF z, MatrixDoF sqrt_W, std::shared_ptr<GP> gp) :
    t_{t}, z_{z}, sqrt_W_{sqrt_W}, gp_{gp}
  {
    set_num_residuals(DoF);
    std::vector<int32_t> param_block_sizes;
    for(int i = 0; i < (gp_->get_type() == gp::ModelType::ACC ? 4 : 6); ++i)
      param_block_sizes.push_back(DoF);
    *mutable_parameter_block_sizes() = param_block_sizes;
  }

  virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
  {
    VectorDoF p;
    std::vector<MatrixDoF> p_jacs;
    gp_->eval(t_, &p, nullptr, nullptr, (jacobians == nullptr) ? nullptr : &p_jacs);

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
  std::shared_ptr<GP> gp_;
};

} // namespace linear_example