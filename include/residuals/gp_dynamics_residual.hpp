#pragma once

#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <ceres/ceres.h>

#include "gp/lie_gp.hpp"

namespace estimator
{

template <typename GP, int S, int MEM, int DoF>
class GPDynamicsResidual : public ceres::CostFunction
{
public:
  GPDynamicsResidual(Eigen::Matrix<double, S, S> sqrt_W, std::shared_ptr<GP> gp, int param_index) : 
    sqrt_W_{sqrt_W}, gp_{gp}, index_{param_index}
  {
    set_num_residuals(S);

    std::vector<int32_t> param_block_sizes;
    // estimation parameters
    for(int i = 0; i < ((gp_->get_type() == gp::ModelType::ACC) ? 4 : 6); ++i)
    {
      if(i < 2) param_block_sizes.push_back(MEM);
      else param_block_sizes.push_back(DoF);
    }

    *mutable_parameter_block_sizes() = param_block_sizes;
  }

  virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
  {
    Eigen::Map<Eigen::Matrix<double, S, 1>> r(residuals);

    std::vector<Eigen::Matrix<double, S, DoF>> jacs;
    r = sqrt_W_ * gp_->template get_dynamics_residual<S>(index_, jacobians == nullptr ? nullptr : &jacs);

    if(jacobians != nullptr)
    {
      for(int i = 0; i < jacs.size(); ++i)
      {
        if(i < 2)
        {
          Eigen::Map<Eigen::Matrix<double, S, MEM, Eigen::RowMajor>> J(jacobians[i]);
          J.col(MEM-1).setZero();
          J.template block<S, DoF>(0,0) = sqrt_W_ * jacs[i];
        }
        else
        {
          Eigen::Map<Eigen::Matrix<double, S, DoF, Eigen::RowMajor>> J(jacobians[i]);
          J = sqrt_W_ * jacs[i];
        }
      }
    }

    return true;
  }

private:
  Eigen::Matrix<double, S, S> sqrt_W_;
  std::shared_ptr<GP> gp_;
  int index_;
};

} // namespace estimator