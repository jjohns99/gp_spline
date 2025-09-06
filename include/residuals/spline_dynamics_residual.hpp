#pragma once

#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <ceres/ceres.h>

namespace estimator
{

template <typename SplineT, typename MP, int S, int MEM, int DoF>
class SplineDynamicsResidual : public ceres::CostFunction
{
private:
  using VectorDoF = Eigen::Matrix<double, DoF, 1>;
  using MatrixDoF = Eigen::Matrix<double, DoF, DoF>;
  using VectorS = Eigen::Matrix<double, S, 1>;
  using MatrixS = Eigen::Matrix<double, S, S>;

public:
  SplineDynamicsResidual(double t1, double t2, MatrixS sqrt_W, std::shared_ptr<MP> mp, std::shared_ptr<SplineT> spl) :
    t1_{t1}, t2_{t2}, sqrt_W_{sqrt_W}, mp_{mp}, spline_{spl}
  {
    set_num_residuals(S);

    num_params_ = (std::min(spline_->get_i(t2_) - spline_->get_i(t1_), spline_->get_order()) + spline_->get_order());
    std::vector<int32_t> param_block_sizes;
    for(int i = 0; i < num_params_; ++i) param_block_sizes.push_back(MEM);
    *mutable_parameter_block_sizes() = param_block_sizes;
  }

  virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
  {
    Eigen::Map<VectorS> r(residuals);

    std::vector<Eigen::Matrix<double, S, DoF>> jacs;

    r = sqrt_W_ * mp_->template get_motion_prior<S>(t1_, t2_, spline_, (jacobians == nullptr) ? nullptr : &jacs);

    if(jacobians != nullptr)
    {
      for(int i = 0; i < num_params_; ++i)
      {
        Eigen::Map<Eigen::Matrix<double, S, MEM, (MEM == 1 ? Eigen::ColMajor : Eigen::RowMajor)>> J(jacobians[i]);
        J.col(MEM-1).setZero();
        J.template block<S, DoF>(0,0) = sqrt_W_ * jacs[i];
      }
    }

    return true;
  }

private:
  double t1_;
  double t2_;
  std::shared_ptr<SplineT> spline_;
  std::shared_ptr<MP> mp_;

  int num_params_;

  MatrixS sqrt_W_;
};

} // namespace estimator