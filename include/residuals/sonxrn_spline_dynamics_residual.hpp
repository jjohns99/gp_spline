#pragma once

#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <ceres/ceres.h>

namespace estimator
{

// con only constrain rn or son, not both.
// SplineT should be either RnSpline or SOnSpline, not SOnxRnSpline
template <typename SplineT, typename MP, int S, int MEM, int DoF>
class SOnxRnSplineDynamicsResidual : public ceres::CostFunction
{
private:
  using VectorS = Eigen::Matrix<double, S, 1>;
  using MatrixS = Eigen::Matrix<double, S, S>;

public:
  // constraint_type = true: son, else rn
  // spl should be either the rn spline or the son spline, not the sonxrn spline
  SOnxRnSplineDynamicsResidual(double t1, double t2, MatrixS sqrt_W, std::shared_ptr<MP> mp, std::shared_ptr<SplineT> spl, bool constraint_type) :
    t1_{t1}, t2_{t2}, sqrt_W_{sqrt_W}, mp_{mp}, spline_{spl}, constraint_type_{constraint_type}
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
        J.setZero();
        if(constraint_type_) J.template block<S, DoF>(0,DoF) = sqrt_W_ * jacs[i];
        else J.template block<S, DoF>(0,0) = sqrt_W_ * jacs[i];
      }
    }

    return true;
  }

private:
  double t1_;
  double t2_;
  std::shared_ptr<SplineT> spline_;
  std::shared_ptr<MP> mp_;
  bool constraint_type_; // true for son, false for rn

  int num_params_;

  MatrixS sqrt_W_;
};

} // namespace estimator