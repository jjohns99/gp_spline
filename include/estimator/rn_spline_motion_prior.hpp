#pragma once

#include <Eigen/Dense>
#include "spline/rn_spline.hpp"

namespace estimator
{

enum ModelType
{
  ACC = 0,
  JERK = 1
};

// this class computes Phi and Q for the same models as GP estimation,
// to be used in spline-based estimation dynamic residuals
template <typename SplineT, int DoF>
class RnSplineMotionPrior
{
private:
  using VectorDoF = Eigen::Matrix<double, DoF, 1>;
  using MatrixDoF = Eigen::Matrix<double, DoF, DoF>;

public:
  RnSplineMotionPrior()
  {}

  RnSplineMotionPrior(ModelType type, MatrixDoF Qc) : type_{type}, Qc_{Qc}, Qc_inv_{Qc_.inverse()}
  {}

  // S = 2*DoF if type_ == ACC, S = 3*DoF if type_ == JERK
  template <int S>
  Eigen::Matrix<double, S, S> get_Phi(double t1, double t2)
  {
    Eigen::Matrix<double, S, S> Phi = Eigen::Matrix<double, S, S>::Identity();
    double dt = t2 - t1;
    if(type_ == ModelType::ACC)
    {
      Phi.topRightCorner(DoF, DoF) = (dt) * MatrixDoF::Identity();
    }
    else
    {
      Phi.template block<DoF, DoF>(0, DoF) = (dt) * MatrixDoF::Identity();
      Phi.template block<DoF, DoF>(DoF, 2*DoF) = (dt) * MatrixDoF::Identity();
      Phi.topRightCorner(DoF, DoF) = 0.5 * std::pow(dt, 2.0) * MatrixDoF::Identity();
    }

    return Phi;
  }

  template <int S>
  Eigen::Matrix<double, S, S> get_Q(double t1, double t2)
  {
    Eigen::Matrix<double, S, S> Q;
    if(type_ == ModelType::ACC) Q << Qc_, Qc_, Qc_, Qc_;
    else Q << Qc_, Qc_, Qc_, Qc_, Qc_, Qc_, Qc_, Qc_, Qc_;
    double dt = t2-t1;

    Q.bottomRightCorner(DoF, DoF) *= dt;
    Q.bottomRightCorner(2*DoF, 2*DoF).topRightCorner(DoF, DoF) *= 0.5 * std::pow(dt, 2.0);
    Q.bottomRightCorner(2*DoF, 2*DoF).bottomLeftCorner(DoF, DoF) *= 0.5 * std::pow(dt, 2.0);
    Q.bottomRightCorner(2*DoF, 2*DoF).topLeftCorner(DoF, DoF) *= 1.0/3.0 * std::pow(dt, 3.0);
    if(type_ == ModelType::JERK)
    {
      Q.topRightCorner(DoF, DoF) *= 1.0/6.0 * std::pow(dt, 3.0);
      Q.bottomLeftCorner(DoF, DoF) *= 1.0/6.0 * std::pow(dt, 3.0);
      Q.template block<DoF, DoF>(0, DoF) *= 1.0/8.0 * std::pow(dt, 4.0);
      Q.template block<DoF, DoF>(DoF, 0) *= 1.0/8.0 * std::pow(dt, 4.0);
      Q.topLeftCorner(DoF, DoF) *= 1.0/20.0 * std::pow(dt, 5.0);
    }

    return Q;
  }

  // Q_inv should be matrix of Q_w_inv_ when passed in
  template <int S>
  Eigen::Matrix<double, S, S> get_Q_inv(double t1, double t2)
  {
    Eigen::Matrix<double, S, S> Q_inv;
    if(type_ == ModelType::ACC) Q_inv << Qc_inv_, Qc_inv_, Qc_inv_, Qc_inv_;
    else Q_inv << Qc_inv_, Qc_inv_, Qc_inv_, Qc_inv_, Qc_inv_, Qc_inv_, Qc_inv_, Qc_inv_, Qc_inv_;
    double dt_inv = 1.0/(t2-t1);
    if(type_ == ModelType::ACC)
    {
      Q_inv.topLeftCorner(DoF, DoF) *= 12.0 * std::pow(dt_inv, 3.0);
      Q_inv.topRightCorner(DoF, DoF) *= -6.0 * std::pow(dt_inv, 2.0);
      Q_inv.bottomLeftCorner(DoF, DoF) *= -6.0 * std::pow(dt_inv, 2.0);
      Q_inv.bottomRightCorner(DoF, DoF) *= 4.0 * dt_inv;
    }
    else
    {
      Q_inv.topLeftCorner(DoF, DoF) *= 720.0 * std::pow(dt_inv, 5);
      Q_inv.template block<DoF, DoF>(0, DoF) *= -360.0 * std::pow(dt_inv, 4);
      Q_inv.template block<DoF, DoF>(DoF, 0) *= -360.0 * std::pow(dt_inv, 4);
      Q_inv.template block<DoF, DoF>(DoF, DoF) *= 192.0 * std::pow(dt_inv, 3);
      Q_inv.topRightCorner(DoF, DoF) *= 60.0 * std::pow(dt_inv, 3.0);
      Q_inv.bottomLeftCorner(DoF, DoF) *= 60.0 * std::pow(dt_inv, 3.0);
      Q_inv.template block<DoF, DoF>(DoF, 2*DoF) *= -36.0 * std::pow(dt_inv, 2.0);
      Q_inv.template block<DoF, DoF>(2*DoF, DoF) *= -36.0 * std::pow(dt_inv, 2.0);
      Q_inv.bottomRightCorner(DoF, DoF) *= 9.0 * dt_inv;
    }

    return Q_inv;
  }

  template <int S>
  Eigen::Matrix<double, S, 1> get_motion_prior(double t1, double t2, std::shared_ptr<SplineT> spline, std::vector<Eigen::Matrix<double, S, DoF>>* jacs = nullptr)
  {
    VectorDoF p1, p2, v1, v2, a1, a2;
    std::vector<MatrixDoF> p1_jacs, p2_jacs, v1_jacs, v2_jacs, a1_jacs, a2_jacs;
    spline->eval(t1, &p1, &v1, (type_ == ModelType::JERK) ? &a1 : nullptr, (jacs == nullptr) ? nullptr : &p1_jacs,
      (jacs == nullptr) ? nullptr : &v1_jacs, (jacs == nullptr || type_ != ModelType::JERK) ? nullptr : &a1_jacs);
    spline->eval(t2, &p2, &v2, (type_ == ModelType::JERK) ? &a2 : nullptr, (jacs == nullptr) ? nullptr : &p2_jacs,
      (jacs == nullptr) ? nullptr : &v2_jacs, (jacs == nullptr || type_ != ModelType::JERK) ? nullptr : &a2_jacs);

    Eigen::Matrix<double, S, 1> gamma1, gamma2;
    gamma1.template head<DoF>() = p1;
    gamma1.template segment<DoF>(DoF) = v1;
    gamma2.template head<DoF>() = p2;
    gamma2.template segment<DoF>(DoF) = v2;
    if(type_ == ModelType::JERK)
    {
      gamma1.template tail<DoF>() = a1;
      gamma2.template tail<DoF>() = a2;
    }

    // TODO: add option to precompute this
    Eigen::Matrix<double, S, S> Phi = get_Phi<S>(t1, t2);

    if(jacs != nullptr)
    {
      int num_cp = (std::min(spline->get_i(t2) - spline->get_i(t1), spline->get_order()) + spline->get_order());

      for(int i = 0; i < num_cp; ++i)
      {
        Eigen::Matrix<double, S, DoF> jac = Eigen::Matrix<double, S, DoF>::Zero();
        if(i < spline->get_order())
        {
          if(type_ == ModelType::ACC) jac -= Phi * (Eigen::Matrix<double, S, DoF>() << p1_jacs[i], v1_jacs[i]).finished();
          else jac -= Phi * (Eigen::Matrix<double, S, DoF>() << p1_jacs[i], v1_jacs[i], a1_jacs[i]).finished();
        }
        int j = spline->get_order() + i - num_cp;
        if(j >= 0)
        {
          if(type_ == ModelType::ACC) jac += (Eigen::Matrix<double, S, DoF>() << p2_jacs[j], v2_jacs[j]).finished();
          else jac += (Eigen::Matrix<double, S, DoF>() << p2_jacs[j], v2_jacs[j], a2_jacs[j]).finished();
        }

        jacs->push_back(jac);
      }
    }

    return gamma2 - Phi * gamma1;
  }

private:
  ModelType type_;
  MatrixDoF Qc_;
  MatrixDoF Qc_inv_;
};

} // namespace estimator