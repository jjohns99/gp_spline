#pragma once

#include <memory>
#include <Eigen/Dense>
#include "estimator/rn_spline_motion_prior.hpp"

namespace estimator
{

template <typename SplineT>
class LieSplineMotionPrior
{
private:
  using G = typename SplineT::NonmapT;
  static const int DoF = G::DoF;
  using VectorDoF = Eigen::Matrix<double, DoF, 1>;
  using MatrixDoF = Eigen::Matrix<double, DoF, DoF>;

public:
  LieSplineMotionPrior()
  {}

  LieSplineMotionPrior(ModelType type, MatrixDoF Qc) : type_{type}, Qc_{Qc}, Qc_inv_{Qc_.inverse()}
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
    G T1, T2;
    VectorDoF v1, v2, a1, a2;
    std::vector<MatrixDoF> T1_jacs, T2_jacs, v1_jacs, v2_jacs, a1_jacs, a2_jacs;
    spline->eval(t1, &T1, &v1, &a1, (jacs == nullptr) ? nullptr : &T1_jacs, (jacs == nullptr) ? nullptr : &v1_jacs,
      (jacs == nullptr || type_ != ModelType::JERK) ? nullptr : &a1_jacs);
    spline->eval(t2, &T2, &v2, &a2, (jacs == nullptr) ? nullptr : &T2_jacs, (jacs == nullptr) ? nullptr : &v2_jacs,
      (jacs == nullptr || type_ != ModelType::JERK) ? nullptr : &a2_jacs);

    double dt = t2 - t1;
    VectorDoF xi_ti1 = (T2 * T1.inverse()).Log();
    MatrixDoF Jl_inv_ti1 = G::Jl_inv(xi_ti1);
    VectorDoF dxi_ti1 = Jl_inv_ti1 * v2;

    Eigen::Matrix<double, S, 1> r;
    r.template head<DoF>() = xi_ti1 - dt * v1;
    r.template segment<DoF>(DoF) = dxi_ti1 - v1;
    if(type_ == ModelType::JERK)
    {
      r.template head<DoF>() -= 0.5 * dt * dt * a1;
      r.template segment<DoF>(DoF) -= dt * a1;
      r.template tail<DoF>() = Jl_inv_ti1 * a2 - 0.5 * G::ad(dxi_ti1) * v2 - a1;
    }

    if(jacs != nullptr)
    {
      // first compute jacs of r w.r.t. T1, T2, etc.
      MatrixDoF xi_ti1_T1_jac = -G::Jr_inv(xi_ti1);
      MatrixDoF xi_ti1_T2_jac = Jl_inv_ti1;
      MatrixDoF v_2_wedge = G::ad(v2);

      Eigen::Matrix<double, S, DoF> r_T1_jac, r_T2_jac, r_v1_jac, r_v2_jac, r_a1_jac, r_a2_jac;
      r_T1_jac.template block<DoF, DoF>(0,0) = xi_ti1_T1_jac;
      r_T2_jac.template block<DoF, DoF>(0,0) = xi_ti1_T2_jac;
      r_T1_jac.template block<DoF, DoF>(DoF,0) = 0.5 * v_2_wedge * xi_ti1_T1_jac;
      r_T2_jac.template block<DoF, DoF>(DoF,0) = 0.5 * v_2_wedge * xi_ti1_T2_jac;

      r_v1_jac.template block<DoF, DoF>(0,0) = -dt * MatrixDoF::Identity();
      r_v2_jac.template block<DoF, DoF>(0,0).setZero();
      r_v1_jac.template block<DoF, DoF>(DoF,0) = -MatrixDoF::Identity();
      r_v2_jac.template block<DoF, DoF>(DoF,0) = Jl_inv_ti1;
      
      if(type_ == ModelType::JERK)
      {
        MatrixDoF r_xi_ti1_jac = (0.5 * G::ad(a2) + 0.25 * v_2_wedge * v_2_wedge);
        r_T1_jac.template block<DoF, DoF>(2*DoF, 0) = r_xi_ti1_jac * xi_ti1_T1_jac;
        r_T2_jac.template block<DoF, DoF>(2*DoF, 0) = r_xi_ti1_jac * xi_ti1_T2_jac;

        r_v1_jac.template block<DoF, DoF>(2*DoF, 0).setZero();
        r_v2_jac.template block<DoF, DoF>(2*DoF, 0) = -0.5 * (G::ad(dxi_ti1) - v_2_wedge * Jl_inv_ti1);

        r_a1_jac.template block<DoF, DoF>(0,0) = -0.5 * dt * dt * MatrixDoF::Identity();
        r_a2_jac.template block<DoF, DoF>(0,0).setZero();
        r_a1_jac.template block<DoF, DoF>(DoF,0) = -dt * MatrixDoF::Identity();
        r_a2_jac.template block<DoF, DoF>(DoF,0).setZero();
        r_a1_jac.template block<DoF, DoF>(2*DoF,0) = -MatrixDoF::Identity();
        r_a2_jac.template block<DoF, DoF>(2*DoF,0) = Jl_inv_ti1;
      }

      // next, compute jacs of r w.r.t. control points
      int num_cp = (std::min(spline->get_i(t2) - spline->get_i(t1), spline->get_order()) + spline->get_order());

      for(int i = 0; i < num_cp; ++i)
      {
        Eigen::Matrix<double, S, DoF> jac = Eigen::Matrix<double, S, DoF>::Zero();
        if(i < spline->get_order())
        {
          jac += r_T1_jac * T1_jacs[i] + r_v1_jac * v1_jacs[i];
          if(type_ == ModelType::JERK) jac += r_a1_jac * a1_jacs[i];
        }
        int j = spline->get_order() + i - num_cp;
        if(j >= 0)
        {
          jac += r_T2_jac * T2_jacs[j] + r_v2_jac * v2_jacs[j];
          if(type_ == ModelType::JERK) jac += r_a2_jac * a2_jacs[j];
        }

        jacs->push_back(jac);
      }
    }

    return r;
  }

private:
  MatrixDoF Qc_;
  MatrixDoF Qc_inv_;
  ModelType type_;
};

} // namespace estimator