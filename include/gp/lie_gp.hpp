#pragma once

#include <vector>
#include <map>
#include <cmath>
#include <memory>
#include <Eigen/Dense>

namespace gp
{

enum ModelType
{
  ACC = 0,
  JERK = 1
};

// G = group (including maps), T = tangent (including maps)
template <typename G, typename T>
class LieGP
{
public:
  using TangentT = typename G::TangentT; // this will not be a map type, whereas T will be if G is
  using NonmapT = typename G::NonmapT;
  using JacT = typename G::JacT;
  static const int DoF = G::DoF;
  using VectorDoF = Eigen::Matrix<double, DoF, 1>;
  using MatrixDoF = Eigen::Matrix<double, DoF, DoF>;
  using Matrix2DoF = Eigen::Matrix<double, 2*DoF, 2*DoF>;
  using Matrix3DoF = Eigen::Matrix<double, 3*DoF, 3*DoF>;
  
  LieGP() : T_{nullptr}, v_{nullptr}, a_{nullptr}
  {}

  LieGP(ModelType type, MatrixDoF Q) : T_{nullptr}, v_{nullptr}, a_{nullptr}, type_{type}, Q_w_{Q}, Q_w_inv_{Q.inverse()}
  {}

  void init_est_params(std::vector<double> est_t, std::shared_ptr<std::vector<G>> p, std::shared_ptr<std::vector<T>> v, 
    std::shared_ptr<std::vector<T>> a = nullptr)
  {
    t_ = est_t;
    T_ = p;
    v_ = v;

    if(a)
    {
      assert(type_ == ModelType::JERK);
      a_ = a;
    }

    // compute transition matrix between estimation times
    for(int i = 1; i < t_.size(); ++i)
    {
      if(type_ == ModelType::ACC)
      {
        Matrix2DoF Phi_ti_i1 = Matrix2DoF::Identity();
        get_Phi<2*DoF>(t_[i-1], t_[i], Phi_ti_i1);
        Phi_ti_ti1_.push_back(Phi_ti_i1);

        Matrix2DoF Q_inv_ti_i1;
        Q_inv_ti_i1 << Q_w_inv_, Q_w_inv_, Q_w_inv_, Q_w_inv_;
        get_Q_inv<2*DoF>(t_[i-1], t_[i], Q_inv_ti_i1);
        Q_inv_ti_ti1_.push_back(Q_inv_ti_i1);
      }
      else 
      {
        Matrix3DoF Phi_ti_i1 = Matrix3DoF::Identity();
        get_Phi<3*DoF>(t_[i-1], t_[i], Phi_ti_i1);
        Phi_ti_ti1_.push_back(Phi_ti_i1);

        Matrix3DoF Q_inv_ti_i1;
        Q_inv_ti_i1 << Q_w_inv_, Q_w_inv_, Q_w_inv_, Q_w_inv_, Q_w_inv_, Q_w_inv_, Q_w_inv_, Q_w_inv_, Q_w_inv_;
        get_Q_inv<3*DoF>(t_[i-1], t_[i], Q_inv_ti_i1);
        Q_inv_ti_ti1_.push_back(Q_inv_ti_i1);
      }
    }
  }

  int get_i(double t)
  {
    assert(is_time_valid(t));
    int i = 0;
    while(t >= t_[i+1]) ++i;
    return i;
  }

  bool is_time_valid(double t)
  {
    return (t >= t_[0] && t <= t_.back());
  }

  // Phi should be identity when passed in
  template <int S>
  void get_Phi(double t1, double t2, Eigen::Matrix<double, S, S>& Phi)
  {
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
  }

  // Q should be matrix of Q_w_ when passed in (i.e. [Q_w_ Q_w_ ...]
  //                                                 [Q_w_ Q_w_ ...]
  //                                                 [...  ...  ...])
  template <int S>
  void get_Q(double t1, double t2, Eigen::Matrix<double, S, S>& Q)
  {
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
  }

  // Q_inv should be matrix of Q_w_inv_ when passed in
  template <int S>
  void get_Q_inv(double t1, double t2, Eigen::Matrix<double, S, S>& Q_inv)
  {
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
  }

  // jacobian vectors will be in the order (T_i, T_i1, v_i, v_i1, a_i, a_i1), as applicable
  int eval(double t, NonmapT* d0_ret = nullptr, TangentT* d1_ret = nullptr, TangentT* d2_ret = nullptr, 
    std::vector<JacT>* J0_ret = nullptr, std::vector<JacT>* J1_ret = nullptr, std::vector<JacT>* J2_ret = nullptr)
  {
    int i = get_i(t);

    // get Lambda, Omega
    Eigen::MatrixXd Lambda, Omega;
    if(type_ == ModelType::ACC)
    {
      Matrix2DoF Phi_ti_t = Matrix2DoF::Identity();
      Matrix2DoF Phi_t_ti1 = Matrix2DoF::Identity();
      get_Phi<2*DoF>(t_[i], t, Phi_ti_t);
      get_Phi<2*DoF>(t, t_[i+1], Phi_t_ti1);

      Matrix2DoF Q;
      Q << Q_w_, Q_w_, Q_w_, Q_w_;
      get_Q<2*DoF>(t_[i], t, Q);

      Omega = Q * Phi_t_ti1.transpose() * Q_inv_ti_ti1_[i];
      Lambda = Phi_ti_t - Omega * Phi_ti_ti1_[i];
    }
    else
    {
      Matrix3DoF Phi_ti_t = Matrix3DoF::Identity();
      Matrix3DoF Phi_t_ti1 = Matrix3DoF::Identity();
      get_Phi<3*DoF>(t_[i], t, Phi_ti_t);
      get_Phi<3*DoF>(t, t_[i+1], Phi_t_ti1);

      Matrix3DoF Q;
      Q << Q_w_, Q_w_, Q_w_, Q_w_, Q_w_, Q_w_, Q_w_, Q_w_, Q_w_;
      get_Q<3*DoF>(t_[i], t, Q);

      Omega = Q * Phi_t_ti1.transpose() * Q_inv_ti_ti1_[i];
      Lambda = Phi_ti_t - Omega * Phi_ti_ti1_[i];
    }

    // compute xi_i(t_{i+1})
    VectorDoF xi_ti1 = (T_->at(i+1) * T_->at(i).inverse()).Log();

    MatrixDoF Jl_inv_ti1 = G::Jl_inv(xi_ti1);
    VectorDoF dxi_ti1 = Jl_inv_ti1 * v_->at(i+1);
    VectorDoF ddxi_ti1;
    if(type_ == ModelType::JERK) ddxi_ti1 = Jl_inv_ti1 * a_->at(i+1) - 0.5 * G::ad(dxi_ti1) * v_->at(i+1);

    VectorDoF xi_t = Omega.topLeftCorner(DoF, DoF) * xi_ti1 + Lambda.template block<DoF, DoF>(0, DoF) * v_->at(i) 
                      + Omega.template block<DoF, DoF>(0, DoF) * dxi_ti1;
    if(type_ == ModelType::JERK) xi_t += Lambda.template block<DoF, DoF>(0, 2*DoF) * a_->at(i) + Omega.template block<DoF, DoF>(0, 2*DoF) * ddxi_ti1;

    VectorDoF dxi_t, ddxi_t;
    MatrixDoF Jl_xi_t;
    if(d1_ret || d2_ret || J1_ret || J2_ret)
    {
      dxi_t = Lambda.template block<DoF, DoF>(DoF, DoF) * v_->at(i) + Omega.template block<DoF, DoF>(DoF, DoF) * dxi_ti1
              + Omega.template block<DoF, DoF>(DoF, 0) * xi_ti1;      
      if(type_ == ModelType::JERK)
      {
        dxi_t += Lambda.template block<DoF, DoF>(DoF, 2*DoF) * a_->at(i) + Omega.template block<DoF, DoF>(DoF, 2*DoF) * ddxi_ti1;
        if(d2_ret || J2_ret)
          ddxi_t = Lambda.template block<DoF, DoF>(2*DoF, DoF) * v_->at(i) + Lambda.template block<DoF, DoF>(2*DoF, 2*DoF) * a_->at(i)
                  + Omega.template block<DoF, DoF>(2*DoF, 0) * xi_ti1 + Omega.template block<DoF, DoF>(2*DoF, DoF) * dxi_ti1
                  + Omega.template block<DoF, DoF>(2*DoF, 2*DoF) * ddxi_ti1;
      }

      Jl_xi_t = G::Jl(xi_t);
    }
    
    NonmapT T_t;
    TangentT v_t, a_t;
    if(d0_ret)
    {
      T_t = G::Exp(xi_t) * T_->at(i);
      *d0_ret = T_t;
    }
    if(d1_ret || J2_ret)
    {
      v_t = Jl_xi_t * dxi_t;
      if(d1_ret) *d1_ret = v_t;
    } 
    if(d2_ret)
    {
      a_t = Jl_xi_t * (ddxi_t + 0.5 * G::ad(dxi_t) * v_t);
      *d2_ret = a_t;
    }

    // jacobians
    if(J0_ret || J1_ret || J2_ret)
    {
      MatrixDoF xi_ti1_Ti_jac = -G::Jr_inv(xi_ti1);
      MatrixDoF xi_ti1_Ti1_jac = Jl_inv_ti1;

      MatrixDoF xi_t_xi_ti1_jac = Omega.topLeftCorner(DoF, DoF) + 0.5 * Omega.template block<DoF, DoF>(0, DoF) * G::ad(v_->at(i+1));;
      MatrixDoF dxi_t_xi_ti1_jac, ddxi_t_xi_ti1_jac;
      if(J1_ret || J2_ret)
        dxi_t_xi_ti1_jac = Omega.template block<DoF, DoF>(DoF, 0) + 0.5 * Omega.template block<DoF, DoF>(DoF, DoF) * G::ad(v_->at(i+1));
      if(type_ == ModelType::JERK)
      {
        MatrixDoF dd = (0.5 * G::ad(v_->at(i+1)) * G::ad(v_->at(i+1)) + G::ad(a_->at(i+1)));
        xi_t_xi_ti1_jac += 0.5 * Omega.template block<DoF, DoF>(0, 2*DoF) * dd;
        if(J1_ret || J2_ret)
        {
          dxi_t_xi_ti1_jac += 0.5 * Omega.template block<DoF, DoF>(DoF, 2*DoF) * dd;
          if(J2_ret)
            ddxi_t_xi_ti1_jac = Omega.template block<DoF, DoF>(2*DoF, 0) + 0.5 * Omega.template block<DoF, DoF>(2*DoF, DoF) * G::ad(v_->at(i+1))
                                + 0.5 * Omega.template block<DoF, DoF>(2*DoF, 2*DoF) * dd;
        }
      }

      // compute all local variable jacobians
      MatrixDoF xi_t_Ti_jac = xi_t_xi_ti1_jac * xi_ti1_Ti_jac;
      MatrixDoF xi_t_Ti1_jac = xi_t_xi_ti1_jac * xi_ti1_Ti1_jac;
      MatrixDoF xi_t_vi1_jac = Omega.template block<DoF, DoF>(0, DoF) * Jl_inv_ti1;
      MatrixDoF xi_t_ai1_jac;
      MatrixDoF ee = (G::ad(dxi_ti1) - G::ad(v_->at(i+1)) * Jl_inv_ti1);
      if(type_ == ModelType::JERK)
      {
        xi_t_vi1_jac -= 0.5 * Omega.template block<DoF, DoF>(0, 2*DoF) * ee;
        xi_t_ai1_jac = Omega.template block<DoF, DoF>(0, 2*DoF) * Jl_inv_ti1;
      }

      MatrixDoF dxi_t_Ti_jac, dxi_t_Ti1_jac, dxi_t_vi1_jac, dxi_t_ai1_jac;
      if(J1_ret || J2_ret)
      {
        dxi_t_Ti_jac = dxi_t_xi_ti1_jac * xi_ti1_Ti_jac;
        dxi_t_Ti1_jac = dxi_t_xi_ti1_jac * xi_ti1_Ti1_jac;
        dxi_t_vi1_jac = Omega.template block<DoF, DoF>(DoF, DoF) * Jl_inv_ti1;
        if(type_ == ModelType::JERK)
        {
          dxi_t_vi1_jac -= 0.5 * Omega.template block<DoF, DoF>(DoF, 2*DoF) * ee;
          dxi_t_ai1_jac = Omega.template block<DoF, DoF>(DoF, 2*DoF) * Jl_inv_ti1;
        }
      }

      MatrixDoF ddxi_t_Ti_jac, ddxi_t_Ti1_jac, ddxi_t_vi1_jac, ddxi_t_ai1_jac;
      if(J2_ret)
      {
        ddxi_t_Ti_jac = ddxi_t_xi_ti1_jac * xi_ti1_Ti_jac;
        ddxi_t_Ti1_jac = ddxi_t_xi_ti1_jac * xi_ti1_Ti1_jac;
        ddxi_t_vi1_jac = Omega.template block<DoF, DoF>(2*DoF, DoF) * Jl_inv_ti1
                         - 0.5 * Omega.template block<DoF, DoF>(2*DoF, 2*DoF) * ee;
        ddxi_t_ai1_jac = Omega.template block<DoF, DoF>(2*DoF, 2*DoF) * Jl_inv_ti1;
      }
      
      if(J0_ret)
      {
        J0_ret->push_back(G::Exp(xi_t).Ad() + Jl_xi_t * xi_t_Ti_jac);
        J0_ret->push_back(Jl_xi_t * xi_t_Ti1_jac);
        J0_ret->push_back(Jl_xi_t * Lambda.template block<DoF, DoF>(0, DoF));
        J0_ret->push_back(Jl_xi_t * xi_t_vi1_jac);
        if(type_ == ModelType::JERK) 
        {
          J0_ret->push_back(Jl_xi_t * Lambda.template block<DoF, DoF>(0, 2*DoF));
          J0_ret->push_back(Jl_xi_t * xi_t_ai1_jac);
        }
      }

      MatrixDoF v_Ti_jac, v_Ti1_jac, v_vi_jac, v_vi1_jac, v_ai_jac, v_ai1_jac;
      if(J1_ret || J2_ret)
      {
        MatrixDoF v_xi_t_jac = -0.5 * G::ad(dxi_t);
        v_Ti_jac = v_xi_t_jac * xi_t_Ti_jac + Jl_xi_t * dxi_t_Ti_jac;
        v_Ti1_jac = v_xi_t_jac * xi_t_Ti1_jac + Jl_xi_t * dxi_t_Ti1_jac;
        v_vi_jac = v_xi_t_jac * Lambda.template block<DoF, DoF>(0, DoF) + Jl_xi_t * Lambda.template block<DoF, DoF>(DoF, DoF);
        v_vi1_jac = v_xi_t_jac * xi_t_vi1_jac + Jl_xi_t * dxi_t_vi1_jac;
        if(type_ == ModelType::JERK)
        {
          v_ai_jac = v_xi_t_jac * Lambda.template block<DoF, DoF>(0, 2*DoF) + Jl_xi_t * Lambda.template block<DoF, DoF>(DoF, 2*DoF);
          v_ai1_jac = v_xi_t_jac * xi_t_ai1_jac + Jl_xi_t * dxi_t_ai1_jac;
        }

        if(J1_ret)
        {
          J1_ret->push_back(v_Ti_jac);
          J1_ret->push_back(v_Ti1_jac);
          J1_ret->push_back(v_vi_jac);
          J1_ret->push_back(v_vi1_jac);
          if(type_ == ModelType::JERK)
          {
            J1_ret->push_back(v_ai_jac);
            J1_ret->push_back(v_ai1_jac);
          }
        }
      }

      if(J2_ret)
      {
        MatrixDoF a_xi_t_jac = -0.5 * G::ad(ddxi_t + 0.5 * G::ad(dxi_t) * v_t);
        MatrixDoF a_dxi_t_jac = -0.5 * Jl_xi_t * G::ad(v_t);
        MatrixDoF a_v_jac = 0.5 * Jl_xi_t * G::ad(dxi_t);

        J2_ret->push_back(a_xi_t_jac * xi_t_Ti_jac + a_dxi_t_jac * dxi_t_Ti_jac + Jl_xi_t * ddxi_t_Ti_jac + a_v_jac * v_Ti_jac);
        J2_ret->push_back(a_xi_t_jac * xi_t_Ti1_jac + a_dxi_t_jac * dxi_t_Ti1_jac + Jl_xi_t * ddxi_t_Ti1_jac + a_v_jac * v_Ti1_jac);
        J2_ret->push_back(a_xi_t_jac * Lambda.template block<DoF, DoF>(0, DoF) + a_dxi_t_jac * Lambda.template block<DoF, DoF>(DoF, DoF) 
          + Jl_xi_t * Lambda.template block<DoF, DoF>(2*DoF, DoF) + a_v_jac * v_vi_jac);
        J2_ret->push_back(a_xi_t_jac * xi_t_vi1_jac + a_dxi_t_jac * dxi_t_vi1_jac + Jl_xi_t * ddxi_t_vi1_jac + a_v_jac * v_vi1_jac);
        J2_ret->push_back(a_xi_t_jac * Lambda.template block<DoF, DoF>(0, 2*DoF) + a_dxi_t_jac * Lambda.template block<DoF, DoF>(DoF, 2*DoF) 
          + Jl_xi_t * Lambda.template block<DoF, DoF>(2*DoF, 2*DoF) + a_v_jac * v_ai_jac);
        J2_ret->push_back(a_xi_t_jac * xi_t_ai1_jac + a_dxi_t_jac * dxi_t_ai1_jac + Jl_xi_t * ddxi_t_ai1_jac + a_v_jac * v_ai1_jac);
      }
    }

    return i;
  }

  // get dynamics residual between t_i and t_{i+1}
  template <int S>
  Eigen::Matrix<double, S, 1> get_dynamics_residual(int i, std::vector<Eigen::Matrix<double, S, DoF>>* jacs = nullptr)
  {
    double dt = t_[i+1] - t_[i];
    VectorDoF xi_ti1 = (T_->at(i+1) * T_->at(i).inverse()).Log();
    MatrixDoF Jl_inv_ti1 = G::Jl_inv(xi_ti1);
    VectorDoF dxi_ti1 = Jl_inv_ti1 * v_->at(i+1);

    Eigen::Matrix<double, S, 1> r;
    r.template head<DoF>() = xi_ti1 - dt * v_->at(i);
    r.template segment<DoF>(DoF) = dxi_ti1 - v_->at(i);
    if(type_ == ModelType::JERK)
    {
      r.template head<DoF>() -= 0.5 * dt * dt * a_->at(i);
      r.template segment<DoF>(DoF) -= dt * a_->at(i);
      r.template tail<DoF>() = Jl_inv_ti1 * a_->at(i+1) - 0.5 * G::ad(dxi_ti1) * v_->at(i+1) - a_->at(i);
    }

    if(jacs)
    {
      MatrixDoF xi_ti1_Ti_jac = -G::Jr_inv(xi_ti1);
      MatrixDoF xi_ti1_Ti1_jac = Jl_inv_ti1;
      MatrixDoF v_i1_wedge = G::ad(v_->at(i+1));

      Eigen::Matrix<double, S, DoF> r_Ti_jac, r_Ti1_jac, r_vi_jac, r_vi1_jac, r_ai_jac, r_ai1_jac;
      r_Ti_jac.template block<DoF, DoF>(0,0) = xi_ti1_Ti_jac;
      r_Ti1_jac.template block<DoF, DoF>(0,0) = xi_ti1_Ti1_jac;
      r_Ti_jac.template block<DoF, DoF>(DoF,0) = 0.5 * v_i1_wedge * xi_ti1_Ti_jac;
      r_Ti1_jac.template block<DoF, DoF>(DoF,0) = 0.5 * v_i1_wedge * xi_ti1_Ti1_jac;

      r_vi_jac.template block<DoF, DoF>(0,0) = -dt * MatrixDoF::Identity();
      r_vi1_jac.template block<DoF, DoF>(0,0).setZero();
      r_vi_jac.template block<DoF, DoF>(DoF,0) = -MatrixDoF::Identity();
      r_vi1_jac.template block<DoF, DoF>(DoF,0) = Jl_inv_ti1;
      
      if(type_ == ModelType::JERK)
      {
        MatrixDoF r_xi_ti1_jac = (0.5 * G::ad(a_->at(i+1)) + 0.25 * v_i1_wedge * v_i1_wedge);
        r_Ti_jac.template block<DoF, DoF>(2*DoF, 0) = r_xi_ti1_jac * xi_ti1_Ti_jac;
        r_Ti1_jac.template block<DoF, DoF>(2*DoF, 0) = r_xi_ti1_jac * xi_ti1_Ti1_jac;

        r_vi_jac.template block<DoF, DoF>(2*DoF, 0).setZero();
        r_vi1_jac.template block<DoF, DoF>(2*DoF, 0) = -0.5 * (G::ad(dxi_ti1) - v_i1_wedge * Jl_inv_ti1);

        r_ai_jac.template block<DoF, DoF>(0,0) = -0.5 * dt * dt * MatrixDoF::Identity();
        r_ai1_jac.template block<DoF, DoF>(0,0).setZero();
        r_ai_jac.template block<DoF, DoF>(DoF,0) = -dt * MatrixDoF::Identity();
        r_ai1_jac.template block<DoF, DoF>(DoF,0).setZero();
        r_ai_jac.template block<DoF, DoF>(2*DoF,0) = -MatrixDoF::Identity();
        r_ai1_jac.template block<DoF, DoF>(2*DoF,0) = Jl_inv_ti1;
      }

      jacs->push_back(r_Ti_jac);
      jacs->push_back(r_Ti1_jac);
      jacs->push_back(r_vi_jac);
      jacs->push_back(r_vi1_jac);
      if(type_ == ModelType::JERK)
      {
        jacs->push_back(r_ai_jac);
        jacs->push_back(r_ai1_jac);
      }
    }

    return r;
  }

  // preevaluate gp and jacobians to be accessed when get_pre_eval is called.
  // this allows for not having to recompute for the same time instance multiple
  // times, e.g. when computing feature tracking residuals
  //
  // erases and rewrites all pre-evaluated data when called
  void pre_eval(std::vector<double> t_vec, bool d0, bool d1, bool d2, bool j0, bool j1, bool j2)
  {
    clear_pre_evals();

    for(double t : t_vec)
    {
      NonmapT T_t;
      TangentT d1_t, d2_t;
      std::vector<JacT> j0_t, j1_t, j2_t;
      eval(t, d0 ? &T_t : nullptr, d1 ? &d1_t : nullptr, (d2 && type_ == ModelType::JERK) ? &d2_t : nullptr,
        j0 ? &j0_t : nullptr, j1 ? &j1_t : nullptr, (j2 && type_ == ModelType::JERK) ? &j2_t : nullptr);
      
      if(d0) T_pre_eval_.push_back(T_t);
      if(d1) d1_pre_eval_.push_back(d1_t);
      if(d2 && type_ == ModelType::JERK) d2_pre_eval_.push_back(d2_t);
      if(j0) j0_pre_eval_.push_back(j0_t);
      if(j1) j1_pre_eval_.push_back(j1_t);
      if(j2 && type_ == ModelType::JERK) j2_pre_eval_.push_back(j2_t);
    }
  }

  // pre_eval_id is the index of the time in t_vec given to pre_eval()
  void get_pre_eval(int pre_eval_id, NonmapT* d0 = nullptr, TangentT* d1 = nullptr, TangentT* d2 = nullptr,
    std::vector<JacT>* J0 = nullptr, std::vector<JacT>* J1 = nullptr, std::vector<JacT>* J2 = nullptr)
  {
    if(d0 != nullptr)
    {
      assert(pre_eval_id < T_pre_eval_.size());
      *d0 = T_pre_eval_[pre_eval_id];
    }

    if(d1 != nullptr)
    {
      assert(pre_eval_id < d1_pre_eval_.size());
      *d1 = d1_pre_eval_[pre_eval_id];
    }

    if(d2 != nullptr)
    {
      assert(pre_eval_id < d2_pre_eval_.size());
      *d2 = d2_pre_eval_[pre_eval_id];
    }

    if(J0 != nullptr)
    {
      assert(pre_eval_id < j0_pre_eval_.size());
      *J0 = j0_pre_eval_[pre_eval_id];
    }

    if(J1 != nullptr)
    {
      assert(pre_eval_id < j1_pre_eval_.size());
      *J1 = j1_pre_eval_[pre_eval_id];
    }

    if(J2 != nullptr)
    {
      assert(pre_eval_id < j2_pre_eval_.size());
      *J2 = j2_pre_eval_[pre_eval_id];
    }
  }

  bool modify_T(int i, G new_T)
  {
    if(i >= T_->size()) return false;
    T_->at(i) = new_T;
    return true;
  }

  bool modify_v(int i, TangentT new_v)
  {
    if(i >= v_->size()) return false;
    v_->at(i) = new_v;
    return true;
  }

  bool modify_a(int i, TangentT new_a)
  {
    if(type_ != ModelType::JERK || i >= a_->size()) return false;
    a_->at(i) = new_a;
    return true;
  }

  std::shared_ptr<std::vector<G>> get_T()
  {
    return T_;
  }

  std::shared_ptr<std::vector<T>> get_v()
  {
    return v_;
  }

  std::shared_ptr<std::vector<T>> get_a()
  {
    return a_;
  }

  ModelType get_type()
  {
    return type_;
  }

  Eigen::MatrixXd get_Q_inv_i_i1(int i)
  {
    return Q_inv_ti_ti1_[i];
  }

private:
  void clear_pre_evals()
  {
    T_pre_eval_.clear();
    d1_pre_eval_.clear();
    d2_pre_eval_.clear();
    j0_pre_eval_.clear();
    j1_pre_eval_.clear();
    j2_pre_eval_.clear();
  }


  ModelType type_;

  std::vector<double> t_;
  std::shared_ptr<std::vector<G>> T_;
  std::shared_ptr<std::vector<T>> v_;
  std::shared_ptr<std::vector<T>> a_;

  // vector of transition matrices between knots
  std::vector<Eigen::MatrixXd> Phi_ti_ti1_;
  std::vector<Eigen::MatrixXd> Q_inv_ti_ti1_;

  MatrixDoF Q_w_;
  MatrixDoF Q_w_inv_;

  // pre-evaluated values
  std::vector<NonmapT> T_pre_eval_;
  std::vector<TangentT> d1_pre_eval_;
  std::vector<TangentT> d2_pre_eval_;
  std::vector<std::vector<JacT>> j0_pre_eval_;
  std::vector<std::vector<JacT>> j1_pre_eval_;
  std::vector<std::vector<JacT>> j2_pre_eval_;
};

} // namespace gp