#pragma once

#include <vector>
#include <map>
#include <cmath>
#include <memory>
#include <Eigen/Dense>

#include <gp/lie_gp.hpp>

namespace gp
{

template <typename G, int DoF>
class RnGP
{
public:
  using VectorDoF = Eigen::Matrix<double, DoF, 1>;
  using MatrixDoF = Eigen::Matrix<double, DoF, DoF>;
  using Matrix2DoF = Eigen::Matrix<double, 2*DoF, 2*DoF>;
  using Matrix3DoF = Eigen::Matrix<double, 3*DoF, 3*DoF>;

  RnGP() : p_{nullptr}, v_{nullptr}, a_{nullptr}
  {}

  RnGP(ModelType type, MatrixDoF Q) : 
    p_{nullptr}, v_{nullptr}, a_{nullptr}, type_{type}, Q_w_{Q}, Q_w_inv_{Q.inverse()}
  {}

  void init_est_params(std::vector<double> est_t, std::shared_ptr<std::vector<G>> p, std::shared_ptr<std::vector<G>> v, 
    std::shared_ptr<std::vector<G>> a = nullptr)
  {
    t_ = est_t;
    p_ = p;
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

  // jacobian vectors will be in the order (p_i, p_i1, v_i, v_i1, a_i, a_i1), as applicable
  int eval(double t, VectorDoF* d0_ret = nullptr, VectorDoF* d1_ret = nullptr, VectorDoF* d2_ret = nullptr, 
    std::vector<MatrixDoF>* J0_ret = nullptr, std::vector<MatrixDoF>* J1_ret = nullptr, std::vector<MatrixDoF>* J2_ret = nullptr)
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

    Eigen::VectorXd gamma_i(type_ == ModelType::ACC ? 2*DoF : 3*DoF);
    Eigen::VectorXd gamma_i1(type_ == ModelType::ACC ? 2*DoF : 3*DoF);
    gamma_i.head(DoF) = p_->at(i);
    gamma_i.segment(DoF, DoF) = v_->at(i);
    gamma_i1.head(DoF) = p_->at(i+1);
    gamma_i1.segment(DoF, DoF) = v_->at(i+1);
    if(type_ == ModelType::JERK)
    {
      gamma_i.tail(DoF) = a_->at(i);
      gamma_i1.tail(DoF) = a_->at(i+1);
    }

    Eigen::VectorXd gamma_t = Lambda * gamma_i + Omega * gamma_i1;

    if(d0_ret) *d0_ret = gamma_t.head(DoF);
    if(d1_ret) *d1_ret = gamma_t.segment(DoF, DoF);
    if(d2_ret && type_ == ModelType::JERK) *d2_ret = gamma_t.tail(DoF);

    if(J0_ret)
    {
      J0_ret->push_back(Lambda.template block<DoF,DoF>(0, 0));
      J0_ret->push_back(Omega.template block<DoF,DoF>(0, 0));
      J0_ret->push_back(Lambda.template block<DoF,DoF>(0, DoF));
      J0_ret->push_back(Omega.template block<DoF,DoF>(0, DoF));
      if(type_ == ModelType::JERK)
      {
        J0_ret->push_back(Lambda.template block<DoF,DoF>(0, 2*DoF));
        J0_ret->push_back(Omega.template block<DoF,DoF>(0, 2*DoF));
      }
    }

    if(J1_ret)
    {
      J1_ret->push_back(Lambda.template block<DoF,DoF>(DoF, 0));
      J1_ret->push_back(Omega.template block<DoF,DoF>(DoF, 0));
      J1_ret->push_back(Lambda.template block<DoF,DoF>(DoF, DoF));
      J1_ret->push_back(Omega.template block<DoF,DoF>(DoF, DoF));
      if(type_ == ModelType::JERK)
      {
        J1_ret->push_back(Lambda.template block<DoF,DoF>(DoF, 2*DoF));
        J1_ret->push_back(Omega.template block<DoF,DoF>(DoF, 2*DoF));
      }
    }

    if(J2_ret)
    {
      J2_ret->push_back(Lambda.template block<DoF,DoF>(2*DoF, 0));
      J2_ret->push_back(Omega.template block<DoF,DoF>(2*DoF, 0));
      J2_ret->push_back(Lambda.template block<DoF,DoF>(2*DoF, DoF));
      J2_ret->push_back(Omega.template block<DoF,DoF>(2*DoF, DoF));
      if(type_ == ModelType::JERK)
      {
        J2_ret->push_back(Lambda.template block<DoF,DoF>(2*DoF, 2*DoF));
        J2_ret->push_back(Omega.template block<DoF,DoF>(2*DoF, 2*DoF));
      }
    }

    return i;
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
      VectorDoF d0_t, d1_t, d2_t;
      std::vector<MatrixDoF> j0_t, j1_t, j2_t;
      eval(t, d0 ? &d0_t : nullptr, d1 ? &d1_t : nullptr, (d2 && type_ == ModelType::JERK) ? &d2_t : nullptr,
        j0 ? &j0_t : nullptr, j1 ? &j1_t : nullptr, (j2 && type_ == ModelType::JERK) ? &j2_t : nullptr);
      
      if(d0) d0_pre_eval_.push_back(d0_t);
      if(d1) d1_pre_eval_.push_back(d1_t);
      if(d2 && type_ == ModelType::JERK) d2_pre_eval_.push_back(d2_t);
      if(j0) j0_pre_eval_.push_back(j0_t);
      if(j1) j1_pre_eval_.push_back(j1_t);
      if(j2 && type_ == ModelType::JERK) j2_pre_eval_.push_back(j2_t);
    }
  }

  // pre_eval_id is the index of the time in t_vec given to pre_eval()
  void get_pre_eval(int pre_eval_id, VectorDoF* d0 = nullptr, VectorDoF* d1 = nullptr, VectorDoF* d2 = nullptr,
    std::vector<MatrixDoF>* J0 = nullptr, std::vector<MatrixDoF>* J1 = nullptr, std::vector<MatrixDoF>* J2 = nullptr)
  {
    if(d0 != nullptr)
    {
      assert(pre_eval_id < d0_pre_eval_.size());
      *d0 = d0_pre_eval_[pre_eval_id];
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

  // get dynamics residual between t_i and t_{i+1}
  template <int S>
  Eigen::Matrix<double, S, 1> get_dynamics_residual(int i, std::vector<Eigen::Matrix<double, S, DoF>>* jacs = nullptr)
  {
    Eigen::VectorXd gamma_i(type_ == ModelType::ACC ? 2*DoF : 3*DoF);
    Eigen::VectorXd gamma_i1(type_ == ModelType::ACC ? 2*DoF : 3*DoF);
    gamma_i.head(DoF) = p_->at(i);
    gamma_i.segment(DoF, DoF) = v_->at(i);
    gamma_i1.head(DoF) = p_->at(i+1);
    gamma_i1.segment(DoF, DoF) = v_->at(i+1);
    if(type_ == ModelType::JERK)
    {
      gamma_i.tail(DoF) = a_->at(i);
      gamma_i1.tail(DoF) = a_->at(i+1);
    }

    Eigen::Matrix<double, S, 1> r = gamma_i1 - Phi_ti_ti1_[i] * gamma_i;

    if(jacs)
    {
      Eigen::Matrix<double, S, DoF> ej;
      ej.setZero();

      jacs->push_back(-Phi_ti_ti1_[i].block<S, DoF>(0,0));

      ej.template block<DoF,DoF>(0,0).setIdentity();
      jacs->push_back(ej);

      jacs->push_back(-Phi_ti_ti1_[i].block<S, DoF>(0,DoF));

      ej.setZero();
      ej.template block<DoF,DoF>(DoF,0).setIdentity();
      jacs->push_back(ej);

      if(type_ == ModelType::JERK)
      {
        jacs->push_back(-Phi_ti_ti1_[i].block<S, DoF>(0,2*DoF));

        ej.setZero();
        ej.template block<DoF,DoF>(2*DoF,0).setIdentity();
        jacs->push_back(ej);
      }
    }

    return r;
  }

  bool modify_p(int i, G new_p)
  {
    if(i >= p_->size()) return false;
    p_->at(i) = new_p;
    return true;
  }

  bool modify_v(int i, G new_v)
  {
    if(i >= v_->size()) return false;
    v_->at(i) = new_v;
    return true;
  }

  bool modify_a(int i, G new_a)
  {
    if(type_ != ModelType::JERK || i >= a_->size()) return false;
    a_->at(i) = new_a;
    return true;
  }

  std::shared_ptr<std::vector<G>> get_p()
  {
    return p_;
  }

  std::shared_ptr<std::vector<G>> get_v()
  {
    return v_;
  }

  std::shared_ptr<std::vector<G>> get_a()
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
    d0_pre_eval_.clear();
    d1_pre_eval_.clear();
    d2_pre_eval_.clear();
    j0_pre_eval_.clear();
    j1_pre_eval_.clear();
    j2_pre_eval_.clear();
  }


  ModelType type_;

  std::vector<double> t_;
  std::shared_ptr<std::vector<G>> p_;
  std::shared_ptr<std::vector<G>> v_;
  std::shared_ptr<std::vector<G>> a_;

  // vector of transition matrices between knots
  std::vector<Eigen::MatrixXd> Phi_ti_ti1_;
  std::vector<Eigen::MatrixXd> Q_inv_ti_ti1_;

  MatrixDoF Q_w_;
  MatrixDoF Q_w_inv_;

  // pre-evaluated values
  std::vector<VectorDoF> d0_pre_eval_;
  std::vector<VectorDoF> d1_pre_eval_;
  std::vector<VectorDoF> d2_pre_eval_;
  std::vector<std::vector<MatrixDoF>> j0_pre_eval_;
  std::vector<std::vector<MatrixDoF>> j1_pre_eval_;
  std::vector<std::vector<MatrixDoF>> j2_pre_eval_;

};

} // namespace gp