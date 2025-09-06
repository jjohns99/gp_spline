#pragma once

#include <vector>
#include <map>
#include <cmath>
#include <memory>
#include <Eigen/Dense>

#include <gp/lie_gp.hpp>
#include <gp/rn_gp.hpp>

namespace gp
{

template <typename SOn, typename T, typename Rn>
class SOnxRnGP
{
public:
  static const int SOnDoF = SOn::DoF;
  static const int RnDoF = SOn::ACT;
  static const int DoF = SOnDoF + RnDoF;
  using VectorRnDoF = Eigen::Matrix<double, RnDoF, 1>;
  using VectorSOnDoF = Eigen::Matrix<double, SOnDoF, 1>;

  using MatrixRnDoF = Eigen::Matrix<double, RnDoF, RnDoF>;
  using MatrixSOnDoF = Eigen::Matrix<double, SOnDoF, SOnDoF>;
  using MatrixDoF = Eigen::Matrix<double, DoF, DoF>;
  using SOnNonmapT = typename SOn::NonmapT;

  using RnGPT = RnGP<Rn, RnDoF>;
  using SOnGPT = LieGP<SOn, T>;


  SOnxRnGP() : p_{nullptr}, v_{nullptr}, a_{nullptr}, R_{nullptr}, om_{nullptr}, om_dot_{nullptr}
  {}

  SOnxRnGP(ModelType rn_type, ModelType son_type, MatrixDoF Q) : p_{nullptr}, v_{nullptr}, a_{nullptr}, R_{nullptr}, om_{nullptr}, om_dot_{nullptr},
    rn_gp_{new RnGPT(rn_type, Q.template topLeftCorner<RnDoF, RnDoF>())}, son_gp_{new SOnGPT(son_type, Q.template bottomRightCorner<SOnDoF, SOnDoF>())},
    rn_type_{rn_type}, son_type_{son_type}
  {}

  void init_est_params(std::vector<double> est_t, std::shared_ptr<std::vector<Rn>> p, std::shared_ptr<std::vector<Rn>> v, 
    std::shared_ptr<std::vector<SOn>> R, std::shared_ptr<std::vector<T>> om, std::shared_ptr<std::vector<Rn>> a = nullptr,
    std::shared_ptr<std::vector<T>> om_dot = nullptr)
  {
    p_ = p;
    v_ = v;
    R_ = R;
    om_ = om;
    if(rn_type_ == ModelType::JERK) a_ = a;
    if(son_type_ == ModelType::JERK) om_dot_ = om_dot;

    rn_gp_->init_est_params(est_t, p, v, a);
    son_gp_->init_est_params(est_t, R, om, om_dot);
  }

  int get_i(double t)
  {
    return rn_gp_->get_i(t);
  }

  bool is_time_valid(double t)
  {
    return rn_gp_->is_time_valid(t);
  }

  // jacobian vectors will be in the order (p_i, p_i1, v_i, v_i1, a_i, a_i1), as applicable
  int eval(double t, VectorRnDoF* p_ret = nullptr, VectorRnDoF* v_ret = nullptr, VectorRnDoF* a_ret = nullptr,
    SOnNonmapT* R_ret = nullptr, VectorSOnDoF* om_ret = nullptr, VectorSOnDoF* om_dot_ret = nullptr, 
    std::vector<MatrixRnDoF>* Jp_ret = nullptr, std::vector<MatrixRnDoF>* Jv_ret = nullptr, std::vector<MatrixRnDoF>* Ja_ret = nullptr,
    std::vector<MatrixSOnDoF>* JR_ret = nullptr, std::vector<MatrixSOnDoF>* Jom_ret = nullptr, std::vector<MatrixRnDoF>* Jomdot_ret = nullptr)
  {
    int i = rn_gp_->eval(t, p_ret, v_ret, a_ret, Jp_ret, Jv_ret, Ja_ret);
    son_gp_->eval(t, R_ret, om_ret, om_dot_ret, JR_ret, Jom_ret, Jomdot_ret);

    return i;
  }

  // void pre_eval(std::vector<double> t_vec, bool p, bool v, bool a, bool R, bool om, bool omdot, 
  //   bool jp, bool jv, bool ja, bool jR, bool jom, bool jomdot)
  // {
  //   rn_gp_->pre_eval(t_vec, p, v, a, jp, jv, ja);
  //   son_gp_->pre_eval(t_vec, R, om, omdot, jR, jom, jomdot);
  // }

  void pre_eval(std::vector<double> t_vec, bool d0, bool d1, bool d2, bool j0, bool j1, bool j2)
  {
    rn_gp_->pre_eval(t_vec, d0, d1, d2, j0, j1, j2);
    son_gp_->pre_eval(t_vec, d0, d1, d2, j0, j1, j2);
  }

  void get_pre_eval(int pre_eval_id, VectorRnDoF* p = nullptr, VectorRnDoF* v = nullptr, VectorRnDoF* a = nullptr,
    SOnNonmapT* R = nullptr, VectorSOnDoF* om = nullptr, VectorSOnDoF* om_dot = nullptr,
    std::vector<MatrixRnDoF>* Jp = nullptr, std::vector<MatrixRnDoF>* Jv = nullptr, std::vector<MatrixRnDoF>* Ja = nullptr,
    std::vector<MatrixSOnDoF>* JR = nullptr, std::vector<MatrixSOnDoF>* Jom = nullptr, std::vector<MatrixSOnDoF>* Jomdot = nullptr)
  {
    rn_gp_->get_pre_eval(pre_eval_id, p, v, a, Jp, Jv, Ja);
    son_gp_->get_pre_eval(pre_eval_id, R, om, om_dot, JR, Jom, Jomdot);
  }

  template <int Sp, int SR>
  std::pair<Eigen::Matrix<double, Sp, 1>, Eigen::Matrix<double, SR, 1>> get_dynamics_residual(int i, std::vector<Eigen::Matrix<double, Sp, RnDoF>>* jacs_p = nullptr,
    std::vector<Eigen::Matrix<double, SR, SOnDoF>>* jacs_R = nullptr)
  {
    return std::make_pair(rn_gp_->get_dynamics_residual<Sp>(i, jacs_p), son_gp_->get_dynamics_residual<SR>(i, jacs_R));
  }

  std::pair<Eigen::MatrixXd, Eigen::MatrixXd> get_Q_inv_i_i1(int i)
  {
    return std::make_pair(rn_gp_->get_Q_inv_i_i1(i), son_gp_->get_Q_inv_i_i1(i));
  }

  std::shared_ptr<std::vector<Rn>> get_p()
  {
    return rn_gp_->get_p();
  }

  std::shared_ptr<std::vector<Rn>> get_v()
  {
    return rn_gp_->get_v();
  }

  std::shared_ptr<std::vector<Rn>> get_a()
  {
    return rn_gp_->get_a();
  }

  std::shared_ptr<std::vector<SOn>> get_R()
  {
    return son_gp_->get_T();
  }

  std::shared_ptr<std::vector<T>> get_om()
  {
    return son_gp_->get_v();
  }

  std::shared_ptr<std::vector<T>> get_omdot()
  {
    return son_gp_->get_a();
  }

  ModelType get_rn_type()
  {
    return rn_type_;
  }

  ModelType get_son_type()
  {
    return son_type_;
  }

  std::shared_ptr<RnGPT> get_rn_gp()
  {
    return rn_gp_;
  }

  std::shared_ptr<SOnGPT> get_son_gp()
  {
    return son_gp_;
  }

private:
  std::shared_ptr<RnGPT> rn_gp_;
  std::shared_ptr<SOnGPT> son_gp_;

  ModelType rn_type_;
  ModelType son_type_;

  std::shared_ptr<std::vector<Rn>> p_;
  std::shared_ptr<std::vector<Rn>> v_;
  std::shared_ptr<std::vector<Rn>> a_;
  std::shared_ptr<std::vector<SOn>> R_;
  std::shared_ptr<std::vector<T>> om_;
  std::shared_ptr<std::vector<T>> om_dot_;
  
};

} // namespace gp