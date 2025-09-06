#pragma once

#include <vector>
#include <memory>
#include <iostream>
#include <Eigen/Dense>

#include "spline/lie_spline.hpp"
#include "spline/rn_spline.hpp"
#include "spline/utils.hpp"

namespace spline
{

// spline evaluations should run much faster than 
// SENSplines, at the expense of accuracy
// G must be SOnxRn or Map<SOnxRn> type

// for derivatives to make sense, the position spline 
// is assumed to represent p_(b/i)^i, and rotation
// represents R_i^b
template <typename G>
class SOnxRnSpline
{
private:
  using SOnT = typename G::SOnT;
  using SOnTangentT = typename SOnT::TangentT;
  using SOnJacT = typename SOnT::JacT;
  using SOnNonmapT = typename SOnT::NonmapT;
  static const int SOnDoF = SOnT::DoF;
  static const int N = SOnT::ACT;
  using RnT = typename G::RnT;
  using VectorNd = Eigen::Matrix<double, N, 1>;
  using MatrixNd = Eigen::Matrix<double, N, N>;

public:
  using TangentT = typename G::TangentT;
  using JacT = typename G::JacT;
  using NonmapT = typename G::NonmapT;
  using GroupT = G;

  SOnxRnSpline()
  {}

  SOnxRnSpline(int k) :
    son_spline_{new LieSpline<SOnT>(k)}, rn_spline_{new RnSpline<RnT, N>(k)}
  {}

  void init_ctrl_pts(std::shared_ptr<std::vector<G>> ctrl_pts, double start_time, double knot_spacing)
  {
    // subsplines hold the data. Create new data vectors and pass to subsplines
    std::shared_ptr<std::vector<SOnT>> son_ctrl_pts(new std::vector<SOnT>);
    std::shared_ptr<std::vector<RnT>> rn_ctrl_pts(new std::vector<RnT>);
    for(G &g : *ctrl_pts)
    {
      son_ctrl_pts->push_back(g.rotation());
      rn_ctrl_pts->push_back(g.translation());
    }
    son_spline_->init_ctrl_pts(son_ctrl_pts, start_time, knot_spacing);
    rn_spline_->init_ctrl_pts(rn_ctrl_pts, start_time, knot_spacing);

    // copy for easier access to SOnxRn control points
    ctrl_pts_ = ctrl_pts;
  }

  int get_i(double t)
  {
    return rn_spline_->get_i(t);
  }

  // this evaluates both the son spline and rn spline
  int eval(double t, NonmapT* d0_ret = nullptr, TangentT* d1_ret = nullptr, TangentT* d2_ret = nullptr,
      std::vector<JacT>* J0_ret = nullptr, std::vector<JacT>* J1_ret = nullptr, 
      std::vector<JacT>* J2_ret = nullptr)
  {
    SOnNonmapT r_d0;
    SOnTangentT r_d1, r_d2;
    std::vector<SOnJacT> r_j0, r_j1, r_j2;
    VectorNd t_d0, t_d1, t_d2;
    std::vector<MatrixNd> t_j0, t_j1, t_j2;

    int i_cp = son_spline_->eval(t, (d0_ret != nullptr) ? &r_d0 : nullptr, (d1_ret != nullptr) ? &r_d1 : nullptr,
      (d2_ret != nullptr) ? &r_d2 : nullptr, (J0_ret != nullptr) ? &r_j0 : nullptr,
      (J1_ret != nullptr) ? &r_j1 : nullptr, (J2_ret != nullptr) ? &r_j2 : nullptr);
    rn_spline_->eval(t, (d0_ret != nullptr) ? &t_d0 : nullptr, (d1_ret != nullptr) ? &t_d1 : nullptr,
      (d2_ret != nullptr) ? &t_d2 : nullptr, (J0_ret != nullptr) ? &t_j0 : nullptr,
      (J1_ret != nullptr) ? &t_j1 : nullptr, (J2_ret != nullptr) ? &t_j2 : nullptr);

    if(d0_ret != nullptr)
      *d0_ret = NonmapT(r_d0, t_d0);
    if(d1_ret != nullptr)
      *d1_ret = (TangentT() << t_d1, r_d1).finished();
    if(d2_ret != nullptr)
      *d2_ret = (TangentT() << t_d2, r_d2).finished();
    
    if(J0_ret != nullptr)
    {
      for(int i = 0; i < r_j0.size(); ++i)
      {
        JacT j0(JacT::Zero());
        j0.template topLeftCorner(N,N) = t_j0[i];
        j0.template bottomRightCorner(SOnDoF,SOnDoF) = r_j0[i];
        J0_ret->push_back(j0);
      }
    }

    if(J1_ret != nullptr)
    {
      for(int i = 0; i < r_j1.size(); ++i)
      {
        JacT j1(JacT::Zero());
        j1.template topLeftCorner(N,N) = t_j1[i];
        j1.template bottomRightCorner(SOnDoF,SOnDoF) = r_j1[i];
        J1_ret->push_back(j1);
      }
    }

    if(J2_ret != nullptr)
    {
      for(int i = 0; i < r_j2.size(); ++i)
      {
        JacT j2(JacT::Zero());
        j2.template topLeftCorner(N,N) = t_j2[i];
        j2.template bottomRightCorner(SOnDoF,SOnDoF) = r_j2[i];
        J2_ret->push_back(j2);
      }
    }

    return i_cp;
  }

  // this only evaluates the son spline
  int eval_son(double t, SOnNonmapT* d0_ret = nullptr, SOnTangentT* d1_ret = nullptr, SOnTangentT* d2_ret = nullptr,
      std::vector<SOnJacT>* J0_ret = nullptr, std::vector<SOnJacT>* J1_ret = nullptr, 
      std::vector<SOnJacT>* J2_ret = nullptr)
  {
    return son_spline_->eval(t, d0_ret, d1_ret, d2_ret, J0_ret, J1_ret, J2_ret);
  }

  // this only evaluates the rn spline
  int eval_rn(double t, VectorNd* d0_ret = nullptr, VectorNd* d1_ret = nullptr, VectorNd* d2_ret = nullptr,
      std::vector<MatrixNd>* J0_ret = nullptr, std::vector<MatrixNd>* J1_ret = nullptr, 
      std::vector<MatrixNd>* J2_ret = nullptr)
  {
    return rn_spline_->eval(t, d0_ret, d1_ret, d2_ret, J0_ret, J1_ret, J2_ret);
  }

  // preevaluate spline and jacobians to be accessed when get_pre_eval is called.
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
      eval(t, d0 ? &T_t : nullptr, d1 ? &d1_t : nullptr, d2 ? &d2_t : nullptr,
        j0 ? &j0_t : nullptr, j1 ? &j1_t : nullptr, j2 ? &j2_t : nullptr);
      
      if(d0) T_pre_eval_.push_back(T_t);
      if(d1) d1_pre_eval_.push_back(d1_t);
      if(d2) d2_pre_eval_.push_back(d2_t);
      if(j0) j0_pre_eval_.push_back(j0_t);
      if(j1) j1_pre_eval_.push_back(j1_t);
      if(j2) j2_pre_eval_.push_back(j2_t);
    }
  }

  // pre_eval_id is the index of the time in t_vec given to pre_eval()
  void get_pre_eval(int pre_eval_id, NonmapT* T = nullptr, TangentT* d1 = nullptr, TangentT* d2 = nullptr,
    std::vector<JacT>* J0 = nullptr, std::vector<JacT>* J1 = nullptr, std::vector<JacT>* J2 = nullptr)
  {
    if(T != nullptr)
    {
      assert(pre_eval_id < T_pre_eval_.size());
      *T = T_pre_eval_[pre_eval_id];
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

  bool modify_ctrl_pt(int l, G ctrl_pt)
  {
    son_spline_->modify_ctrl_pt(l, ctrl_pt.rotation());
    return rn_spline_->modify_ctrl_pt(l, ctrl_pt.translation());
  }

  int get_order()
  {
    return son_spline_->get_order();
  }

  double get_dt()
  {
    return son_spline_->get_dt();
  }

  utils::TimeRange get_time_range()
  {
    return son_spline_->get_time_range();
  }

  // this will only make sense for map types, or if the control points
  // have not been modified since instantiation
  std::shared_ptr<std::vector<G>> get_ctrl_pts()
  {
    return ctrl_pts_;
  }

  std::shared_ptr<LieSpline<SOnT>> get_son_spline()
  {
    return son_spline_;
  }

  std::shared_ptr<RnSpline<RnT, N>> get_rn_spline()
  {
    return rn_spline_;
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

  std::shared_ptr<LieSpline<SOnT>> son_spline_;
  std::shared_ptr<RnSpline<RnT, N>> rn_spline_;

  std::shared_ptr<std::vector<G>> ctrl_pts_;

  // pre-evaluated values
  std::vector<NonmapT> T_pre_eval_;
  std::vector<TangentT> d1_pre_eval_;
  std::vector<TangentT> d2_pre_eval_;
  std::vector<std::vector<JacT>> j0_pre_eval_;
  std::vector<std::vector<JacT>> j1_pre_eval_;
  std::vector<std::vector<JacT>> j2_pre_eval_;

};

} // namespace spline