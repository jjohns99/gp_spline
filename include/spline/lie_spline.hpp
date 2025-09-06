#pragma once

#include <vector>
#include <map>
#include <cmath>
#include <memory>
#include <iostream>
#include <Eigen/Dense>
#include <iostream>

#include "spline/utils.hpp"

#include "lie_groups/so2.hpp"
#include "lie_groups/so3.hpp"
#include "lie_groups/se2.hpp"
#include "lie_groups/se3.hpp"

namespace spline 
{

template <typename G>
class LieSpline
{
public:
  using TangentT = typename G::TangentT;
  using JacT = typename G::JacT;
  using NonmapT = typename G::NonmapT;
  using GroupT = G;

  LieSpline()
  {}

  LieSpline(int k) : k_{k}
  {}

  ~LieSpline()
  {}
  
  void init_ctrl_pts(std::shared_ptr<std::vector<G>> ctrl_pts, double start_time, double knot_spacing)
  {
    ctrl_pts_ = ctrl_pts;
    tr_.start = start_time;
    dt_ = knot_spacing;

    // compute C matrices
    utils::compute_C(C_, k_, dt_, 2);

    //construct the knot vector
    knot_pts_.clear();
    knot_pts_.push_back(tr_.start - (k_-1)*dt_);
    for(int i = 0; i < ctrl_pts_->size() + k_ - 2; ++i)
      knot_pts_.push_back(knot_pts_.back() + dt_);

    tr_.end = knot_pts_[knot_pts_.size()-k_];
  }

  int get_i(double t)
  {
    return static_cast<int>(std::floor((t - tr_.start)/dt_));
  }

  // evaluate the spline at time t. Also calculates d1 and/or d2 if supplied,
  // as well as the k nonzero jacobians of T, d1, and/or d2 at t. Returns the 
  // index of the first control point used
  int eval(double t, NonmapT* d0_ret = nullptr, TangentT* d1_ret = nullptr, TangentT* d2_ret = nullptr,
      std::vector<JacT>* J0_ret = nullptr, std::vector<JacT>* J1_ret = nullptr, 
      std::vector<JacT>* J2_ret = nullptr)
  {
    // assert that the time to evaluate is within a valid range
    assert(tr_.valid(t));

    // this i corresponds to how many knot points above t0 we are
    // need to add k-1 when indexing knot points and 2 when indexing control points
    int i = get_i(t);
    int i_kp = i + k_ - 1;
    int i_cp = i + 2;

    double u = (t - knot_pts_[i_kp])/dt_;

    Eigen::MatrixXd u_vec(k_, 1);
    for(int j = 0; j < k_; ++j)
      u_vec(j) = std::pow(u, j);

    Eigen::MatrixXd b0 = C_[0] * u_vec;
    
    std::vector<NonmapT> T;
    std::vector<NonmapT> A;
    std::vector<TangentT> Om;

    // these only get used if d1_ret and/or d2_ret isn't nullptr
    std::vector<TangentT> a1, d1;
    std::vector<TangentT> a2, d2;
    Eigen::MatrixXd b1, b2;

    T.push_back(ctrl_pts_->at(i_cp-1));
    for(int j = 1; j < k_; ++j)
    {
      Om.push_back((ctrl_pts_->at(i_cp+j-2).inverse() * ctrl_pts_->at(i_cp+j-1)).Log());
      A.push_back(G::Exp(b0(j)*Om.back()));

      // Om and A will have length k-1, whereas T_ will have length k
      T.push_back(T[j-1] * A.back());
    }
    if(d0_ret != nullptr) *d0_ret = T.back();

    if((d1_ret != nullptr) || (d2_ret != nullptr) 
      || (J1_ret != nullptr) || (J2_ret != nullptr))
    {
      // calculate first derivative
      b1 = C_[1] * u_vec;

      d1.push_back(TangentT::Zero());
      for(int j = 1; j < k_; ++j)
      {
        a1.push_back(b1(j) * Om[j-1]);

        // a1 will have length k-1, whereas d1 will have length k
        d1.push_back(d1[j-1] + T[j-1].Ad() * a1.back());
      }
      if(d1_ret != nullptr) *d1_ret = d1.back();

      if((d2_ret != nullptr) || (J2_ret != nullptr))
      {
        // calculate second derivative
        b2 = C_[2] * u_vec;

        d2.push_back(TangentT::Zero());
        for(int j = 1; j < k_; ++j)
        {
          a2.push_back(b2(j) * Om[j-1]);

          // a2 will have length k-1, whereas d2 will have length k
          d2.push_back(d2[j-1] - G::ad(d1[j]) * d1[j-1] + T[j-1].Ad() * a2.back());
        }
        if(d2_ret != nullptr) *d2_ret = d2.back();
      }
    }

    // return if jacobians are not needed
    if((J0_ret == nullptr) && (J1_ret == nullptr) && (J2_ret == nullptr)) return i_cp - 1;

    // compute jacobians
    // loop through k control points
    for(int l = i_cp - 1; l < i_cp + k_ - 1; ++l)
    {
      std::vector<JacT> j0, j1, j2, om_l;
      
      if(l == i_cp - 1) j0.push_back(JacT::Identity());
      else j0.push_back(JacT::Zero());

      for(int j = 1; j < k_; ++j)
      {
        // calculate dOm/dTl
        JacT om_jac;
        if(l == i_cp + j - 2)
        {
          om_jac = -G::Jl_inv(Om[j-1]) * ctrl_pts_->at(l).inverse().Ad();
        }
        else if(l == i_cp + j - 1)
        {
          om_jac = G::Jl_inv(Om[j-1]) * ctrl_pts_->at(l-1).inverse().Ad();
        }
        else
        {
          om_l.push_back(JacT::Zero());

          // everything will multiply by zero. Avoid unnecessary computation
          j0.push_back(j0[j-1]);
          continue;
        }
        om_l.push_back(om_jac);

        JacT a_jac = b0(j) * G::Jl(b0(j) * Om[j-1]) * om_l[j-1];
        j0.push_back(j0[j-1] + T[j-1].Ad() * a_jac);
      }
      if(J0_ret != nullptr) J0_ret->push_back(j0.back());

      if((J1_ret != nullptr) || (J2_ret != nullptr))
      {
        j1.push_back(JacT::Zero());
        for(int j = 1; j < k_; ++j)
        {
          JacT Tj1 = j1[j-1] - G::ad(d1[j] - d1[j-1]) * j0[j-1];
          if((l == i_cp + j - 2) || (l == i_cp + j - 1))
          {
            Tj1 += T[j-1].Ad() * (b1(j) * om_l[j-1]);
          }
          j1.push_back(Tj1);
        }
        if(J1_ret != nullptr) J1_ret->push_back(j1.back());

        if(J2_ret != nullptr)
        {
          j2.push_back(JacT::Zero());
          for(int j = 1; j < k_; ++j)
          {
            JacT Tj2 = j2[j-1] - G::ad(d1[j]) * j1[j-1]
              + G::ad(d1[j-1]) * j1[j] - G::ad(T[j-1].Ad() * a2[j-1]) * j0[j-1];
            if((l == i_cp + j - 2) || (l == i_cp + j - 1))
            {
              Tj2 += T[j-1].Ad() * (b2(j) * om_l[j-1]);
            }
            j2.push_back(Tj2);
          }
          J2_ret->push_back(j2.back());
        }
      }
    }

    return i_cp - 1;
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
  
  // this wont work with lie_group::Map types
  void update_ctrl_pts(const std::map<int, TangentT> updates)
  {
    for(const auto &update : updates)
    {
      // should this be a left or right update?
      ctrl_pts_->at(update.first) = G::Exp(update.second) * ctrl_pts_->at(update.first);
    }
  }

  void add_ctrl_pt(G ctrl_pt)
  {
    ctrl_pts_->push_back(ctrl_pt);
    knot_pts_.push_back(knot_pts_.back() + dt_);
  }

  bool modify_ctrl_pt(int l, G ctrl_pt)
  {
    if(l >= ctrl_pts_->size())
      return false;

    ctrl_pts_->at(l) = ctrl_pt;
    return true;
  }
  
  int get_order()
  {
    return k_;
  }

  double get_dt()
  {
    return dt_;
  }

  utils::TimeRange get_time_range()
  {
    return tr_;
  }

  std::shared_ptr<std::vector<G>> get_ctrl_pts()
  {
    return ctrl_pts_;
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

  std::shared_ptr<std::vector<G>> ctrl_pts_;
  std::vector<double> knot_pts_;
  double dt_;
  utils::TimeRange tr_;
  int k_;
  // holds C matrix for 0th, 1st, 2nd spline derivatives
  std::vector<Eigen::MatrixXd> C_;

  // pre-evaluated values
  std::vector<NonmapT> T_pre_eval_;
  std::vector<TangentT> d1_pre_eval_;
  std::vector<TangentT> d2_pre_eval_;
  std::vector<std::vector<JacT>> j0_pre_eval_;
  std::vector<std::vector<JacT>> j1_pre_eval_;
  std::vector<std::vector<JacT>> j2_pre_eval_;
};
 
typedef LieSpline<lie_groups::SO2d> SO2Spline;
typedef LieSpline<lie_groups::SO3d> SO3Spline;
typedef LieSpline<lie_groups::SE2d> SE2Spline;
typedef LieSpline<lie_groups::SE3d> SE3Spline;

} //namespace spline