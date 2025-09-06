#pragma once

#include <cmath>
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <iostream>

#include "spline/utils.hpp"

namespace spline
{

// G must be Matrix<s, N, 1> or Map<Matrix<s, N, 1>>
template <typename G, int N>
class RnSpline
{
public:
  using VectorNd = Eigen::Matrix<double, N, 1>;
  using MatrixNd = Eigen::Matrix<double, N, N>;

  RnSpline()
  {}

  RnSpline(int k) : k_{k}
  {}

  void init_ctrl_pts(std::shared_ptr<std::vector<G>> ctrl_pts, double start_time, double knot_spacing)
  {
    ctrl_pts_ = ctrl_pts;
    tr_.start = start_time;
    dt_ = knot_spacing;

    //compute C matrices
    utils::compute_C(C_, k_, dt_, 3);

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
  int eval(double t, VectorNd* d0_ret = nullptr, VectorNd* d1_ret = nullptr, VectorNd* d2_ret = nullptr,
      std::vector<MatrixNd>* J0_ret = nullptr, std::vector<MatrixNd>* J1_ret = nullptr, 
      std::vector<MatrixNd>* J2_ret = nullptr)
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

    if((d0_ret != nullptr) || (J0_ret != nullptr))
    {
      Eigen::MatrixXd b0 = C_[0] * u_vec;

      VectorNd d0 = VectorNd::Zero();
      for(int j = 0; j < k_ - 1; ++j) d0 += (b0(j) - b0(j+1)) * ctrl_pts_->at(i_cp-1+j);
      d0 += b0(k_-1) * ctrl_pts_->at(i_cp-2+k_);
      if(d0_ret != nullptr) *d0_ret = d0;

      if(J0_ret != nullptr)
      {
        for(int j = 0; j < k_ - 1; ++j) J0_ret->push_back((b0(j) - b0(j+1)) * MatrixNd::Identity());
        J0_ret->push_back(b0(k_-1) * MatrixNd::Identity());
      }
    }

    if((d1_ret != nullptr) || (J1_ret != nullptr))
    {
      Eigen::MatrixXd b1 = C_[1] * u_vec;

      VectorNd d1 = VectorNd::Zero();
      for(int j = 0; j < k_ - 1; ++j) d1 += (b1(j) - b1(j+1)) * ctrl_pts_->at(i_cp-1+j);
      d1 += b1(k_-1) * ctrl_pts_->at(i_cp-2+k_);
      if(d1_ret != nullptr) *d1_ret = d1;

      if(J1_ret != nullptr)
      {
        for(int j = 0; j < k_ - 1; ++j) J1_ret->push_back((b1(j) - b1(j+1)) * MatrixNd::Identity());
        J1_ret->push_back(b1(k_-1) * MatrixNd::Identity());
      }
    }

    if((d2_ret != nullptr) || (J2_ret != nullptr))
    {
      Eigen::MatrixXd b2 = C_[2] * u_vec;

      VectorNd d2 = VectorNd::Zero();
      for(int j = 0; j < k_ - 1; ++j) d2 += (b2(j) - b2(j+1)) * ctrl_pts_->at(i_cp-1+j);
      d2 += b2(k_-1) * ctrl_pts_->at(i_cp-2+k_);
      if(d2_ret != nullptr) *d2_ret = d2;

      if(J2_ret != nullptr)
      {
        for(int j = 0; j < k_ - 1; ++j) J2_ret->push_back((b2(j) - b2(j+1)) * MatrixNd::Identity());
        J2_ret->push_back(b2(k_-1) * MatrixNd::Identity());
      }
    }

    return i_cp - 1;
  }

  // option to evaluate 3rd derivative (without changing API)
  // this is only provided for rn splines, not lie splines
  int eval_d3(double t, VectorNd* d3_ret = nullptr, std::vector<MatrixNd>* J3_ret = nullptr)
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

    if((d3_ret != nullptr) || (J3_ret != nullptr))
    {
      Eigen::MatrixXd b3 = C_[3] * u_vec;

      VectorNd d3 = VectorNd::Zero();
      for(int j = 0; j < k_ - 1; ++j) d3 += (b3(j) - b3(j+1)) * ctrl_pts_->at(i_cp-1+j);
      d3 += b3(k_-1) * ctrl_pts_->at(i_cp-2+k_);
      if(d3_ret != nullptr) *d3_ret = d3;

      if(J3_ret != nullptr)
      {
        for(int j = 0; j < k_ - 1; ++j) J3_ret->push_back((b3(j) - b3(j+1)) * MatrixNd::Identity());
        J3_ret->push_back(b3(k_-1) * MatrixNd::Identity());
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
      VectorNd d0_t, d1_t, d2_t;
      std::vector<MatrixNd> j0_t, j1_t, j2_t;
      eval(t, d0 ? &d0_t : nullptr, d1 ? &d1_t : nullptr, d2 ? &d2_t : nullptr,
        j0 ? &j0_t : nullptr, j1 ? &j1_t : nullptr, j2 ? &j2_t : nullptr);
      
      if(d0) d0_pre_eval_.push_back(d0_t);
      if(d1) d1_pre_eval_.push_back(d1_t);
      if(d2) d2_pre_eval_.push_back(d2_t);
      if(j0) j0_pre_eval_.push_back(j0_t);
      if(j1) j1_pre_eval_.push_back(j1_t);
      if(j2) j2_pre_eval_.push_back(j2_t);
    }
  }

  // pre_eval_id is the index of the time in t_vec given to pre_eval()
  void get_pre_eval(int pre_eval_id, VectorNd* d0 = nullptr, VectorNd* d1 = nullptr, VectorNd* d2 = nullptr,
    std::vector<MatrixNd>* J0 = nullptr, std::vector<MatrixNd>* J1 = nullptr, std::vector<MatrixNd>* J2 = nullptr)
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
    d0_pre_eval_.clear();
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
  std::vector<Eigen::MatrixXd> C_;

  // pre-evaluated values
  std::vector<VectorNd> d0_pre_eval_;
  std::vector<VectorNd> d1_pre_eval_;
  std::vector<VectorNd> d2_pre_eval_;
  std::vector<std::vector<MatrixNd>> j0_pre_eval_;
  std::vector<std::vector<MatrixNd>> j1_pre_eval_;
  std::vector<std::vector<MatrixNd>> j2_pre_eval_;

};

} // namespace spline