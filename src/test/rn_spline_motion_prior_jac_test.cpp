#include <vector>
#include <chrono>
#include <memory>
#include <iostream>
#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "utils/multivariate_gaussian.hpp"
#include "spline/rn_spline.hpp"
#include "estimator/rn_spline_motion_prior.hpp"

namespace linear_example
{

template <int DoF>
class RnSplineMotionPriorTest : public testing::Test
{
protected:
  using VectorDoF = Eigen::Matrix<double, DoF, 1>;
  using MatrixDoF = Eigen::Matrix<double, DoF, DoF>;
  using SplineT = spline::RnSpline<VectorDoF, DoF>;

public:
  RnSplineMotionPriorTest() : h_{1e-5}, tolerance_{1e-4}
  {}

  void create_spline(int k, int n, double dt)
  {
    spline_.reset(new SplineT(k));

    std::shared_ptr<std::vector<VectorDoF>> ctrl_pts(new std::vector<VectorDoF>);
    for(int i = 0; i < n; ++i) ctrl_pts->push_back(VectorDoF::Random());
    spline_->init_ctrl_pts(ctrl_pts, 0.0, dt);
    ctrl_pts_ = *ctrl_pts;
  }

  void create_mp(estimator::ModelType type, MatrixDoF Q)
  {
    mp_ = estimator::RnSplineMotionPrior<SplineT, DoF>(type, Q);
  }

  template <int S>
  std::vector<Eigen::Matrix<double, S, DoF>> get_fin_diff_jacs(double t1, double t2)
  {
    int i1 = spline_->get_i(t1);
    int i2 = spline_->get_i(t2);
    int num_cp = std::min(i2 - i1, spline_->get_order()) + spline_->get_order();
    
    std::vector<Eigen::Matrix<double, S, DoF>> jacs;

    Eigen::Matrix<double, S, 1> r_unpert = mp_.template get_motion_prior<S>(t1, t2, spline_);
    for(int j = 0; j < num_cp; ++j)
    {
      // determine control point index
      int cp_ind = (j < spline_->get_order()) ? (i1 + j + 1) : (i2 + j - std::min(i2 - i1, spline_->get_order()) + 1);

      Eigen::Matrix<double, S, DoF> jac;
      jac.setZero();
      for(int d = 0; d < DoF; ++d)
      {
        VectorDoF tau = VectorDoF::Zero();
        tau(d) = h_;
        assert(spline_->modify_ctrl_pt(cp_ind, ctrl_pts_[cp_ind] + tau));

        Eigen::Matrix<double, S, 1> r_pert = mp_.template get_motion_prior<S>(t1, t2, spline_);
        jac.col(d) = (r_pert - r_unpert)/h_;

        assert(spline_->modify_ctrl_pt(cp_ind, ctrl_pts_[cp_ind]));
      }
      jacs.push_back(jac);
    }
    return jacs;
  }

protected:
  double h_;
  double tolerance_;

  std::vector<VectorDoF> ctrl_pts_;
  std::shared_ptr<SplineT> spline_;
  estimator::RnSplineMotionPrior<SplineT, DoF> mp_;
};

} // linear_example

typedef linear_example::RnSplineMotionPriorTest<2> R2SplineMPTest;
typedef linear_example::RnSplineMotionPriorTest<3> R3SplineMPTest;

TEST_F(R2SplineMPTest, R2SplineAccMPTest)
{
  int k = 4;
  int n = 20;
  double dt = 0.1;
  create_spline(k, n, dt);

  MatrixDoF Q = 0.2 * MatrixDoF::Identity();
  create_mp(estimator::ModelType::ACC, Q);

  double dt_samp = 0.1;
  double t1 = 0.0;
  for(double t2 = t1 + dt_samp; spline_->get_time_range().valid(t2 + dt_samp); t2 += dt_samp)
  {
    std::vector<Eigen::Matrix<double, 4, 2>> eval_jacs;
    mp_.get_motion_prior<4>(t1, t2, spline_, &eval_jacs);

    std::vector<Eigen::Matrix<double, 4, 2>> fin_diff_jacs = get_fin_diff_jacs<4>(t1, t2);

    EXPECT_TRUE(fin_diff_jacs.size() == eval_jacs.size());

    for(int d = 0; d < eval_jacs.size(); ++d)
    {
      EXPECT_TRUE((fin_diff_jacs[d] - eval_jacs[d]).isZero(tolerance_));
      if(!(fin_diff_jacs[d] - eval_jacs[d]).isZero(tolerance_))
        std::cout << "d = " << d << ", t1 = " << t1 << ", t2 = " << t2 << "\nEval: \n" << eval_jacs[d] << "\n Fin diff: \n" << fin_diff_jacs[d] << "\n\n";
    }

    t1 = t2;
  }
}

TEST_F(R2SplineMPTest, R2SplineJerkMPTest)
{
  int k = 4;
  int n = 20;
  double dt = 0.1;
  create_spline(k, n, dt);

  MatrixDoF Q = 0.2 * MatrixDoF::Identity();
  create_mp(estimator::ModelType::JERK, Q);

  double dt_samp = 0.1;
  double t1 = 0.0;
  for(double t2 = t1 + dt_samp; spline_->get_time_range().valid(t2 + dt_samp); t2 += dt_samp)
  {
    std::vector<Eigen::Matrix<double, 6, 2>> eval_jacs;
    mp_.get_motion_prior<6>(t1, t2, spline_, &eval_jacs);

    std::vector<Eigen::Matrix<double, 6, 2>> fin_diff_jacs = get_fin_diff_jacs<6>(t1, t2);

    EXPECT_TRUE(fin_diff_jacs.size() == eval_jacs.size());

    for(int d = 0; d < eval_jacs.size(); ++d)
    {
      EXPECT_TRUE((fin_diff_jacs[d] - eval_jacs[d]).isZero(tolerance_));
      if(!(fin_diff_jacs[d] - eval_jacs[d]).isZero(tolerance_))
        std::cout << "d = " << d << ", t1 = " << t1 << ", t2 = " << t2 << "\nEval: \n" << eval_jacs[d] << "\n Fin diff: \n" << fin_diff_jacs[d] << "\n\n";
    }

    t1 = t2;
  }
}

TEST_F(R3SplineMPTest, R3SplineAccMPTest)
{
  int k = 4;
  int n = 20;
  double dt = 0.1;
  create_spline(k, n, dt);

  MatrixDoF Q = 0.2 * MatrixDoF::Identity();
  create_mp(estimator::ModelType::ACC, Q);

  double dt_samp = 0.1;
  double t1 = 0.0;
  for(double t2 = t1 + dt_samp; spline_->get_time_range().valid(t2 + dt_samp); t2 += dt_samp)
  {
    std::vector<Eigen::Matrix<double, 6, 3>> eval_jacs;
    mp_.get_motion_prior<6>(t1, t2, spline_, &eval_jacs);

    std::vector<Eigen::Matrix<double, 6, 3>> fin_diff_jacs = get_fin_diff_jacs<6>(t1, t2);

    EXPECT_TRUE(fin_diff_jacs.size() == eval_jacs.size());

    for(int d = 0; d < eval_jacs.size(); ++d)
    {
      EXPECT_TRUE((fin_diff_jacs[d] - eval_jacs[d]).isZero(tolerance_));
      if(!(fin_diff_jacs[d] - eval_jacs[d]).isZero(tolerance_))
        std::cout << "d = " << d << ", t1 = " << t1 << ", t2 = " << t2 << "\nEval: \n" << eval_jacs[d] << "\n Fin diff: \n" << fin_diff_jacs[d] << "\n\n";
    }

    t1 = t2;
  }
}

TEST_F(R3SplineMPTest, R3SplineJerkMPTest)
{
  int k = 4;
  int n = 20;
  double dt = 0.1;
  create_spline(k, n, dt);

  MatrixDoF Q = 0.2 * MatrixDoF::Identity();
  create_mp(estimator::ModelType::JERK, Q);

  double dt_samp = 0.1;
  double t1 = 0.0;
  for(double t2 = t1 + dt_samp; spline_->get_time_range().valid(t2 + dt_samp); t2 += dt_samp)
  {
    std::vector<Eigen::Matrix<double, 9, 3>> eval_jacs;
    mp_.get_motion_prior<9>(t1, t2, spline_, &eval_jacs);

    std::vector<Eigen::Matrix<double, 9, 3>> fin_diff_jacs = get_fin_diff_jacs<9>(t1, t2);

    EXPECT_TRUE(fin_diff_jacs.size() == eval_jacs.size());

    for(int d = 0; d < eval_jacs.size(); ++d)
    {
      EXPECT_TRUE((fin_diff_jacs[d] - eval_jacs[d]).isZero(tolerance_));
      if(!(fin_diff_jacs[d] - eval_jacs[d]).isZero(tolerance_))
        std::cout << "d = " << d << ", t1 = " << t1 << ", t2 = " << t2 << "\nEval: \n" << eval_jacs[d] << "\n Fin diff: \n" << fin_diff_jacs[d] << "\n\n";
    }

    t1 = t2;
  }
}