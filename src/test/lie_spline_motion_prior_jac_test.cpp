#include <vector>
#include <chrono>
#include <memory>
#include <iostream>
#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "utils/multivariate_gaussian.hpp"
#include "spline/lie_spline.hpp"
#include "estimator/rn_spline_motion_prior.hpp"
#include "estimator/lie_spline_motion_prior.hpp"

namespace estimator
{

template <typename G>
class LieSplineMotionPriorTest : public testing::Test
{
protected:
  static const int DoF = G::DoF;
  using VectorDoF = Eigen::Matrix<double, DoF, 1>;
  using MatrixDoF = Eigen::Matrix<double, DoF, DoF>;
  using SplineT = spline::LieSpline<G>;

public:
  LieSplineMotionPriorTest() : h_{1e-5}, tolerance_{1e-4}
  {}

  void create_spline(int k, int n, double dt)
  {
    spline_.reset(new SplineT(k));

    std::shared_ptr<std::vector<G>> ctrl_pts(new std::vector<G>);
    for(int i = 0; i < n; ++i) ctrl_pts->push_back(G::random());
    spline_->init_ctrl_pts(ctrl_pts, 0.0, dt);
    ctrl_pts_ = *ctrl_pts;
  }

  void create_mp(estimator::ModelType type, MatrixDoF Q)
  {
    mp_ = estimator::LieSplineMotionPrior<SplineT>(type, Q);
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
        assert(spline_->modify_ctrl_pt(cp_ind, G::Exp(tau) * ctrl_pts_[cp_ind]));

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

  std::vector<G> ctrl_pts_;
  std::shared_ptr<SplineT> spline_;
  estimator::LieSplineMotionPrior<SplineT> mp_;
};

} // namespace estimator

typedef estimator::LieSplineMotionPriorTest<lie_groups::SO2d> SO2SplineMPTest;
typedef estimator::LieSplineMotionPriorTest<lie_groups::SO3d> SO3SplineMPTest;
typedef estimator::LieSplineMotionPriorTest<lie_groups::SE2d> SE2SplineMPTest;
typedef estimator::LieSplineMotionPriorTest<lie_groups::SE3d> SE3SplineMPTest;

TEST_F(SO2SplineMPTest, SO2SplineAccMPTest)
{
  int k = 4;
  int n = 200;
  double dt = 0.1;

  for(int z = 0; z < 10; ++z)
  {
    create_spline(k, n, dt);

    MatrixDoF Q = 0.2 * MatrixDoF::Identity();
    create_mp(estimator::ModelType::ACC, Q);

    double dt_samp = 0.1;
    double t1 = 0.0;
    for(double t2 = t1 + dt_samp; spline_->get_time_range().valid(t2 + dt_samp); t2 += dt_samp)
    {
      std::vector<Eigen::Matrix<double, 2*DoF, DoF>> eval_jacs;
      mp_.get_motion_prior<2*DoF>(t1, t2, spline_, &eval_jacs);

      std::vector<Eigen::Matrix<double, 2*DoF, DoF>> fin_diff_jacs = get_fin_diff_jacs<2*DoF>(t1, t2);

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
}

TEST_F(SO2SplineMPTest, SO2SplineJerkMPTest)
{
  int k = 4;
  int n = 200;
  double dt = 0.1;

  for(int z = 0; z < 10; ++z)
  {
    create_spline(k, n, dt);

    MatrixDoF Q = 0.2 * MatrixDoF::Identity();
    create_mp(estimator::ModelType::JERK, Q);

    double dt_samp = 0.1;
    double t1 = 0.0;
    for(double t2 = t1 + dt_samp; spline_->get_time_range().valid(t2 + dt_samp); t2 += dt_samp)
    {
      std::vector<Eigen::Matrix<double, 3*DoF, DoF>> eval_jacs;
      mp_.get_motion_prior<3*DoF>(t1, t2, spline_, &eval_jacs);

      std::vector<Eigen::Matrix<double, 3*DoF, DoF>> fin_diff_jacs = get_fin_diff_jacs<3*DoF>(t1, t2);

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
}

// TEST_F(SO3SplineMPTest, SO3SplineAccMPTest)
// {
//   int k = 4;
//   int n = 20;
//   double dt = 0.1;

//   for(int z = 0; z < 1; ++z)
//   {
//     create_spline(k, n, dt);

//     MatrixDoF Q = 0.2 * MatrixDoF::Identity();
//     create_mp(estimator::ModelType::ACC, Q);

//     double dt_samp = 0.1;
//     double t1 = 0.0;
//     for(double t2 = t1 + dt_samp; spline_->get_time_range().valid(t2 + dt_samp); t2 += dt_samp)
//     {
//       std::vector<Eigen::Matrix<double, 2*DoF, DoF>> eval_jacs;
//       mp_.get_motion_prior<2*DoF>(t1, t2, spline_, &eval_jacs);

//       std::vector<Eigen::Matrix<double, 2*DoF, DoF>> fin_diff_jacs = get_fin_diff_jacs<2*DoF>(t1, t2);

//       EXPECT_TRUE(fin_diff_jacs.size() == eval_jacs.size());

//       for(int d = 0; d < eval_jacs.size(); ++d)
//       {
//         EXPECT_TRUE((fin_diff_jacs[d] - eval_jacs[d]).isZero(tolerance_));
//         if(!(fin_diff_jacs[d] - eval_jacs[d]).isZero(tolerance_))
//           std::cout << "d = " << d << ", t1 = " << t1 << ", t2 = " << t2 << "\nEval: \n" << eval_jacs[d] << "\n Fin diff: \n" << fin_diff_jacs[d] << "\n\n";
//       }

//       t1 = t2;
//     }
//   }
// }

// TEST_F(SO3SplineMPTest, SO3SplineJerkMPTest)
// {
//   int k = 4;
//   int n = 20;
//   double dt = 0.1;
//   create_spline(k, n, dt);

//   MatrixDoF Q = 0.2 * MatrixDoF::Identity();
//   create_mp(estimator::ModelType::JERK, Q);

//   double dt_samp = 0.1;
//   double t1 = 0.0;
//   for(double t2 = t1 + dt_samp; spline_->get_time_range().valid(t2 + dt_samp); t2 += dt_samp)
//   {
//     std::vector<Eigen::Matrix<double, 3*DoF, DoF>> eval_jacs;
//     mp_.get_motion_prior<3*DoF>(t1, t2, spline_, &eval_jacs);

//     std::vector<Eigen::Matrix<double, 3*DoF, DoF>> fin_diff_jacs = get_fin_diff_jacs<3*DoF>(t1, t2);

//     EXPECT_TRUE(fin_diff_jacs.size() == eval_jacs.size());

//     for(int d = 0; d < eval_jacs.size(); ++d)
//     {
//       EXPECT_TRUE((fin_diff_jacs[d] - eval_jacs[d]).isZero(tolerance_));
//       if(!(fin_diff_jacs[d] - eval_jacs[d]).isZero(tolerance_))
//         std::cout << "d = " << d << ", t1 = " << t1 << ", t2 = " << t2 << "\nEval: \n" << eval_jacs[d] << "\n Fin diff: \n" << fin_diff_jacs[d] << "\n\n";
//     }

//     t1 = t2;
//   }
// }

// TEST_F(SE2SplineMPTest, SE2SplineAccMPTest)
// {
//   int k = 4;
//   int n = 20;
//   double dt = 0.1;

//   for(int z = 0; z < 1; ++z)
//   {
//     create_spline(k, n, dt);

//     MatrixDoF Q = 0.2 * MatrixDoF::Identity();
//     create_mp(estimator::ModelType::ACC, Q);

//     double dt_samp = 0.1;
//     double t1 = 0.0;
//     for(double t2 = t1 + dt_samp; spline_->get_time_range().valid(t2 + dt_samp); t2 += dt_samp)
//     {
//       std::vector<Eigen::Matrix<double, 2*DoF, DoF>> eval_jacs;
//       mp_.get_motion_prior<2*DoF>(t1, t2, spline_, &eval_jacs);

//       std::vector<Eigen::Matrix<double, 2*DoF, DoF>> fin_diff_jacs = get_fin_diff_jacs<2*DoF>(t1, t2);

//       EXPECT_TRUE(fin_diff_jacs.size() == eval_jacs.size());

//       for(int d = 0; d < eval_jacs.size(); ++d)
//       {
//         EXPECT_TRUE((fin_diff_jacs[d] - eval_jacs[d]).isZero(tolerance_));
//         if(!(fin_diff_jacs[d] - eval_jacs[d]).isZero(tolerance_))
//           std::cout << "d = " << d << ", t1 = " << t1 << ", t2 = " << t2 << "\nEval: \n" << eval_jacs[d] << "\n Fin diff: \n" << fin_diff_jacs[d] << "\n\n";
//       }

//       t1 = t2;
//     }
//   }
// }

// TEST_F(SE2SplineMPTest, SE2SplineJerkMPTest)
// {
//   int k = 4;
//   int n = 20;
//   double dt = 0.1;

//   for(int z = 0; z < 1; ++z)
//   {
//     create_spline(k, n, dt);

//     MatrixDoF Q = 0.2 * MatrixDoF::Identity();
//     create_mp(estimator::ModelType::JERK, Q);

//     double dt_samp = 0.1;
//     double t1 = 0.0;
//     for(double t2 = t1 + dt_samp; spline_->get_time_range().valid(t2 + dt_samp); t2 += dt_samp)
//     {
//       std::vector<Eigen::Matrix<double, 3*DoF, DoF>> eval_jacs;
//       mp_.get_motion_prior<3*DoF>(t1, t2, spline_, &eval_jacs);

//       std::vector<Eigen::Matrix<double, 3*DoF, DoF>> fin_diff_jacs = get_fin_diff_jacs<3*DoF>(t1, t2);

//       EXPECT_TRUE(fin_diff_jacs.size() == eval_jacs.size());

//       for(int d = 0; d < eval_jacs.size(); ++d)
//       {
//         EXPECT_TRUE((fin_diff_jacs[d] - eval_jacs[d]).isZero(tolerance_));
//         if(!(fin_diff_jacs[d] - eval_jacs[d]).isZero(tolerance_))
//           std::cout << "d = " << d << ", t1 = " << t1 << ", t2 = " << t2 << "\nEval: \n" << eval_jacs[d] << "\n Fin diff: \n" << fin_diff_jacs[d] << "\n\n";
//       }

//       t1 = t2;
//     }
//   }
// }

// TEST_F(SE3SplineMPTest, SE3SplineAccMPTest)
// {
//   int k = 4;
//   int n = 20;
//   double dt = 0.1;

//   for(int z = 0; z < 1; ++z)
//   {
//     create_spline(k, n, dt);

//     MatrixDoF Q = 0.2 * MatrixDoF::Identity();
//     create_mp(estimator::ModelType::ACC, Q);

//     double dt_samp = 0.1;
//     double t1 = 0.0;
//     for(double t2 = t1 + dt_samp; spline_->get_time_range().valid(t2 + dt_samp); t2 += dt_samp)
//     {
//       std::vector<Eigen::Matrix<double, 2*DoF, DoF>> eval_jacs;
//       mp_.get_motion_prior<2*DoF>(t1, t2, spline_, &eval_jacs);

//       std::vector<Eigen::Matrix<double, 2*DoF, DoF>> fin_diff_jacs = get_fin_diff_jacs<2*DoF>(t1, t2);

//       EXPECT_TRUE(fin_diff_jacs.size() == eval_jacs.size());

//       for(int d = 0; d < eval_jacs.size(); ++d)
//       {
//         EXPECT_TRUE((fin_diff_jacs[d] - eval_jacs[d]).isZero(tolerance_));
//         if(!(fin_diff_jacs[d] - eval_jacs[d]).isZero(tolerance_))
//           std::cout << "d = " << d << ", t1 = " << t1 << ", t2 = " << t2 << "\nEval: \n" << eval_jacs[d] << "\n Fin diff: \n" << fin_diff_jacs[d] << "\n\n";
//       }

//       t1 = t2;
//     }
//   }
// }

TEST_F(SE3SplineMPTest, SE3SplineJerkMPTest)
{
  int k = 4;
  int n = 20;
  double dt = 0.1;

  for(int z = 0; z < 1; ++z)
  {
    create_spline(k, n, dt);

    MatrixDoF Q = 0.2 * MatrixDoF::Identity();
    create_mp(estimator::ModelType::JERK, Q);

    double dt_samp = 0.1;
    double t1 = 0.0;
    for(double t2 = t1 + dt_samp; spline_->get_time_range().valid(t2 + dt_samp); t2 += dt_samp)
    {
      std::vector<Eigen::Matrix<double, 3*DoF, DoF>> eval_jacs;
      mp_.get_motion_prior<3*DoF>(t1, t2, spline_, &eval_jacs);

      std::vector<Eigen::Matrix<double, 3*DoF, DoF>> fin_diff_jacs = get_fin_diff_jacs<3*DoF>(t1, t2);

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
}