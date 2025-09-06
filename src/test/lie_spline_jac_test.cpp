#include <vector>
#include <chrono>
#include <memory>
#include <iostream>
#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "lie_groups/so2.hpp"
#include "lie_groups/so3.hpp"
#include "lie_groups/se2.hpp"
#include "lie_groups/se3.hpp"
#include "lie_groups/sonxrn.hpp"
#include "spline/lie_spline.hpp"
#include "spline/sonxrn_spline.hpp"

template <typename G, typename S>
class LieSplineJacTest : public testing::Test
{
protected:
  using TangentT = typename G::TangentT;
  using JacT = typename G::JacT;
  using NonmapT = typename G::NonmapT;
  static const int DoF = G::DoF;

public:
  LieSplineJacTest()
  {}

  void create_spline(int k, int n, double dt)
  {
    k_ = k;
    s_ = S(k);

    std::shared_ptr<std::vector<G>> ctrl_pts(new std::vector<G>);
    for(int l = 0; l < n; ++l)
    {
      ctrl_pts->push_back(G::random());
    }
    s_.init_ctrl_pts(ctrl_pts, 0.0, dt);
    ctrl_pts_ = *ctrl_pts;
  }

  JacT fin_diff_jac(int l, double t, JacT *j1 = nullptr, JacT *j2 = nullptr)
  {
    JacT jac = JacT::Zero();
    NonmapT T_t;
    TangentT vel, acc;
    s_.eval(t, &T_t, j1 == nullptr ? nullptr : &vel, j2 == nullptr ? nullptr : &acc);
    for(int i = 0; i < DoF; ++i)
    {
      TangentT tau = TangentT::Zero();
      tau(i) = h_;
      assert(s_.modify_ctrl_pt(l, G::Exp(tau) * ctrl_pts_[l]));

      NonmapT T_t_pert;
      TangentT vel_pert, acc_pert;
      s_.eval(t, &T_t_pert, j1 == nullptr ? nullptr : &vel_pert, j2 == nullptr ? nullptr : &acc_pert);
      assert(s_.modify_ctrl_pt(l, ctrl_pts_[l]));

      jac.col(i) = ((T_t_pert * T_t.inverse()).Log())/h_;

      if(j1 != nullptr)
        j1->col(i) = (vel_pert - vel) / h_;

      if(j2 != nullptr)
        j2->col(i) = (acc_pert - acc) / h_;
    }

    return jac;
  }

  void fin_diff_vel_acc(double t, TangentT &vel, TangentT &acc)
  {
    NonmapT T_eval, T_eval_pert;
    TangentT vel_eval, vel_eval_pert;
    s_.eval(t, &T_eval, &vel_eval);
    s_.eval(t+h_, &T_eval_pert, &vel_eval_pert);
    vel = (T_eval_pert * T_eval.inverse()).Log()/h_;
    acc = (vel_eval_pert - vel_eval)/h_;
  }

  void get_nonzero_jacs(double t, std::vector<JacT> &j_eval, std::vector<JacT> &j_fin_diff,
                        std::vector<JacT> *j1_eval = nullptr, std::vector<JacT> *j1_fin_diff = nullptr,
                        std::vector<JacT> *j2_eval = nullptr, std::vector<JacT> *j2_fin_diff = nullptr)
  {
    TangentT vel, acc;
    int l_init = s_.eval(t, nullptr, j1_eval == nullptr ? nullptr : &vel, j2_eval == nullptr ? nullptr : &acc,
      &j_eval, j1_eval, j2_eval);

    for(int l = l_init; l < l_init + k_; ++l)
    {
      JacT J0, J1, J2;
      J0 = fin_diff_jac(l, t, j1_fin_diff == nullptr ? nullptr : &J1, j2_fin_diff == nullptr ? nullptr : &J2);

      j_fin_diff.push_back(J0);
      if(j1_fin_diff != nullptr) j1_fin_diff->push_back(J1);
      if(j2_fin_diff != nullptr) j2_fin_diff->push_back(J2);
    }
  }

protected:
  // we need our own copy of ctrl_pts so we can 
  // reset the spline points after we've modified them
  std::vector<G> ctrl_pts_;
  S s_;
  int k_;

  double tolerance_{1e-3};
  double h_{1e-5};

};

typedef LieSplineJacTest<lie_groups::SO2d, spline::LieSpline<lie_groups::SO2d>> SO2SplineJacTest;
typedef LieSplineJacTest<lie_groups::SO3d, spline::LieSpline<lie_groups::SO3d>> SO3SplineJacTest;
typedef LieSplineJacTest<lie_groups::SE2d, spline::LieSpline<lie_groups::SE2d>> SE2SplineJacTest;
typedef LieSplineJacTest<lie_groups::SE3d, spline::LieSpline<lie_groups::SE3d>> SE3SplineJacTest;
typedef LieSplineJacTest<lie_groups::SO2xR2d, spline::SOnxRnSpline<lie_groups::SO2xR2d>> SO2xR2SplineJacTest;
typedef LieSplineJacTest<lie_groups::SO3xR3d, spline::SOnxRnSpline<lie_groups::SO3xR3d>> SO3xR3SplineJacTest;


TEST_F(SO2SplineJacTest, SO2TimeDerivatives)
{
  int k = 4;
  int n = 10;
  double dt = 0.5;

  for(int z = 0; z < 10; ++z)
  {
    create_spline(k, n, dt);

    int num_eval = 100;
    double end_time = dt * (n - k);
    std::vector<double> t_eval;
    for(int i = 0; i < num_eval; ++i)
      t_eval.push_back(end_time * static_cast<double>(i)/num_eval);

    for(const double &t : t_eval)
    {
      Eigen::Matrix<double, 1, 1> vel_eval, acc_eval, vel_fin_diff, acc_fin_diff;
      s_.eval(t, nullptr, &vel_eval, &acc_eval);
      fin_diff_vel_acc(t, vel_fin_diff, acc_fin_diff);

      EXPECT_TRUE((vel_fin_diff - vel_eval).isZero(tolerance_));
      if(!(vel_fin_diff - vel_eval).isZero(tolerance_))
        std::cout << "Eval: \n" << vel_eval << "\n Fin diff: \n" << vel_fin_diff << "\n\n";
      
      EXPECT_TRUE((acc_fin_diff - acc_eval).isZero(tolerance_));
      if(!(acc_fin_diff - acc_eval).isZero(tolerance_))
        std::cout << "Eval: \n" << acc_eval << "\n Fin diff: \n" << acc_fin_diff << "\n\n";
    }
  }
}

TEST_F(SO2SplineJacTest, SO2Jacobians)
{
  int k = 4;
  int n = 10;
  double dt = 0.5;

  for(int z = 0; z < 10; ++z)
  {
    create_spline(k, n, dt);

    int num_eval = 100;
    double end_time = dt * (n - k);
    std::vector<double> t_eval;
    for(int i = 0; i < num_eval; ++i)
      t_eval.push_back(end_time * static_cast<double>(i)/num_eval);

    for(const double &t : t_eval)
    {
      std::vector<Eigen::Matrix<double, 1, 1>> j_eval, j_fin_diff, j1_eval, j1_fin_diff, j2_eval, j2_fin_diff;
      get_nonzero_jacs(t, j_eval, j_fin_diff, &j1_eval, &j1_fin_diff, &j2_eval, &j2_fin_diff);

      for(int i = 0; i < k_; ++i)
      {
        if(j_fin_diff[i].hasNaN()) continue;

        EXPECT_TRUE((j_fin_diff[i] - j_eval[i]).isZero(1e-2));
        if(!(j_fin_diff[i] - j_eval[i]).isZero(1e-2))
          std::cout << "Eval: \n" << j_eval[i] << "\n Fin diff: \n" << j_fin_diff[i] << "\n\n";


        if(j1_fin_diff[i].hasNaN()) continue;

        EXPECT_TRUE((j1_fin_diff[i] - j1_eval[i]).isZero(tolerance_));
        if(!(j1_fin_diff[i] - j1_eval[i]).isZero(tolerance_))
          std::cout << "Eval: \n" << j1_eval[i] << "\n Fin diff: \n" << j1_fin_diff[i] << "\n\n";


        if(j2_fin_diff[i].hasNaN()) continue;

        EXPECT_TRUE((j2_fin_diff[i] - j2_eval[i]).isZero(tolerance_));
        if(!(j2_fin_diff[i] - j2_eval[i]).isZero(tolerance_))
          std::cout << "Eval: \n" << j2_eval[i] << "\n Fin diff: \n" << j2_fin_diff[i] << "\n\n";
      }
    }
  }
}

TEST_F(SO3SplineJacTest, SO3TimeDerivatives)
{
  int k = 4;
  int n = 10;
  double dt = 1.0;

  for(int z = 0; z < 10; ++z)
  {
    create_spline(k, n, dt);

    int num_eval = 100;
    double end_time = dt * (n - k);
    std::vector<double> t_eval;
    for(int i = 0; i < num_eval; ++i)
      t_eval.push_back(end_time * static_cast<double>(i)/num_eval);

    for(const double &t : t_eval)
    {
      Eigen::Matrix<double, 3, 1> vel_eval, acc_eval, vel_fin_diff, acc_fin_diff;
      s_.eval(t, nullptr, &vel_eval, &acc_eval);
      fin_diff_vel_acc(t, vel_fin_diff, acc_fin_diff);

      EXPECT_TRUE((vel_fin_diff - vel_eval).isZero(tolerance_));
      if(!(vel_fin_diff - vel_eval).isZero(tolerance_))
        std::cout << "Eval: \n" << vel_eval << "\n Fin diff: \n" << vel_fin_diff << "\n\n";

      EXPECT_TRUE((acc_fin_diff - acc_eval).isZero(tolerance_));
      if(!(acc_fin_diff - acc_eval).isZero(tolerance_))
        std::cout << "Eval: \n" << acc_eval << "\n Fin diff: \n" << acc_fin_diff << "\n\n";
    }
  }
}

TEST_F(SO3SplineJacTest, SO3Jacobians)
{
  int k = 4;
  int n = 10;
  double dt = 1.0;

  for(int z = 0; z < 10; ++z)
  {
    create_spline(k, n, dt);

    int num_eval = 100;
    double end_time = dt * (n - k);
    std::vector<double> t_eval;
    for(int i = 0; i < num_eval; ++i)
      t_eval.push_back(end_time * static_cast<double>(i)/num_eval);

    // std::vector<Eigen::Matrix<double, 3, 3>> j_eval, j1_eval, j2_eval;
    // Eigen::Matrix<double, 3, 1> vel, acc;

    // auto start = std::chrono::high_resolution_clock::now();
    // s_.eval(0.0, &vel, &acc);
    // s_.get_jac(&j_eval, &j1_eval, &j2_eval);
    // auto stop = std::chrono::high_resolution_clock::now();
    // std::cout << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << std::endl;

    for(const double &t : t_eval)
    {
      std::vector<Eigen::Matrix<double, 3, 3>> j_eval, j_fin_diff, j1_eval, j1_fin_diff, j2_eval, j2_fin_diff;
      get_nonzero_jacs(t, j_eval, j_fin_diff, &j1_eval, &j1_fin_diff, &j2_eval, &j2_fin_diff);

      for(int i = 0; i < k_; ++i)
      {
        if(j_fin_diff[i].hasNaN()) continue;

        EXPECT_TRUE((j_fin_diff[i] - j_eval[i]).isZero(1e-2));
        if(!(j_fin_diff[i] - j_eval[i]).isZero(1e-2))
          std::cout << "Eval: \n" << j_eval[i] << "\n Fin diff: \n" << j_fin_diff[i] << "\n\n";


        if(j1_fin_diff[i].hasNaN()) continue;

        EXPECT_TRUE((j1_fin_diff[i] - j1_eval[i]).isZero(tolerance_));
        if(!(j1_fin_diff[i] - j1_eval[i]).isZero(tolerance_))
          std::cout << "Eval: \n" << j1_eval[i] << "\n Fin diff: \n" << j1_fin_diff[i] << "\n\n";


        if(j2_fin_diff[i].hasNaN()) continue;

        EXPECT_TRUE((j2_fin_diff[i] - j2_eval[i]).isZero(tolerance_));
        if(!(j2_fin_diff[i] - j2_eval[i]).isZero(tolerance_))
          std::cout << "Eval: \n" << j2_eval[i] << "\n Fin diff: \n" << j2_fin_diff[i] << "\n\n";
      }
    }
  }
}

TEST_F(SE2SplineJacTest, SE2TimeDerivatives)
{
  int k = 4;
  int n = 10;
  double dt = 1.0;
  h_ = 1e-4;
  tolerance_ = 1e-2;

  for(int z = 0; z < 10; ++z)
  {
    create_spline(k, n, dt);

    int num_eval = 100;
    double end_time = dt * (n - k);
    std::vector<double> t_eval;
    for(int i = 0; i < num_eval; ++i)
      t_eval.push_back(end_time * static_cast<double>(i)/num_eval);

    for(const double &t : t_eval)
    {
      Eigen::Matrix<double, 3, 1> vel_eval, acc_eval, vel_fin_diff, acc_fin_diff;
      s_.eval(t, nullptr, &vel_eval, &acc_eval);
      fin_diff_vel_acc(t, vel_fin_diff, acc_fin_diff);

      EXPECT_TRUE((vel_fin_diff - vel_eval).isZero(tolerance_));
      if(!(vel_fin_diff - vel_eval).isZero(tolerance_))
        std::cout << "Eval: \n" << vel_eval << "\n Fin diff: \n" << vel_fin_diff << "\n\n";

      EXPECT_TRUE((acc_fin_diff - acc_eval).isZero(tolerance_));
      if(!(acc_fin_diff - acc_eval).isZero(tolerance_))
        std::cout << "Eval: \n" << acc_eval << "\n Fin diff: \n" << acc_fin_diff << "\n\n";
    }
  }
}

TEST_F(SE2SplineJacTest, SE2Jacobians)
{
  int k = 4;
  int n = 10;
  double dt = 1.0;
  h_ = 1e-4;

  for(int z = 0; z < 10; ++z)
  {
    create_spline(k, n, dt);

    int num_eval = 100;
    double end_time = dt * (n - k);
    std::vector<double> t_eval;
    for(int i = 0; i < num_eval; ++i)
      t_eval.push_back(end_time * static_cast<double>(i)/num_eval);

    for(const double &t : t_eval)
    {
      std::vector<Eigen::Matrix3d> j_eval, j_fin_diff, j1_eval, j1_fin_diff, j2_eval, j2_fin_diff;
      get_nonzero_jacs(t, j_eval, j_fin_diff, &j1_eval, &j1_fin_diff, &j2_eval, &j2_fin_diff);

      for(int i = 0; i < k_; ++i)
      {
        if(!j_fin_diff[i].allFinite()) continue;

        EXPECT_TRUE((j_fin_diff[i] - j_eval[i]).isZero(tolerance_));
        if(!(j_fin_diff[i] - j_eval[i]).isZero(tolerance_))
          std::cout << "Eval: \n" << j_eval[i] << "\n Fin diff: \n" << j_fin_diff[i] << "\n\n";

        if(!j1_fin_diff[i].allFinite()) continue;

        EXPECT_TRUE((j1_fin_diff[i] - j1_eval[i]).isZero(tolerance_));
        if(!(j1_fin_diff[i] - j1_eval[i]).isZero(tolerance_))
          std::cout << "Eval: \n" << j1_eval[i] << "\n Fin diff: \n" << j1_fin_diff[i] << "\n\n";

        if(!j2_fin_diff[i].allFinite()) continue;

        EXPECT_TRUE((j2_fin_diff[i] - j2_eval[i]).isZero(tolerance_));
        if(!(j2_fin_diff[i] - j2_eval[i]).isZero(tolerance_))
          std::cout << "Eval: \n" << j2_eval[i] << "\n Fin diff: \n" << j2_fin_diff[i] << "\n\n";
      }
    }
  }
}

TEST_F(SE3SplineJacTest, SE3TimeDerivatives)
{
  int k = 4;
  int n = 10;
  double dt = 1.0;

  for(int z = 0; z < 10; ++z)
  {
    create_spline(k, n, dt);

    int num_eval = 100;
    double end_time = dt * (n - k);
    std::vector<double> t_eval;
    for(int i = 0; i < num_eval; ++i)
      t_eval.push_back(end_time * static_cast<double>(i)/num_eval);

    for(const double &t : t_eval)
    {
      Eigen::Matrix<double, 6, 1> vel_eval, acc_eval, vel_fin_diff, acc_fin_diff;
      s_.eval(t, nullptr, &vel_eval, &acc_eval);
      fin_diff_vel_acc(t, vel_fin_diff, acc_fin_diff);

      EXPECT_TRUE((vel_fin_diff - vel_eval).isZero(tolerance_));
      if(!(vel_fin_diff - vel_eval).isZero(tolerance_))
        std::cout << "Eval: \n" << vel_eval << "\n Fin diff: \n" << vel_fin_diff << "\n\n";

      EXPECT_TRUE((acc_fin_diff - acc_eval).isZero(tolerance_));
      if(!(acc_fin_diff - acc_eval).isZero(tolerance_))
        std::cout << "Eval: \n" << acc_eval << "\n Fin diff: \n" << acc_fin_diff << "\n\n";
    }
  }
}

TEST_F(SE3SplineJacTest, SE3Jacobians)
{
  int k = 4;
  int n = 10;
  double dt = 1.0;

  for(int z = 0; z < 10; ++z)
  {
    create_spline(k, n, dt);

    int num_eval = 100;
    double end_time = dt * (n - k);
    std::vector<double> t_eval;
    for(int i = 0; i < num_eval; ++i)
      t_eval.push_back(end_time * static_cast<double>(i)/num_eval);

    // std::vector<Eigen::Matrix<double, 6, 6>> j_eval, j1_eval, j2_eval;
    // Eigen::Matrix<double, 6, 1> vel, acc;

    // auto start = std::chrono::high_resolution_clock::now();
    // s_.eval(0.0);
    // // s_.get_jac(&j_eval);
    // auto stop = std::chrono::high_resolution_clock::now();
    // std::cout << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << std::endl;

    for(const double &t : t_eval)
    {
      std::vector<Eigen::Matrix<double, 6, 6>> j_eval, j_fin_diff, j1_eval, j1_fin_diff, j2_eval, j2_fin_diff;
      get_nonzero_jacs(t, j_eval, j_fin_diff, &j1_eval, &j1_fin_diff, &j2_eval, &j2_fin_diff);

      for(int i = 0; i < k_; ++i)
      {
        if(j_fin_diff[i].hasNaN()) continue;

        EXPECT_TRUE((j_fin_diff[i] - j_eval[i]).isZero(1e-2));
        if(!(j_fin_diff[i] - j_eval[i]).isZero(1e-2))
          std::cout << "Eval: \n" << j_eval[i] << "\n Fin diff: \n" << j_fin_diff[i] << "\n\n";
        // finite differencing zeros out random columns of the bottom right 3x3 matrix sometimes. Not sure why


        if(j1_fin_diff[i].hasNaN()) continue;

        EXPECT_TRUE((j1_fin_diff[i] - j1_eval[i]).isZero(tolerance_));
        if(!(j1_fin_diff[i] - j1_eval[i]).isZero(tolerance_))
          std::cout << "Eval: \n" << j1_eval[i] << "\n Fin diff: \n" << j1_fin_diff[i] << "\n\n";


        if(j2_fin_diff[i].hasNaN()) continue;

        EXPECT_TRUE((j2_fin_diff[i] - j2_eval[i]).isZero(tolerance_));
        if(!(j2_fin_diff[i] - j2_eval[i]).isZero(tolerance_))
          std::cout << "Eval: \n" << j2_eval[i] << "\n Fin diff: \n" << j2_fin_diff[i] << "\n\n";
      }
    }
  }
}


TEST_F(SO2xR2SplineJacTest, SO2xR2TimeDerivatives)
{
  int k = 4;
  int n = 10;
  double dt = 1.0;

  for(int z = 0; z < 10; ++z)
  {
    create_spline(k, n, dt);

    int num_eval = 100;
    double end_time = dt * (n - k);
    std::vector<double> t_eval;
    for(int i = 0; i < num_eval; ++i)
      t_eval.push_back(end_time * static_cast<double>(i)/num_eval);

    for(const double &t : t_eval)
    {
      Eigen::Matrix<double, 3, 1> vel_eval, acc_eval, vel_fin_diff, acc_fin_diff;
      s_.eval(t, nullptr, &vel_eval, &acc_eval);
      fin_diff_vel_acc(t, vel_fin_diff, acc_fin_diff);

      EXPECT_TRUE((vel_fin_diff - vel_eval).isZero(tolerance_));
      if(!(vel_fin_diff - vel_eval).isZero(tolerance_))
        std::cout << "Eval: \n" << vel_eval << "\n Fin diff: \n" << vel_fin_diff << "\n\n";

      EXPECT_TRUE((acc_fin_diff - acc_eval).isZero(tolerance_));
      if(!(acc_fin_diff - acc_eval).isZero(tolerance_))
        std::cout << "Eval: \n" << acc_eval << "\n Fin diff: \n" << acc_fin_diff << "\n\n";
    }
  }
}

TEST_F(SO2xR2SplineJacTest, SO2xR2Jacobians)
{
  int k = 4;
  int n = 10;
  double dt = 1.0;

  for(int z = 0; z < 10; ++z)
  {
    create_spline(k, n, dt);

    int num_eval = 100;
    double end_time = dt * (n - k);
    std::vector<double> t_eval;
    for(int i = 0; i < num_eval; ++i)
      t_eval.push_back(end_time * static_cast<double>(i)/num_eval);

    for(const double &t : t_eval)
    {
      std::vector<Eigen::Matrix<double, 3, 3>> j_eval, j_fin_diff, j1_eval, j1_fin_diff, j2_eval, j2_fin_diff;
      get_nonzero_jacs(t, j_eval, j_fin_diff, &j1_eval, &j1_fin_diff, &j2_eval, &j2_fin_diff);

      for(int i = 0; i < k_; ++i)
      {
        if(j_fin_diff[i].hasNaN()) continue;

        EXPECT_TRUE((j_fin_diff[i] - j_eval[i]).isZero(1e-2));
        if(!(j_fin_diff[i] - j_eval[i]).isZero(1e-2))
          std::cout << "Eval: \n" << j_eval[i] << "\n Fin diff: \n" << j_fin_diff[i] << "\n\n";


        if(j1_fin_diff[i].hasNaN()) continue;

        EXPECT_TRUE((j1_fin_diff[i] - j1_eval[i]).isZero(tolerance_));
        if(!(j1_fin_diff[i] - j1_eval[i]).isZero(tolerance_))
          std::cout << "Eval: \n" << j1_eval[i] << "\n Fin diff: \n" << j1_fin_diff[i] << "\n\n";


        if(j2_fin_diff[i].hasNaN()) continue;

        EXPECT_TRUE((j2_fin_diff[i] - j2_eval[i]).isZero(tolerance_));
        if(!(j2_fin_diff[i] - j2_eval[i]).isZero(tolerance_))
          std::cout << "Eval: \n" << j2_eval[i] << "\n Fin diff: \n" << j2_fin_diff[i] << "\n\n";
      }
    }
  }
}

TEST_F(SO3xR3SplineJacTest, SO3xR3TimeDerivatives)
{
  int k = 4;
  int n = 10;
  double dt = 1.0;

  for(int z = 0; z < 10; ++z)
  {
    create_spline(k, n, dt);

    int num_eval = 100;
    double end_time = dt * (n - k);
    std::vector<double> t_eval;
    for(int i = 0; i < num_eval; ++i)
      t_eval.push_back(end_time * static_cast<double>(i)/num_eval);

    for(const double &t : t_eval)
    {
      Eigen::Matrix<double, 6, 1> vel_eval, acc_eval, vel_fin_diff, acc_fin_diff;
      s_.eval(t, nullptr, &vel_eval, &acc_eval);
      fin_diff_vel_acc(t, vel_fin_diff, acc_fin_diff);

      EXPECT_TRUE((vel_fin_diff - vel_eval).isZero(tolerance_));
      if(!(vel_fin_diff - vel_eval).isZero(tolerance_))
        std::cout << "Eval: \n" << vel_eval << "\n Fin diff: \n" << vel_fin_diff << "\n\n";

      EXPECT_TRUE((acc_fin_diff - acc_eval).isZero(tolerance_));
      if(!(acc_fin_diff - acc_eval).isZero(tolerance_))
        std::cout << "Eval: \n" << acc_eval << "\n Fin diff: \n" << acc_fin_diff << "\n\n";
    }
  }
}

TEST_F(SO3xR3SplineJacTest, SO3xR3Jacobians)
{
  int k = 4;
  int n = 10;
  double dt = 1.0;

  for(int z = 0; z < 10; ++z)
  {
    create_spline(k, n, dt);

    int num_eval = 100;
    double end_time = dt * (n - k);
    std::vector<double> t_eval;
    for(int i = 0; i < num_eval; ++i)
      t_eval.push_back(end_time * static_cast<double>(i)/num_eval);

    for(const double &t : t_eval)
    {
      std::vector<Eigen::Matrix<double, 6, 6>> j_eval, j_fin_diff, j1_eval, j1_fin_diff, j2_eval, j2_fin_diff;
      get_nonzero_jacs(t, j_eval, j_fin_diff, &j1_eval, &j1_fin_diff, &j2_eval, &j2_fin_diff);

      for(int i = 0; i < k_; ++i)
      {
        if(j_fin_diff[i].hasNaN()) continue;

        EXPECT_TRUE((j_fin_diff[i] - j_eval[i]).isZero(1e-2));
        if(!(j_fin_diff[i] - j_eval[i]).isZero(1e-2))
          std::cout << "Eval: \n" << j_eval[i] << "\n Fin diff: \n" << j_fin_diff[i] << "\n\n";


        if(j1_fin_diff[i].hasNaN()) continue;

        EXPECT_TRUE((j1_fin_diff[i] - j1_eval[i]).isZero(tolerance_));
        if(!(j1_fin_diff[i] - j1_eval[i]).isZero(tolerance_))
          std::cout << "Eval: \n" << j1_eval[i] << "\n Fin diff: \n" << j1_fin_diff[i] << "\n\n";


        if(j2_fin_diff[i].hasNaN()) continue;

        EXPECT_TRUE((j2_fin_diff[i] - j2_eval[i]).isZero(tolerance_));
        if(!(j2_fin_diff[i] - j2_eval[i]).isZero(tolerance_))
          std::cout << "Eval: \n" << j2_eval[i] << "\n Fin diff: \n" << j2_fin_diff[i] << "\n\n";
      }
    }
  }
}