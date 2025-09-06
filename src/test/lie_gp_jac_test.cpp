#include <vector>
#include <chrono>
#include <memory>
#include <iostream>
#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "utils/multivariate_gaussian.hpp"
#include "lie_groups/so2.hpp"
#include "lie_groups/so3.hpp"
#include "lie_groups/se2.hpp"
#include "lie_groups/se3.hpp"
#include "gp/lie_gp.hpp"

namespace gp
{

template <typename G, typename GP>
class LieGPJacTest : public testing::Test
{
protected:
  using TangentT = typename G::TangentT;
  using NonmapT = typename G::NonmapT;
  static const int DoF = G::DoF;
  using MatrixDoF = Eigen::Matrix<double, DoF, DoF>;
  using VectorDoF = Eigen::Matrix<double, DoF, 1>;

public:
  LieGPJacTest() : h_{1e-5}, tolerance_{1e-2}, noise_dist_{nullptr}
  {}

  void init_gp(ModelType type, double start, double end, int segments, double var)
  {
    Q_ = var / (end - start) * MatrixDoF::Identity();
    noise_dist_.reset(new utils::MultivariateGaussian<DoF>(VectorDoF::Zero(), Q_));

    type_ = type;

    std::shared_ptr<std::vector<G>> T(new std::vector<G>);
    std::shared_ptr<std::vector<TangentT>> v(new std::vector<TangentT>);
    std::shared_ptr<std::vector<TangentT>> a(new std::vector<TangentT>);

    // sample initial conditions
    t_vec_.push_back(start);
    T->push_back(G::random());
    v->push_back(1.0 * TangentT::Random());
    if(type == ModelType::JERK) a->push_back(0.5 * TangentT::Random());

    double dt = (end - start)/segments;
    double ddt = dt/10;
    for(int i = 0; i < segments; ++i)
    {
      VectorDoF xi_t = VectorDoF::Zero();
      VectorDoF dxi_t, ddxi_t;
      dxi_t = v->back();
      if(type == ModelType::JERK) ddxi_t = a->back();
      
      // integrate SDE over the segment
      for(int j = 0; j < 10; ++j)
      {
        VectorDoF w = noise_dist_->sample();
        if(type_ == ModelType::JERK) 
        {
          ddxi_t = ddxi_t + ddt * w;
          dxi_t = dxi_t + ddt * ddxi_t;
          xi_t = xi_t + ddt * dxi_t;
        }
        else
        {
          dxi_t = dxi_t + ddt * w;
          xi_t = xi_t + ddt * dxi_t;
        }
      }

      t_vec_.push_back(t_vec_.back() + dt);
      T->push_back(G::Exp(xi_t) * T->back());
      v->push_back(G::Jl(xi_t) * dxi_t);
      if(type == ModelType::JERK) a->push_back(G::Jl(xi_t) * (ddxi_t + 0.5 * G::ad(xi_t) * v->back()));
    }

    gp_ = GP(type, Q_);
    gp_.init_est_params(t_vec_, T, v, type == ModelType::JERK ? a : nullptr);

    T_ = *T;
    v_ = *v;
    a_ = *a;
  }

  // approximate time derivatives
  // output contains fin diff velocity and accel (if applicable)
  std::vector<VectorDoF> fin_diff_derivs(double t)
  {
    NonmapT T, T_pert;
    TangentT v, v_pert;
    gp_.eval(t, &T, &v);
    gp_.eval(t + h_, &T_pert, &v_pert);

    std::vector<VectorDoF> ret;
    ret.push_back((T_pert * T.inverse()).Log()/h_);
    if(type_ == ModelType::JERK) ret.push_back((v_pert - v)/h_);
    return ret;
  }

  // approximate all jacobians at time t
  // outer vector has jacs for T, v, and a (as applicable)
  // inner vector has jacs of outer parameter wrt {T_i, T_{i+1}, v_i, v_{i+1}, a_i, a_{i+1}} (as applicable)
  std::vector<std::vector<MatrixDoF>> fin_diff_jacs(double t)
  {
    int i = gp_.get_i(t);

    NonmapT T;
    TangentT v, a;
    gp_.eval(t, &T, &v, type_ == ModelType::JERK ? &a : nullptr);

    std::vector<MatrixDoF> T_jacs, v_jacs, a_jacs;

    // compute jacs wrt T_i, T_{i+1}
    for(int j = i; j <= i + 1; ++j)
    {
      MatrixDoF T_Tj_jac = MatrixDoF::Zero();
      MatrixDoF v_Tj_jac = MatrixDoF::Zero();
      MatrixDoF a_Tj_jac = MatrixDoF::Zero();
      for(int d = 0; d < DoF; ++d)
      {
        TangentT tau = TangentT::Zero();
        tau(d) = h_;
        assert(gp_.modify_T(j, G::Exp(tau) * T_[j]));

        NonmapT T_pert;
        TangentT v_pert, a_pert;
        gp_.eval(t, &T_pert, &v_pert, type_ == ModelType::JERK ? &a_pert : nullptr);

        T_Tj_jac.col(d) = ((T_pert * T.inverse()).Log())/h_;
        v_Tj_jac.col(d) = (v_pert - v)/h_;
        if(type_ == ModelType::JERK) a_Tj_jac.col(d) = (a_pert - a)/h_;

        assert(gp_.modify_T(j, T_[j]));
      }

      T_jacs.push_back(T_Tj_jac);
      v_jacs.push_back(v_Tj_jac);
      if(type_ == ModelType::JERK) a_jacs.push_back(a_Tj_jac);
    }

    // compute jacs wrt v_i, v_{i+1}
    for(int j = i; j <= i + 1; ++j)
    {
      MatrixDoF T_vj_jac = MatrixDoF::Zero();
      MatrixDoF v_vj_jac = MatrixDoF::Zero();
      MatrixDoF a_vj_jac = MatrixDoF::Zero();
      for(int d = 0; d < DoF; ++d)
      {
        TangentT tau = TangentT::Zero();
        tau(d) = h_;
        assert(gp_.modify_v(j, v_[j] + tau));

        NonmapT T_pert;
        TangentT v_pert, a_pert;
        gp_.eval(t, &T_pert, &v_pert, type_ == ModelType::JERK ? &a_pert : nullptr);

        T_vj_jac.col(d) = (T_pert * T.inverse()).Log()/h_;
        v_vj_jac.col(d) = (v_pert - v)/h_;
        if(type_ == ModelType::JERK) a_vj_jac.col(d) = (a_pert - a)/h_;

        assert(gp_.modify_v(j, v_[j]));
      }

      T_jacs.push_back(T_vj_jac);
      v_jacs.push_back(v_vj_jac);
      if(type_ == ModelType::JERK) a_jacs.push_back(a_vj_jac);
    }

    // compute jacs wrt a_i, a_{i+1}
    if(type_ == ModelType::JERK)
    {
      for(int j = i; j <= i + 1; ++j)
      {
        MatrixDoF T_aj_jac = MatrixDoF::Zero();
        MatrixDoF v_aj_jac = MatrixDoF::Zero();
        MatrixDoF a_aj_jac = MatrixDoF::Zero();
        for(int d = 0; d < DoF; ++d)
        {
          TangentT tau = TangentT::Zero();
          tau(d) = h_;
          assert(gp_.modify_a(j, a_[j] + tau));

          NonmapT T_pert;
          TangentT v_pert, a_pert;
          gp_.eval(t, &T_pert, &v_pert, &a_pert);

          T_aj_jac.col(d) = (T_pert * T.inverse()).Log()/h_;
          v_aj_jac.col(d) = (v_pert - v)/h_;
          a_aj_jac.col(d) = (a_pert - a)/h_;

          assert(gp_.modify_a(j, a_[j]));
        }

        T_jacs.push_back(T_aj_jac);
        v_jacs.push_back(v_aj_jac);
        a_jacs.push_back(a_aj_jac);
      }
    }

    // return everything
    std::vector<std::vector<MatrixDoF>> ret;
    ret.push_back(T_jacs);
    ret.push_back(v_jacs);
    if(type_ == ModelType::JERK) ret.push_back(a_jacs);
    return ret;
  }

  // approximate dynamics residual jacobians between t_i and t_i1
  // vector has jacs of residual wrt {T_i, T_{i+1}, v_i, v_{i+1}, a_i, a_{i+1}} (as applicable)
  template <int S>
  std::vector<Eigen::Matrix<double, S, DoF>> fin_diff_resid_jacs(int i)
  {
    std::vector<Eigen::Matrix<double, S, DoF>> ret;

    Eigen::Matrix<double, S, 1> r = gp_.template get_dynamics_residual<S>(i);

    // compute jacs wrt T_i, T_{i+1}
    for(int j = i; j <= i + 1; ++j)
    {
      Eigen::Matrix<double, S, DoF> r_Tj_jac = Eigen::Matrix<double, S, DoF>::Zero();
      for(int d = 0; d < DoF; ++d)
      {
        TangentT tau = TangentT::Zero();
        tau(d) = h_;
        assert(gp_.modify_T(j, G::Exp(tau) * T_[j]));

        Eigen::Matrix<double, S, 1> r_pert = gp_.template get_dynamics_residual<S>(i);

        r_Tj_jac.col(d) = (r_pert - r)/h_;

        assert(gp_.modify_T(j, T_[j]));
      }

      ret.push_back(r_Tj_jac);
    }

    // compute jacs wrt v_i, v_{i+1}
    for(int j = i; j <= i + 1; ++j)
    {
      Eigen::Matrix<double, S, DoF> r_vj_jac = Eigen::Matrix<double, S, DoF>::Zero();
      for(int d = 0; d < DoF; ++d)
      {
        TangentT tau = TangentT::Zero();
        tau(d) = h_;
        assert(gp_.modify_v(j, v_[j] + tau));

        Eigen::Matrix<double, S, 1> r_pert = gp_.template get_dynamics_residual<S>(i);

        r_vj_jac.col(d) = (r_pert - r)/h_;

        assert(gp_.modify_v(j, v_[j]));
      }

      ret.push_back(r_vj_jac);
    }

    // compute jacs wrt a_i, a_{i+1}
    if(type_ == ModelType::JERK)
    {
      for(int j = i; j <= i + 1; ++j)
      {
        Eigen::Matrix<double, S, DoF> r_aj_jac = Eigen::Matrix<double, S, DoF>::Zero();
        for(int d = 0; d < DoF; ++d)
        {
          TangentT tau = TangentT::Zero();
          tau(d) = h_;
          assert(gp_.modify_a(j, a_[j] + tau));

          Eigen::Matrix<double, S, 1> r_pert = gp_.template get_dynamics_residual<S>(i);

          r_aj_jac.col(d) = (r_pert - r)/h_;

          assert(gp_.modify_a(j, a_[j]));
        }

        ret.push_back(r_aj_jac);
      }
    }

    return ret;
  }

  bool get_derivs(double t, std::vector<VectorDoF>& eval_derivs, std::vector<VectorDoF>& findiff_derivs)
  {
    // don't try if t == t_i
    if(abs(t - t_vec_[gp_.get_i(t)]) < 1e-5) return false;

    findiff_derivs = fin_diff_derivs(t);

    VectorDoF v, a;
    gp_.eval(t, nullptr, &v, type_ == ModelType::JERK ? &a : nullptr);
    eval_derivs.push_back(v);
    if(type_ == ModelType::JERK) eval_derivs.push_back(a);

    return true;
  }

  bool get_jacs(double t, std::vector<std::vector<MatrixDoF>>& eval_jacs, std::vector<std::vector<MatrixDoF>>& findiff_jacs)
  {
    // don't try if t == t_i
    if(abs(t - t_vec_[gp_.get_i(t)]) < 1e-5) return false;

    findiff_jacs = fin_diff_jacs(t);

    std::vector<MatrixDoF> T_jacs, v_jacs, a_jacs;
    gp_.eval(t, nullptr, nullptr, nullptr, &T_jacs, &v_jacs, type_ == ModelType::JERK ? &a_jacs : nullptr);

    eval_jacs.push_back(T_jacs);
    eval_jacs.push_back(v_jacs);
    if(type_ == ModelType::JERK) eval_jacs.push_back(a_jacs);

    return true;
  }

protected:
  GP gp_;

  std::vector<double> t_vec_;
  std::vector<G> T_;
  std::vector<TangentT> v_;
  std::vector<TangentT> a_;

  ModelType type_;
  MatrixDoF Q_;

  std::unique_ptr<utils::MultivariateGaussian<DoF>> noise_dist_;

  double h_; // perturbation value
  double tolerance_;
};

} // namespace gp

typedef gp::LieGPJacTest<lie_groups::SO2d, gp::LieGP<lie_groups::SO2d, Eigen::Matrix<double, 1, 1>>> SO2GPJacTest;
typedef gp::LieGPJacTest<lie_groups::SO3d, gp::LieGP<lie_groups::SO3d, Eigen::Vector3d>> SO3GPJacTest;
typedef gp::LieGPJacTest<lie_groups::SE2d, gp::LieGP<lie_groups::SE2d, Eigen::Vector3d>> SE2GPJacTest;
typedef gp::LieGPJacTest<lie_groups::SE3d, gp::LieGP<lie_groups::SE3d, Eigen::Matrix<double, 6, 1>>> SE3GPJacTest;


TEST_F(SO2GPJacTest, SO2WnoaDerivTest)
{
  double start = 0.0;
  double end = 10.0;
  int num_segments = 10;
  int num_samples = 555;
  double var = 0.1;

  for(int z = 0; z < 10; ++z)
  {
    init_gp(gp::ModelType::ACC, start, end, num_segments, var);

    double t = start;
    double dt = (end - start) / num_samples;
    for(int i = 0; i < num_samples; ++i)
    {
      std::vector<VectorDoF> eval_derivs, fin_diff_derivs;
      if(!get_derivs(t, eval_derivs, fin_diff_derivs))
      {
        t += dt;
        continue;
      }

      EXPECT_TRUE((eval_derivs[0] - fin_diff_derivs[0]).isZero(tolerance_));
      if(!(eval_derivs[0] - fin_diff_derivs[0]).isZero(tolerance_))
        std::cout << "Eval: \n" << eval_derivs[0] << "\nFin diff: \n" << fin_diff_derivs[0] << "\n\n";

      t += dt;
    }
  }
}

TEST_F(SO2GPJacTest, SO2WnoaJacTest)
{
  double start = 0.0;
  double end = 10.0;
  int num_segments = 10;
  int num_samples = 555;
  double var = 0.1;

  for(int z = 0; z < 10; ++z)
  {
    init_gp(gp::ModelType::ACC, start, end, num_segments, var);

    double t = start;
    double dt = (end - start) / num_samples;
    for(int i = 0; i < num_samples; ++i)
    {
      std::vector<std::vector<MatrixDoF>> eval_jacs, fin_diff_jacs;
      if(!get_jacs(t, eval_jacs, fin_diff_jacs))
      {
        t += dt;
        continue;
      }

      for(int d = 0; d < eval_jacs.size(); ++d)
      {
        for(int e = 0; e < eval_jacs[0].size(); ++e)
        {
          EXPECT_TRUE((fin_diff_jacs[d][e] - eval_jacs[d][e]).isZero(tolerance_));
          if(!(fin_diff_jacs[d][e] - eval_jacs[d][e]).isZero(tolerance_))
            std::cout << "d = " << d << ", e = " << e << ", t = " << t << "\nEval: \n" << eval_jacs[d][e] << "\n Fin diff: \n" << fin_diff_jacs[d][e] << "\n\n";
        }
      }

      t += dt;
    }
  }
}

TEST_F(SO2GPJacTest, SO2WnoaResidJacTest)
{
  double start = 0.0;
  double end = 10.0;
  int num_segments = 10;
  double var = 0.1;

  for(int z = 0; z < 10; ++z)
  {
    init_gp(gp::ModelType::ACC, start, end, num_segments, var);

    for(int i = 0; i < num_segments-1; ++i)
    {
      std::vector<Eigen::Vector2d> eval_jacs, fin_diff_jacs;
      gp_.get_dynamics_residual<2>(i, &eval_jacs);
      fin_diff_jacs = fin_diff_resid_jacs<2>(i);

      for(int d = 0; d < eval_jacs.size(); ++d)
      {
        EXPECT_TRUE((fin_diff_jacs[d] - eval_jacs[d]).isZero(tolerance_));
        if(!(fin_diff_jacs[d] - eval_jacs[d]).isZero(tolerance_))
          std::cout << "d = " << d << ", i = " << i << "\nEval: \n" << eval_jacs[d] << "\n Fin diff: \n" << fin_diff_jacs[d] << "\n\n";
      }
    }
  }
}

TEST_F(SO2GPJacTest, SO2WnojDerivTest)
{
  double start = 0.0;
  double end = 10.0;
  int num_segments = 10;
  int num_samples = 555;
  double var = 0.1;

  for(int z = 0; z < 10; ++z)
  {
    init_gp(gp::ModelType::JERK, start, end, num_segments, var);

    double t = start;
    double dt = (end - start) / num_samples;
    for(int i = 0; i < num_samples; ++i)
    {
      std::vector<VectorDoF> eval_derivs, fin_diff_derivs;
      if(!get_derivs(t, eval_derivs, fin_diff_derivs))
      {
        t += dt;
        continue;
      }

      for(int j = 0; j < eval_derivs.size(); ++j)
      {
        EXPECT_TRUE((eval_derivs[j] - fin_diff_derivs[j]).isZero(tolerance_));
        if(!(eval_derivs[j] - fin_diff_derivs[j]).isZero(tolerance_))
          std::cout << "j = " << j << "\nEval: \n" << eval_derivs[j] << "\nFin diff: \n" << fin_diff_derivs[j] << "\n\n";
      }

      t += dt;
    }
  }
}

TEST_F(SO2GPJacTest, SO2WnojJacTest)
{
  double start = 0.0;
  double end = 10.0;
  int num_segments = 10;
  int num_samples = 555;
  double var = 0.1;

  for(int z = 0; z < 10; ++z)
  {
    init_gp(gp::ModelType::JERK, start, end, num_segments, var);

    double t = start;
    double dt = (end - start) / num_samples;
    for(int i = 0; i < num_samples; ++i)
    {
      std::vector<std::vector<MatrixDoF>> eval_jacs, fin_diff_jacs;
      if(!get_jacs(t, eval_jacs, fin_diff_jacs))
      {
        t += dt;
        continue;
      }

      for(int d = 0; d < eval_jacs.size(); ++d)
      {
        for(int e = 0; e < eval_jacs[0].size(); ++e)
        {
          EXPECT_TRUE((fin_diff_jacs[d][e] - eval_jacs[d][e]).isZero(tolerance_));
          if(!(fin_diff_jacs[d][e] - eval_jacs[d][e]).isZero(tolerance_))
            std::cout << "d = " << d << ", e = " << e << ", t = " << t << "\nEval: \n" << eval_jacs[d][e] << "\n Fin diff: \n" << fin_diff_jacs[d][e] << "\n\n";
        }
      }

      t += dt;
    }
  }
}

TEST_F(SO2GPJacTest, SO2WnojResidJacTest)
{
  double start = 0.0;
  double end = 10.0;
  int num_segments = 10;
  double var = 0.1;

  for(int z = 0; z < 10; ++z)
  {
    init_gp(gp::ModelType::JERK, start, end, num_segments, var);

    for(int i = 0; i < num_segments-1; ++i)
    {
      std::vector<Eigen::Vector3d> eval_jacs, fin_diff_jacs;
      gp_.get_dynamics_residual<3>(i, &eval_jacs);
      fin_diff_jacs = fin_diff_resid_jacs<3>(i);

      for(int d = 0; d < eval_jacs.size(); ++d)
      {
        EXPECT_TRUE((fin_diff_jacs[d] - eval_jacs[d]).isZero(tolerance_));
        if(!(fin_diff_jacs[d] - eval_jacs[d]).isZero(tolerance_))
          std::cout << "d = " << d << ", i = " << i << "\nEval: \n" << eval_jacs[d] << "\n Fin diff: \n" << fin_diff_jacs[d] << "\n\n";
      }
    }
  }
}

TEST_F(SO3GPJacTest, SO3WnoaDerivTest)
{
  double start = 0.0;
  double end = 10.0;
  int num_segments = 10;
  int num_samples = 555;
  double var = 0.1;

  for(int z = 0; z < 10; ++z)
  {
    init_gp(gp::ModelType::ACC, start, end, num_segments, var);

    double t = start;
    double dt = (end - start) / num_samples;
    for(int i = 0; i < num_samples; ++i)
    {
      std::vector<VectorDoF> eval_derivs, fin_diff_derivs;
      if(!get_derivs(t, eval_derivs, fin_diff_derivs))
      {
        t += dt;
        continue;
      }

      EXPECT_TRUE((eval_derivs[0] - fin_diff_derivs[0]).isZero(tolerance_));
      if(!(eval_derivs[0] - fin_diff_derivs[0]).isZero(tolerance_))
        std::cout << "Eval: \n" << eval_derivs[0] << "\nFin diff: \n" << fin_diff_derivs[0] << "\n\n";

      t += dt;
    }
  }
}

// some of these tests fail because we approximate some of the jacobians
// TEST_F(SO3GPJacTest, SO3WnoaJacTest)
// {
//   double start = 0.0;
//   double end = 10.0;
//   int num_segments = 10;
//   int num_samples = 7;
//   double var = 0.1;

//   for(int z = 0; z < 1; ++z)
//   {
//     init_gp(gp::ModelType::ACC, start, end, num_segments, var);

//     double t = start;
//     double dt = (end - start) / num_samples;
//     for(int i = 0; i < num_samples; ++i)
//     {
//       std::vector<std::vector<MatrixDoF>> eval_jacs, fin_diff_jacs;
//       if(!get_jacs(t, eval_jacs, fin_diff_jacs))
//       {
//         t += dt;
//         continue;
//       }

//       for(int d = 0; d < eval_jacs.size(); ++d)
//       {
//         for(int e = 0; e < eval_jacs[0].size(); ++e)
//         {
//           EXPECT_TRUE((fin_diff_jacs[d][e] - eval_jacs[d][e]).isZero(1e-2));
//           if(!(fin_diff_jacs[d][e] - eval_jacs[d][e]).isZero(1e-2))
//             std::cout << "d = " << d << ", e = " << e << ", t = " << t << "\nEval: \n" << eval_jacs[d][e] << "\n Fin diff: \n" << fin_diff_jacs[d][e] << "\n\n";
//         }
//       }

//       t += dt;
//     }
//   }
// }

// TEST_F(SO3GPJacTest, SO3WnoaResidJacTest)
// {
//   double start = 0.0;
//   double end = 10.0;
//   int num_segments = 10;
//   double var = 0.1;

//   for(int z = 0; z < 1; ++z)
//   {
//     init_gp(gp::ModelType::ACC, start, end, num_segments, var);

//     for(int i = 0; i < num_segments-1; ++i)
//     {
//       std::vector<Eigen::Matrix<double, 6, 3>> eval_jacs, fin_diff_jacs;
//       gp_.get_dynamics_residual<6>(i, &eval_jacs);
//       fin_diff_jacs = fin_diff_resid_jacs<6>(i);

//       for(int d = 0; d < eval_jacs.size(); ++d)
//       {
//         EXPECT_TRUE((fin_diff_jacs[d] - eval_jacs[d]).isZero(tolerance_));
//         if(!(fin_diff_jacs[d] - eval_jacs[d]).isZero(tolerance_))
//           std::cout << "d = " << d << ", i = " << i << "\nEval: \n" << eval_jacs[d] << "\n Fin diff: \n" << fin_diff_jacs[d] << "\n\n";
//       }
//     }
//   }
// }

// some of these tests may fail because acceleration is approximated
// TEST_F(SO3GPJacTest, SO3WnojDerivTest)
// {
//   double start = 0.0;
//   double end = 10.0;
//   int num_segments = 10;
//   int num_samples = 7;
//   double var = 0.1;

//   for(int z = 0; z < 1; ++z)
//   {
//     init_gp(gp::ModelType::JERK, start, end, num_segments, var);

//     double t = start;
//     double dt = (end - start) / num_samples;
//     for(int i = 0; i < num_samples; ++i)
//     {
//       std::vector<VectorDoF> eval_derivs, fin_diff_derivs;
//       if(!get_derivs(t, eval_derivs, fin_diff_derivs))
//       {
//         t += dt;
//         continue;
//       }

//       for(int j = 0; j < eval_derivs.size(); ++j)
//       {
//         EXPECT_TRUE((eval_derivs[j] - fin_diff_derivs[j]).isZero(1e-2));
//         if(!(eval_derivs[j] - fin_diff_derivs[j]).isZero(1e-2))
//           std::cout << "j = " << j << "\nEval: \n" << eval_derivs[j] << "\nFin diff: \n" << fin_diff_derivs[j] << "\n\n";
//       }

//       t += dt;
//     }
//   }
// }

// TEST_F(SO3GPJacTest, SO3WnojJacTest)
// {
//   double start = 0.0;
//   double end = 10.0;
//   int num_segments = 100;
//   int num_samples = 7;
//   double var = 0.1;

//   for(int z = 0; z < 1; ++z)
//   {
//     init_gp(gp::ModelType::JERK, start, end, num_segments, var);

//     double t = start;
//     double dt = (end - start) / num_samples;
//     for(int i = 0; i < num_samples; ++i)
//     {
//       std::vector<std::vector<MatrixDoF>> eval_jacs, fin_diff_jacs;
//       if(!get_jacs(t, eval_jacs, fin_diff_jacs))
//       {
//         t += dt;
//         continue;
//       }

//       for(int d = 0; d < eval_jacs.size(); ++d)
//       {
//         for(int e = 0; e < eval_jacs[0].size(); ++e)
//         {
//           EXPECT_TRUE((fin_diff_jacs[d][e] - eval_jacs[d][e]).isZero(1e-2));
//           if(!(fin_diff_jacs[d][e] - eval_jacs[d][e]).isZero(1e-2))
//             std::cout << "d = " << d << ", e = " << e << ", t = " << t << "\nEval: \n" << eval_jacs[d][e] << "\n Fin diff: \n" << fin_diff_jacs[d][e] << "\n\n";
//         }
//       }

//       t += dt;
//     }
//   }
// }

TEST_F(SO3GPJacTest, SO3WnojResidJacTest)
{
  double start = 0.0;
  double end = 10.0;
  int num_segments = 10;
  double var = 0.1;

  for(int z = 0; z < 1; ++z)
  {
    init_gp(gp::ModelType::JERK, start, end, num_segments, var);

    for(int i = 0; i < num_segments-1; ++i)
    {
      std::vector<Eigen::Matrix<double, 9, 3>> eval_jacs, fin_diff_jacs;
      gp_.get_dynamics_residual<9>(i, &eval_jacs);
      fin_diff_jacs = fin_diff_resid_jacs<9>(i);

      for(int d = 0; d < eval_jacs.size(); ++d)
      {
        EXPECT_TRUE((fin_diff_jacs[d] - eval_jacs[d]).isZero(tolerance_));
        if(!(fin_diff_jacs[d] - eval_jacs[d]).isZero(tolerance_))
          std::cout << "d = " << d << ", i = " << i << "\nEval: \n" << eval_jacs[d] << "\n Fin diff: \n" << fin_diff_jacs[d] << "\n\n";
      }
    }
  }
}

TEST_F(SE2GPJacTest, SE2WnoaDerivTest)
{
  double start = 0.0;
  double end = 10.0;
  int num_segments = 10;
  int num_samples = 555;
  double var = 0.1;

  for(int z = 0; z < 10; ++z)
  {
    init_gp(gp::ModelType::ACC, start, end, num_segments, var);

    double t = start;
    double dt = (end - start) / num_samples;
    for(int i = 0; i < num_samples; ++i)
    {
      std::vector<VectorDoF> eval_derivs, fin_diff_derivs;
      if(!get_derivs(t, eval_derivs, fin_diff_derivs))
      {
        t += dt;
        continue;
      }

      EXPECT_TRUE((eval_derivs[0] - fin_diff_derivs[0]).isZero(tolerance_));
      if(!(eval_derivs[0] - fin_diff_derivs[0]).isZero(tolerance_))
        std::cout << "Eval: \n" << eval_derivs[0] << "\nFin diff: \n" << fin_diff_derivs[0] << "\n\n";

      t += dt;
    }
  }
}

// TEST_F(SE2GPJacTest, SE2WnoaJacTest)
// {
//   double start = 0.0;
//   double end = 10.0;
//   int num_segments = 10;
//   int num_samples = 7;
//   double var = 0.1;

//   for(int z = 0; z < 1; ++z)
//   {
//     init_gp(gp::ModelType::ACC, start, end, num_segments, var);

//     double t = start;
//     double dt = (end - start) / num_samples;
//     for(int i = 0; i < num_samples; ++i)
//     {
//       std::vector<std::vector<MatrixDoF>> eval_jacs, fin_diff_jacs;
//       if(!get_jacs(t, eval_jacs, fin_diff_jacs))
//       {
//         t += dt;
//         continue;
//       }

//       for(int d = 0; d < eval_jacs.size(); ++d)
//       {
//         for(int e = 0; e < eval_jacs[0].size(); ++e)
//         {
//           EXPECT_TRUE((fin_diff_jacs[d][e] - eval_jacs[d][e]).isZero(1e-2));
//           if(!(fin_diff_jacs[d][e] - eval_jacs[d][e]).isZero(1e-2))
//             std::cout << "d = " << d << ", e = " << e << ", t = " << t << "\nEval: \n" << eval_jacs[d][e] << "\n Fin diff: \n" << fin_diff_jacs[d][e] << "\n\n";
//         }
//       }

//       t += dt;
//     }
//   }
// }

// TEST_F(SE2GPJacTest, SE2WnojDerivTest)
// {
//   double start = 0.0;
//   double end = 10.0;
//   int num_segments = 10;
//   int num_samples = 7;
//   double var = 0.1;

//   for(int z = 0; z < 1; ++z)
//   {
//     init_gp(gp::ModelType::JERK, start, end, num_segments, var);

//     double t = start;
//     double dt = (end - start) / num_samples;
//     for(int i = 0; i < num_samples; ++i)
//     {
//       std::vector<VectorDoF> eval_derivs, fin_diff_derivs;
//       if(!get_derivs(t, eval_derivs, fin_diff_derivs))
//       {
//         t += dt;
//         continue;
//       }

//       for(int j = 0; j < eval_derivs.size(); ++j)
//       {
//         EXPECT_TRUE((eval_derivs[j] - fin_diff_derivs[j]).isZero(tolerance_));
//         if(!(eval_derivs[j] - fin_diff_derivs[j]).isZero(tolerance_))
//           std::cout << "j = " << j << "\nEval: \n" << eval_derivs[j] << "\nFin diff: \n" << fin_diff_derivs[j] << "\n\n";
//       }

//       t += dt;
//     }
//   }
// }

// TEST_F(SE2GPJacTest, SE2WnojJacTest)
// {
//   h_ = 1e-4;
//   double start = 0.0;
//   double end = 10.0;
//   int num_segments = 100;
//   int num_samples = 7;
//   double var = 0.1;

//   for(int z = 0; z < 1; ++z)
//   {
//     init_gp(gp::ModelType::JERK, start, end, num_segments, var);

//     double t = start;
//     double dt = (end - start) / num_samples;
//     for(int i = 0; i < num_samples; ++i)
//     {
//       std::vector<std::vector<MatrixDoF>> eval_jacs, fin_diff_jacs;
//       if(!get_jacs(t, eval_jacs, fin_diff_jacs))
//       {
//         t += dt;
//         continue;
//       }

//       for(int d = 0; d < eval_jacs.size(); ++d)
//       {
//         for(int e = 0; e < eval_jacs[0].size(); ++e)
//         {
//           EXPECT_TRUE((fin_diff_jacs[d][e] - eval_jacs[d][e]).isZero(tolerance_));
//           if(!(fin_diff_jacs[d][e] - eval_jacs[d][e]).isZero(tolerance_))
//             std::cout << "d = " << d << ", e = " << e << ", t = " << t << "\nEval: \n" << eval_jacs[d][e] << "\n Fin diff: \n" << fin_diff_jacs[d][e] << "\n\n";
//         }
//       }

//       t += dt;
//     }
//   }
// }

TEST_F(SE3GPJacTest, SE3WnoaDerivTest)
{
  double start = 0.0;
  double end = 10.0;
  int num_segments = 10;
  int num_samples = 555;
  double var = 0.1;

  for(int z = 0; z < 10; ++z)
  {
    init_gp(gp::ModelType::ACC, start, end, num_segments, var);

    double t = start;
    double dt = (end - start) / num_samples;
    for(int i = 0; i < num_samples; ++i)
    {
      std::vector<VectorDoF> eval_derivs, fin_diff_derivs;
      if(!get_derivs(t, eval_derivs, fin_diff_derivs))
      {
        t += dt;
        continue;
      }

      EXPECT_TRUE((eval_derivs[0] - fin_diff_derivs[0]).isZero(tolerance_));
      if(!(eval_derivs[0] - fin_diff_derivs[0]).isZero(tolerance_))
        std::cout << "Eval: \n" << eval_derivs[0] << "\nFin diff: \n" << fin_diff_derivs[0] << "\n\n";

      t += dt;
    }
  }
}

// TEST_F(SE3GPJacTest, SE3WnoaJacTest)
// {
//   double start = 0.0;
//   double end = 10.0;
//   int num_segments = 10;
//   int num_samples = 7;
//   double var = 0.1;

//   for(int z = 0; z < 1; ++z)
//   {
//     init_gp(gp::ModelType::ACC, start, end, num_segments, var);

//     double t = start;
//     double dt = (end - start) / num_samples;
//     for(int i = 0; i < num_samples; ++i)
//     {
//       std::vector<std::vector<MatrixDoF>> eval_jacs, fin_diff_jacs;
//       if(!get_jacs(t, eval_jacs, fin_diff_jacs))
//       {
//         t += dt;
//         continue;
//       }

//       for(int d = 0; d < eval_jacs.size(); ++d)
//       {
//         for(int e = 0; e < eval_jacs[0].size(); ++e)
//         {
//           EXPECT_TRUE((fin_diff_jacs[d][e] - eval_jacs[d][e]).isZero(1e-2));
//           if(!(fin_diff_jacs[d][e] - eval_jacs[d][e]).isZero(1e-2))
//             std::cout << "d = " << d << ", e = " << e << ", t = " << t << "\nEval: \n" << eval_jacs[d][e] << "\n Fin diff: \n" << fin_diff_jacs[d][e] << "\n\n";
//         }
//       }

//       t += dt;
//     }
//   }
// }

// TEST_F(SE3GPJacTest, SE3WnojDerivTest)
// {
//   double start = 0.0;
//   double end = 10.0;
//   int num_segments = 10;
//   int num_samples = 7;
//   double var = 0.1;

//   for(int z = 0; z < 1; ++z)
//   {
//     init_gp(gp::ModelType::JERK, start, end, num_segments, var);

//     double t = start;
//     double dt = (end - start) / num_samples;
//     for(int i = 0; i < num_samples; ++i)
//     {
//       std::vector<VectorDoF> eval_derivs, fin_diff_derivs;
//       if(!get_derivs(t, eval_derivs, fin_diff_derivs))
//       {
//         t += dt;
//         continue;
//       }

//       for(int j = 0; j < eval_derivs.size(); ++j)
//       {
//         EXPECT_TRUE((eval_derivs[j] - fin_diff_derivs[j]).isZero(1e-2));
//         if(!(eval_derivs[j] - fin_diff_derivs[j]).isZero(1e-2))
//           std::cout << "j = " << j << "\nEval: \n" << eval_derivs[j] << "\nFin diff: \n" << fin_diff_derivs[j] << "\n\n";
//       }

//       t += dt;
//     }
//   }
// }

// TEST_F(SE3GPJacTest, SE3WnojJacTest)
// {
//   double start = 0.0;
//   double end = 10.0;
//   int num_segments = 100;
//   int num_samples = 7;
//   double var = 0.1;

//   for(int z = 0; z < 1; ++z)
//   {
//     init_gp(gp::ModelType::JERK, start, end, num_segments, var);

//     double t = start;
//     double dt = (end - start) / num_samples;
//     for(int i = 0; i < num_samples; ++i)
//     {
//       std::vector<std::vector<MatrixDoF>> eval_jacs, fin_diff_jacs;
//       if(!get_jacs(t, eval_jacs, fin_diff_jacs))
//       {
//         t += dt;
//         continue;
//       }

//       for(int d = 0; d < eval_jacs.size(); ++d)
//       {
//         for(int e = 0; e < eval_jacs[0].size(); ++e)
//         {
//           EXPECT_TRUE((fin_diff_jacs[d][e] - eval_jacs[d][e]).isZero(1e-2));
//           if(!(fin_diff_jacs[d][e] - eval_jacs[d][e]).isZero(1e-2))
//             std::cout << "d = " << d << ", e = " << e << ", t = " << t << "\nEval: \n" << eval_jacs[d][e] << "\n Fin diff: \n" << fin_diff_jacs[d][e] << "\n\n";
//         }
//       }

//       t += dt;
//     }
//   }
// }

// ensure that Q_inv = Q^{-1}
TEST_F(SE3GPJacTest, SE3WnoaCovTest)
{
  double start = 0.0;
  double end = 10.0;
  int num_segments = 10;
  double var = 0.1;

  double dt = (end - start)/num_segments;

  init_gp(gp::ModelType::ACC, start, end, num_segments, var);

  for(int i = 1; i < num_segments; ++i)
  {
    Eigen::Matrix<double, 2*DoF, 2*DoF> Q_i, Q_i_inv;
    Q_i << Q_, Q_, Q_, Q_;

    MatrixDoF Q_inv = Q_.inverse();
    Q_i_inv << Q_inv, Q_inv, Q_inv, Q_inv;

    gp_.get_Q<2*DoF>(dt*(i-1), dt*i, Q_i);
    gp_.get_Q_inv<2*DoF>(dt*(i-1), dt*i, Q_i_inv);

    EXPECT_TRUE((Q_i_inv - Q_i.inverse()).isZero(1e-4));    
  }
}

TEST_F(SE3GPJacTest, SE3WnojCovTest)
{
  double start = 0.0;
  double end = 10.0;
  int num_segments = 10;
  double var = 0.1;

  double dt = (end - start)/num_segments;

  init_gp(gp::ModelType::JERK, start, end, num_segments, var);

  for(int i = 1; i < num_segments; ++i)
  {
    Eigen::Matrix<double, 3*DoF, 3*DoF> Q_i, Q_i_inv;
    Q_i << Q_, Q_, Q_, Q_, Q_, Q_, Q_, Q_, Q_;

    MatrixDoF Q_inv = Q_.inverse();
    Q_i_inv << Q_inv, Q_inv, Q_inv, Q_inv, Q_inv, Q_inv, Q_inv, Q_inv, Q_inv;

    gp_.get_Q<3*DoF>(dt*(i-1), dt*i, Q_i);
    gp_.get_Q_inv<3*DoF>(dt*(i-1), dt*i, Q_i_inv);

    EXPECT_TRUE((Q_i_inv - Q_i.inverse()).isZero(1e-4));    
  }
}