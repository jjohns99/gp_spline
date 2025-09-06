#include <vector>
#include <chrono>
#include <memory>
#include <iostream>
#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "utils/multivariate_gaussian.hpp"
#include "gp/rn_gp.hpp"

namespace gp
{

template <int N>
class RnGPResidTest : public testing::Test
{
protected:
  static const int DoF = N;
  using VectorDoF = Eigen::Matrix<double, DoF, 1>;
  using MatrixDoF = Eigen::Matrix<double, DoF, DoF>;
  using GP = RnGP<VectorDoF, DoF>;

public:
  RnGPResidTest() : h_{1e-5}, tolerance_{1e-2}, noise_dist_{nullptr}
  {}

  void init_gp(ModelType type, double start, double end, int segments, double var)
  {
    Q_ = var / (end - start) * MatrixDoF::Identity();
    noise_dist_.reset(new utils::MultivariateGaussian<DoF>(VectorDoF::Zero(), Q_));

    type_ = type;

    std::shared_ptr<std::vector<VectorDoF>> p(new std::vector<VectorDoF>);
    std::shared_ptr<std::vector<VectorDoF>> v(new std::vector<VectorDoF>);
    std::shared_ptr<std::vector<VectorDoF>> a(new std::vector<VectorDoF>);

    // sample initial conditions
    t_vec_.push_back(start);
    p->push_back(5.0 * VectorDoF::Random());
    v->push_back(1.0 * VectorDoF::Random());
    if(type == ModelType::JERK) a->push_back(0.5 * VectorDoF::Random());

    double dt = (end - start)/segments;
    double ddt = dt/10;
    for(int i = 0; i < segments; ++i)
    {
      VectorDoF pn = p->back();
      VectorDoF vn = v->back();
      VectorDoF an;
      if(type == ModelType::JERK) an = a->back();
      
      // integrate SDE over the segment
      for(int j = 0; j < 10; ++j)
      {
        VectorDoF w = noise_dist_->sample();
        if(type == ModelType::ACC) an = w;
        else an = an + ddt * w;
        vn = vn + ddt * an;
        pn = pn + ddt * vn;
      }

      t_vec_.push_back(t_vec_.back() + dt);
      p->push_back(pn);
      v->push_back(vn);
      if(type == ModelType::JERK) a->push_back(an);
    }

    gp_ = GP(type, Q_);
    gp_.init_est_params(t_vec_, p, v, type == ModelType::JERK ? a : nullptr);

    p_ = *p;
    v_ = *v;
    a_ = *a;
  }

  // calculate dynamics residual jacobians between t_i and t_i1 (jacobian should not vary with p, v, a)
  // vector has jacs of residual wrt {p_i, p_{i+1}, v_i, v_{i+1}, a_i, a_{i+1}} (as applicable)
  template <int S>
  std::vector<Eigen::Matrix<double, S, DoF>> fin_diff_resid_jacs(int i)
  {
    std::vector<Eigen::Matrix<double, S, DoF>> ret;

    Eigen::Matrix<double, S, 1> r = gp_.template get_dynamics_residual<S>(i);

    // compute jacs wrt p_i, p_{i+1}
    for(int j = i; j <= i + 1; ++j)
    {
      Eigen::Matrix<double, S, DoF> r_pj_jac = Eigen::Matrix<double, S, DoF>::Zero();
      for(int d = 0; d < DoF; ++d)
      {
        VectorDoF tau = VectorDoF::Zero();
        tau(d) = h_;
        assert(gp_.modify_p(j, p_[j] + tau));

        Eigen::Matrix<double, S, 1> r_pert = gp_.template get_dynamics_residual<S>(i);

        r_pj_jac.col(d) = (r_pert - r)/h_;

        assert(gp_.modify_p(j, p_[j]));
      }

      ret.push_back(r_pj_jac);
    }

    // compute jacs wrt v_i, v_{i+1}
    for(int j = i; j <= i + 1; ++j)
    {
      Eigen::Matrix<double, S, DoF> r_vj_jac = Eigen::Matrix<double, S, DoF>::Zero();
      for(int d = 0; d < DoF; ++d)
      {
        VectorDoF tau = VectorDoF::Zero();
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
          VectorDoF tau = VectorDoF::Zero();
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

protected:
  GP gp_;

  std::vector<double> t_vec_;
  std::vector<VectorDoF> p_;
  std::vector<VectorDoF> v_;
  std::vector<VectorDoF> a_;

  ModelType type_;
  MatrixDoF Q_;

  std::unique_ptr<utils::MultivariateGaussian<DoF>> noise_dist_;

  double h_; // perturbation value
  double tolerance_;
};

} // namespace gp

typedef gp::RnGPResidTest<2> R2GPResidTest;
typedef gp::RnGPResidTest<3> R3GPResidTest;

TEST_F(R2GPResidTest, R2GPResidWnoaTest)
{
  double start = 0.0;
  double end = 10.0;
  int num_segments = 10;
  double var = 0.1;

  init_gp(gp::ModelType::ACC, start, end, num_segments, var);

  for(int i = 0; i < p_.size()-1; ++i)
  {
    std::vector<Eigen::Matrix<double, 2*DoF, DoF>> fin_diff_jacs = fin_diff_resid_jacs<2*DoF>(i);
    std::vector<Eigen::Matrix<double, 2*DoF, DoF>> eval_jacs;
    gp_.get_dynamics_residual<2*DoF>(i, &eval_jacs);

    for(int d = 0; d < eval_jacs.size(); ++d)
    {
      EXPECT_TRUE((fin_diff_jacs[d] - eval_jacs[d]).isZero(tolerance_));
      if(!(fin_diff_jacs[d] - eval_jacs[d]).isZero(tolerance_))
        std::cout << "d = " << d << ", i = " << i << "\nEval: \n" << eval_jacs[d] << "\n Fin diff: \n" << fin_diff_jacs[d] << "\n\n";
    }
  }
}

TEST_F(R2GPResidTest, R2GPResidWnojTest)
{
  double start = 0.0;
  double end = 10.0;
  int num_segments = 10;
  double var = 0.1;

  init_gp(gp::ModelType::JERK, start, end, num_segments, var);

  for(int i = 0; i < p_.size()-1; ++i)
  {
    std::vector<Eigen::Matrix<double, 3*DoF, DoF>> fin_diff_jacs = fin_diff_resid_jacs<3*DoF>(i);
    std::vector<Eigen::Matrix<double, 3*DoF, DoF>> eval_jacs;
    gp_.get_dynamics_residual<3*DoF>(i, &eval_jacs);

    for(int d = 0; d < eval_jacs.size(); ++d)
    {
      EXPECT_TRUE((fin_diff_jacs[d] - eval_jacs[d]).isZero(tolerance_));
      if(!(fin_diff_jacs[d] - eval_jacs[d]).isZero(tolerance_))
        std::cout << "d = " << d << ", i = " << i << "\nEval: \n" << eval_jacs[d] << "\n Fin diff: \n" << fin_diff_jacs[d] << "\n\n";
    }
  }
}

TEST_F(R3GPResidTest, R3GPResidWnoaTest)
{
  double start = 0.0;
  double end = 10.0;
  int num_segments = 10;
  double var = 0.1;

  init_gp(gp::ModelType::ACC, start, end, num_segments, var);

  for(int i = 0; i < p_.size()-1; ++i)
  {
    std::vector<Eigen::Matrix<double, 2*DoF, DoF>> fin_diff_jacs = fin_diff_resid_jacs<2*DoF>(i);
    std::vector<Eigen::Matrix<double, 2*DoF, DoF>> eval_jacs;
    gp_.get_dynamics_residual<2*DoF>(i, &eval_jacs);

    for(int d = 0; d < eval_jacs.size(); ++d)
    {
      EXPECT_TRUE((fin_diff_jacs[d] - eval_jacs[d]).isZero(tolerance_));
      if(!(fin_diff_jacs[d] - eval_jacs[d]).isZero(tolerance_))
        std::cout << "d = " << d << ", i = " << i << "\nEval: \n" << eval_jacs[d] << "\n Fin diff: \n" << fin_diff_jacs[d] << "\n\n";
    }
  }
}

TEST_F(R3GPResidTest, R3GPResidWnojTest)
{
  double start = 0.0;
  double end = 10.0;
  int num_segments = 10;
  double var = 0.1;

  init_gp(gp::ModelType::JERK, start, end, num_segments, var);

  for(int i = 0; i < p_.size()-1; ++i)
  {
    std::vector<Eigen::Matrix<double, 3*DoF, DoF>> fin_diff_jacs = fin_diff_resid_jacs<3*DoF>(i);
    std::vector<Eigen::Matrix<double, 3*DoF, DoF>> eval_jacs;
    gp_.get_dynamics_residual<3*DoF>(i, &eval_jacs);

    for(int d = 0; d < eval_jacs.size(); ++d)
    {
      EXPECT_TRUE((fin_diff_jacs[d] - eval_jacs[d]).isZero(tolerance_));
      if(!(fin_diff_jacs[d] - eval_jacs[d]).isZero(tolerance_))
        std::cout << "d = " << d << ", i = " << i << "\nEval: \n" << eval_jacs[d] << "\n Fin diff: \n" << fin_diff_jacs[d] << "\n\n";
    }
  }
}