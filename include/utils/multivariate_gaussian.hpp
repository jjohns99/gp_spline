#pragma once

#include <random>
#include <time.h>

#include <Eigen/Dense>

namespace utils
{

template <int N>
class MultivariateGaussian
{
public:
  MultivariateGaussian(Eigen::Matrix<double, N, 1> mean, Eigen::Matrix<double, N, N> cov, bool seed_rand = true) :
    mean_{mean}, RD_{}, RNG_{seed_rand ? RD_() : static_cast<int>(N/cov(0,0))}, SND_{0,1}
  {
    Eigen::LLT<Eigen::Matrix<double, N, N>> chol(cov);
    assert(chol.info() == Eigen::Success); // will fail if cov isn't positive definite
    L_ = chol.matrixL();
  }

  Eigen::Matrix<double, N, 1> sample()
  {
    Eigen::Matrix<double, N, 1> samp_standard;
    for(int i = 0; i < N; ++ i) samp_standard(i) = SND_(RNG_);

    return mean_ + L_ * samp_standard;
  }
  
  // option to call without object (must already be seeded)
  static Eigen::Matrix<double, N, 1> sample(Eigen::Matrix<double, N, 1> mean, Eigen::Matrix<double, N, N> cov)
  {
    std::random_device RD;
    std::mt19937 RNG(RD);
    std::normal_distribution<> SND(0,1);

    Eigen::Matrix<double, N, 1> samp_standard;
    for(int i = 0; i < N; ++ i) samp_standard(i) = SND(RNG);

    Eigen::LLT<Eigen::Matrix<double, N, N>> chol(cov);
    assert(chol.info() == Eigen::Success); // will fail if cov isn't positive definite

    return mean + chol.matrixL() * samp_standard;
  }

private:
  std::random_device RD_;
  std::mt19937 RNG_;
  std::normal_distribution<> SND_;

  Eigen::Matrix<double, N, 1> mean_;
  Eigen::Matrix<double, N, N> L_;
};

} // namespace utils