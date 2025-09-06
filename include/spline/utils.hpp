#pragma once

#include <cmath>
#include <Eigen/Dense>

namespace spline
{

namespace utils
{

namespace internal
{

inline int fact(int n)
{
  return (n==0) || (n==1) ? 1 : n * fact(n-1);
}

inline double bin_coef(int k, int s)
{
  return fact(k) / (fact(s) * fact(k-s));
}

inline Eigen::MatrixXd compute_M(int k)
{
  Eigen::MatrixXd M(k,k);
  for(int s = 0; s < k; ++s)
  {
    for(int n = 0; n < k; ++n)
    {
      double sum = 0.0;
      for(int l = s; l < k; ++l)
        sum += std::pow(-1,l-s) * bin_coef(k,l-s) * std::pow(k-1-l,k-1-n);
      M(s,n) = bin_coef(k-1,n)/fact(k-1) * sum;
    }
  }
  return M;
}

inline Eigen::MatrixXd compute_C0(int k)
{
  Eigen::MatrixXd M = compute_M(k);
  Eigen::MatrixXd C0(k,k);
  for(int j = 0; j < k; ++j)
  {
    for(int n = 0; n < k; ++n)
    {
      double sum = 0;
      for(int s = j; s < k; ++s)
        sum += M(s,n);
      C0(j,n) = sum;
    }
  }
  return C0;
}

} // namespace internal

struct TimeRange
{
  double start;
  double end;

  inline bool valid(double t)
  {
    return (t >= start && t <= end);
  }
};

inline void compute_C(std::vector<Eigen::MatrixXd>& C, int k, double dt, int deg)
{
  C.push_back(internal::compute_C0(k));

  Eigen::MatrixXd D(k, k);
  D.setZero();
  for(int j = 1; j < k; ++j)
    D(j, j-1) = static_cast<double>(j)/dt;

  for(int i = 1; i <= deg; ++i)
    C.push_back(C[i-1] * D);
}

} // namespace utils

} // namespace spline