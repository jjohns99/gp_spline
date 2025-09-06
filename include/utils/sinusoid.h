#pragma once

#include <vector>
#include <cmath>

namespace utils
{

class Sinusoid
{
public:
  Sinusoid(double frequency, double amplitude, double phase, double offset);

  double sample(double t);
  double sample_d1(double t);
  // derivatives is a vector of desired derivatives to evaluate
  std::vector<double> sample(double t, std::vector<int> derivatives);

private:
  double omega_;
  double amp_;
  double phi_;
  double o_;
};

} // namespace utils