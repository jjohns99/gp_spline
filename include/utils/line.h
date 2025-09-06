#pragma once

#include <vector>
#include <cmath>

namespace utils
{

class Line
{
public:
  Line(double slope, double offset);
  Line(double slope, double offset, bool angle);

  double sample(double t);
  double sample_d1(double t);

  std::vector<double> sample(double t, std::vector<int> derivatives);

private:
  double slope_;
  double offset_;
  bool angle_; // if angle, mod by 2 pi
};

} // namespace utils