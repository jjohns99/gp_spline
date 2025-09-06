#include "utils/line.h"

namespace utils
{

Line::Line(double slope, double offset) :
  slope_{slope}, offset_{offset}, angle_{false}
{}

Line::Line(double slope, double offset, bool angle) :
  slope_{slope}, offset_{offset}, angle_{angle}
{}

double Line::sample(double t)
{
  return angle_ ? std::fmod(slope_ * t + offset_, 2.0 * M_PI) : slope_ * t + offset_;
}

double Line::sample_d1(double t)
{
  return slope_;
}

std::vector<double> Line::sample(double t, std::vector<int> derivatives)
{
  std::vector<double> ret;
  for(int const& d : derivatives)
  {
    if(d == 0) ret.push_back(sample(t));
    else if(d == 1) ret.push_back(slope_);
    else ret.push_back(0.0);
  }
  return ret;
}

} // namespace utils