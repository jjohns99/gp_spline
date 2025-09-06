#include "utils/sinusoid.h"

namespace utils
{

Sinusoid::Sinusoid(double frequency, double amplitude, double phase, double offset) :
  omega_{2.0 * M_PI * frequency}, amp_{amplitude}, phi_{phase}, o_{offset}
{}

double Sinusoid::sample(double t)
{
  return amp_ * std::sin(omega_ * t + phi_) + o_;
}

double Sinusoid::sample_d1(double t)
{
  return amp_ * omega_ * std::cos(omega_ * t + phi_);
}

// derivatives is a vector of desired derivatives to evaluate
std::vector<double> Sinusoid::sample(double t, std::vector<int> derivatives)
{
  std::vector<double> ret;

  double interior = omega_*t + phi_;
  for(int const& d : derivatives)
  {
    double val =  amp_ * std::pow(omega_, static_cast<double>(d));
    if(!(d % 2)) val *= std::sin(interior);
    else val *= std::cos(interior);

    if(static_cast<int>(static_cast<double>(d) / 2.0) % 2) val *= -1.0;

    if(d == 0) val += o_;

    ret.push_back(val);
  }

  return ret;
}

} // namespace utils