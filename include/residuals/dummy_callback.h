#pragma once

#include <ceres/ceres.h>

namespace estimator
{

// ceres doesn't write to original parameter memory locations during optimization unless
// the problem has a callback. If no other callback is in use, add this dummy callback to
// the problem to ensure the spline uses the updated parameter values
class DummyCallback : public ceres::EvaluationCallback
{
public:
  DummyCallback()
  {}

  void PrepareForEvaluation(bool evaluate_jacobians, bool new_evaluation_point) override
  {}
};

} // namespace estimator