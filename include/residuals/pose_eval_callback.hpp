#pragma once

#include <vector>
#include <memory>
#include <ceres/ceres.h>

template <typename EstT>
class PoseEvalCallback : public ceres::EvaluationCallback
{
public:
  PoseEvalCallback(std::vector<double> t_pre_eval, std::shared_ptr<EstT> est) :
    t_pre_eval_{t_pre_eval}, est_{est}
  {}

  void PrepareForEvaluation(bool evaluate_jacobians, bool new_evaluation_point) override
  {
    // if(new_evaluation_point)
    // {
    if(evaluate_jacobians)
      est_->pre_eval(t_pre_eval_, true, false, false, true, false, false);
    else
      est_->pre_eval(t_pre_eval_, true, false, false, false, false, false);
    // } 
  }

private:
  std::vector<double> t_pre_eval_;
  std::shared_ptr<EstT> est_;
};