#include <vector>
#include <chrono>
#include <memory>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>

#include "utils/multivariate_gaussian.hpp"
#include "lie_groups/so2.hpp"
#include "lie_groups/so3.hpp"
#include "lie_groups/se2.hpp"
#include "lie_groups/se3.hpp"
#include "gp/lie_gp.hpp"

using G = lie_groups::SO2d;
static const int DoF = G::DoF;
using VectorDoF = Eigen::Matrix<double, DoF, 1>;
using MatrixDoF = Eigen::Matrix<double, DoF, DoF>;

int main()
{
  double start = 0.0;
  double end = 10.0;
  int segments = 10;
  double var = 0.05;
  gp::ModelType type = gp::ModelType::JERK;

  int num_samples = 1000;

  MatrixDoF Q = var * MatrixDoF::Identity();
  utils::MultivariateGaussian<DoF> noise_dist(VectorDoF::Zero(), Q); 

  std::vector<double> t_vec;
  std::shared_ptr<std::vector<G>> T(new std::vector<G>);
  std::shared_ptr<std::vector<VectorDoF>> v(new std::vector<VectorDoF>);
  std::shared_ptr<std::vector<VectorDoF>> a(new std::vector<VectorDoF>);

  t_vec.push_back(start);
  T->push_back(G::random());
  v->push_back(1.0 * VectorDoF::Random());
  if(type == gp::ModelType::JERK) a->push_back(0.5 * VectorDoF::Random());
  std::cout << "t: " << t_vec.back() << " T: " << T->back().Log() << " v: " << v->back() << "\n";

  double dt = (end - start)/segments;
  double ddt = dt/10;
  for(int i = 0; i < segments; ++i)
  {
    VectorDoF xi_t = VectorDoF::Zero();
    VectorDoF dxi_t, ddxi_t;
    dxi_t = v->back();
    if(type == gp::ModelType::JERK) ddxi_t = a->back();
    
    // integrate SDE over the segment
    for(int j = 0; j < 10; ++j)
    {
      MatrixDoF w = noise_dist.sample();
      if(type == gp::ModelType::JERK) 
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

    t_vec.push_back(t_vec.back() + dt);
    T->push_back(G::Exp(xi_t) * T->back());
    v->push_back(G::Jl(xi_t) * dxi_t);
    if(type == gp::ModelType::JERK) a->push_back(G::Jl(xi_t) * (ddxi_t + 0.5 * G::ad(xi_t) * v->back()));
    std::cout << "t: " << t_vec.back() << " T: " << T->back().Log() << " v: " << v->back() << "\n";
  }

  std::ofstream est_file("../data/test/est.txt");
  for(int i = 0; i < t_vec.size(); ++i)
  {
    est_file << t_vec[i] << " " << T->at(i).Log()(0) << " " << v->at(i)(0);
    if(type == gp::ModelType::JERK) est_file << " " << a->at(i)(0);
    est_file << "\n";
  }

  gp::LieGP<G, VectorDoF> gp(type, Q);
  gp.init_est_params(t_vec, T, v, a);

  std::ofstream out_file("../data/test/interp.txt");

  double t = start;
  double dt_samp = (end - start)/num_samples;
  for(int i = 0; i < num_samples; ++i)
  {
    G T;
    VectorDoF v, a;
    gp.eval(t, &T, &v, type == gp::ModelType::JERK ? &a : nullptr);

    out_file << t << " " << T.Log()(0) << " " << v(0);
    if(type == gp::ModelType::JERK) out_file << " " << a(0);
    out_file << "\n";

    t += dt_samp;
  }

  return 0;
}