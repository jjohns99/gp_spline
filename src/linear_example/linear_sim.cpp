#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include <boost/filesystem.hpp>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>

#include "gp/rn_gp.hpp"
#include "spline/rn_spline.hpp"
#include "linear_example/gp_lin_est.hpp"
#include "linear_example/spline_lin_est.hpp"
#include "estimator/rn_spline_motion_prior.hpp"
#include "utils/multivariate_gaussian.hpp"
#include "utils/sinusoid.h"

using vec2 = Eigen::Vector2d;
using mat2 = Eigen::Matrix2d;

double compute_average(std::vector<double> v)
{
  double sum = 0.0;
  for(double const& d : v) sum += d;
  return sum/v.size();
}

template <typename T>
T compute_median(std::vector<T> vec)
{
  int N = vec.size();
  std::nth_element(vec.begin(), vec.begin() + N/2, vec.end());
  if(N % 2)
    return vec[N/2];
  else
  {
    std::nth_element(vec.begin(), vec.begin() + N/2 - 1, vec.end());
    return (vec[N/2] + vec[N/2 - 1])/2.0;
  }
}

int main()
{
  srand((unsigned)time(NULL));

  // load parameters
  YAML::Node yaml = YAML::LoadFile("../params/linear_sim.yaml");

  // trajectory initial conditions 
  std::vector<double> x0_vec = yaml["x0"].as<std::vector<double>>();
  std::vector<double> v0_vec = yaml["v0"].as<std::vector<double>>();
  std::vector<double> a0_vec = yaml["a0"].as<std::vector<double>>();
  vec2 x0(x0_vec.data());
  vec2 v0(v0_vec.data());
  vec2 a0(a0_vec.data());

  // model type
  gp::ModelType model_type = static_cast<gp::ModelType>(yaml["model_type"].as<int>());

  // simulated trajectory type
  int sim_traj_type = yaml["sim_traj_type"].as<int>();

  // sinusoidal trajectory params (if used)
  double x_amp = yaml["x_amp"].as<double>();
  double x_freq = yaml["x_freq"].as<double>();
  double x_phase = yaml["x_phase"].as<double>();
  double x_offset = yaml["x_offset"].as<double>();
  double y_amp = yaml["y_amp"].as<double>();
  double y_freq = yaml["y_freq"].as<double>();
  double y_phase = yaml["y_phase"].as<double>();
  double y_offset = yaml["y_offset"].as<double>();

  // trajectory noise
  double x_stdev = yaml["x_stdev"].as<double>();
  double y_stdev = yaml["y_stdev"].as<double>();
  mat2 Q;
  Q << x_stdev * x_stdev, 0.0, 0.0, y_stdev * y_stdev;
  
  // measurements
  double pos_meas_period = 1.0 / yaml["pos_meas_freq"].as<double>();
  double pos_meas_x_stdev = yaml["pos_meas_x_stdev"].as<double>();
  double pos_meas_y_stdev = yaml["pos_meas_y_stdev"].as<double>();
  mat2 R;
  R << std::pow(pos_meas_x_stdev, 2.0), 0.0, 0.0, std::pow(pos_meas_y_stdev, 2.0);

  // sim timing
  double sim_end_time = yaml["sim_end_time"].as<double>();
  double sim_time_step = yaml["sim_time_step"].as<double>();

  // monte carlo
  int num_monte_carlo_runs = yaml["num_monte_carlo_runs"].as<int>();

  // gp estimator
  int gp_n_const = yaml["gp_n"].as<int>();
  bool gp_use_motion_priors = yaml["gp_use_motion_priors"].as<bool>();

  // gp n sweep
  bool sweep_params = yaml["sweep_params"].as<bool>();
  int gp_n_start = yaml["gp_n_start"].as<int>();
  int gp_n_end = yaml["gp_n_end"].as<int>();

  // spline estimator
  int spline_k = yaml["spline_k"].as<int>();
  double spline_dt_const = yaml["spline_dt"].as<double>();
  bool spline_use_motion_priors = yaml["spline_use_motion_priors"].as<bool>();
  double spline_motion_prior_dt = yaml["spline_motion_prior_dt"].as<double>();

  // spline dt sweep
  double spline_dt_start = yaml["spline_dt_start"].as<double>();
  double spline_dt_end = yaml["spline_dt_end"].as<double>();
  bool adjust_spline_motion_prior_dt = yaml["adjust_spline_motion_prior_dt"].as<bool>();

  // logging
  bool log_results = yaml["log_results"].as<bool>();
  std::string log_directory = yaml["log_directory"].as<std::string>();
  bool log_monte_carlo = yaml["log_monte_carlo"].as<bool>();
  bool log_mc_sweep = yaml["log_mc_sweep"].as<bool>();
  bool print_est_prog = yaml["print_est_prog"].as<bool>();
  bool print_mc_run = yaml["print_mc_run"].as<bool>();

  if(!sweep_params)
  {
    gp_n_start = gp_n_const;
    gp_n_end = gp_n_const;
    spline_dt_start = spline_dt_const;
    spline_dt_end = spline_dt_const;
  }

  int gp_n = gp_n_start;
  double spline_dt = spline_dt_start;

  std::ofstream gp_mc_sweep_file, spline_mc_sweep_file;
  if(log_mc_sweep)
  {
    std::string mc_directory = log_directory + "mc_sweep/";
    boost::filesystem::path dir(mc_directory);
    if(!boost::filesystem::exists(dir))
    {
      std::cout << "Creating folder " << mc_directory << "\n";
      if(!boost::filesystem::create_directory(mc_directory))
        std::cout << "Couldn't create the folder!\n";
    }

    std::stringstream gp_filename, spline_filename;
    gp_filename.precision(3);
    spline_filename.precision(3);
    int traj_num = (sim_traj_type) == 0 ? static_cast<int>(model_type) : 2;
    gp_filename << std::fixed << "gp_" << traj_num << "_" << static_cast<int>(model_type) << "_" << gp_use_motion_priors << "_sweep.txt";
    spline_filename << std::fixed << "spline_" << traj_num << "_" << static_cast<int>(model_type) << "_" << spline_k << "_" << spline_use_motion_priors << "_sweep.txt";

    gp_mc_sweep_file = std::ofstream(mc_directory + gp_filename.str());
    spline_mc_sweep_file = std::ofstream(mc_directory + spline_filename.str());
  }

  while(1)
  {
    if(adjust_spline_motion_prior_dt)
      spline_motion_prior_dt = (model_type == gp::ModelType::JERK) ? 3 * spline_dt : 2 * spline_dt;

    std::cout << "\nGP n: " << gp_n << "\n";
    std::cout << "Spline dt: " << spline_dt << ", spline mp dt: " << spline_motion_prior_dt << "\n\n";
    // mc run results
    std::vector<double> mc_spline_x_rmse;
    std::vector<double> mc_spline_v_rmse;
    std::vector<double> mc_spline_solve_time;
    std::vector<int> mc_spline_num_params;
    std::vector<double> mc_gp_x_rmse;
    std::vector<double> mc_gp_v_rmse;
    std::vector<double> mc_gp_solve_time;
    std::vector<int> mc_gp_num_params;
    std::vector<double> mc_gp_pos_cost;
    std::vector<double> mc_gp_mp_cost;
    std::vector<double> mc_spline_pos_cost;
    std::vector<double> mc_spline_mp_cost;

    // monte carlo runs
    bool first_run = true;
    for(int mm = 0; mm < num_monte_carlo_runs; ++mm)
    {

      // trajectory data
      std::vector<double> t;
      std::vector<vec2> x_true, v_true, a_true;

      utils::MultivariateGaussian<2> traj_noise_dist(vec2::Zero(), Q);
      utils::MultivariateGaussian<2> meas_noise_dist(vec2::Zero(), R);

      utils::Sinusoid x_sin(x_freq, x_amp, x_phase*M_PI/180.0, x_offset);
      utils::Sinusoid y_sin(y_freq, y_amp, y_phase*M_PI/180.0, y_offset);

      std::shared_ptr<std::vector<std::pair<double, vec2>>> z_pos(new std::vector<std::pair<double, vec2>>);

      double last_pos_meas = -10.0;

      bool init = false;
      for(double sim_time = 0.0; sim_time <= sim_end_time; sim_time += sim_time_step)
      {
        // propagate state
        vec2 x;
        vec2 v;
        vec2 a; 
        if(sim_traj_type == 0)
        {
          if(!init)
          {
            x = x0;
            v = v0;
            a = a0;

            init = true;
          }
          else
          {
            a = (model_type == gp::ModelType::JERK) ? (a_true.back() + traj_noise_dist.sample() * sim_time_step) : traj_noise_dist.sample();
            v = v_true.back() + a * sim_time_step;
            x = x_true.back() + v * sim_time_step;
          }
        }
        else
        {
          std::vector<double> xd = x_sin.sample(sim_time, {0,1,2});
          std::vector<double> yd = y_sin.sample(sim_time, {0,1,2});
          x << xd[0], yd[0];
          v << xd[1], yd[1];
          a << xd[2], yd[2];
        }

        // simulate measurement
        if(sim_time - last_pos_meas > pos_meas_period)
        {
          z_pos->push_back(std::make_pair(sim_time, x + meas_noise_dist.sample()));
          last_pos_meas = sim_time - 1e-6;
        }

        t.push_back(sim_time);
        x_true.push_back(x);
        v_true.push_back(v);
        a_true.push_back(a);
      }

      // compute measurement sqrt info matrix
      Eigen::LLT<mat2> chol(R.inverse());
      mat2 pos_sqrt_W = chol.matrixL().transpose();

      // run gp estimator
      std::unique_ptr<linear_example::GPLinEst<2>> gp_est(new linear_example::GPLinEst<2>(Q, model_type, gp_n, gp_use_motion_priors));

      gp_est->add_pos_measurements(z_pos, pos_sqrt_W);
      ceres::Solver::Summary gp_summary = gp_est->solve(print_est_prog);
      double gp_solve_time = gp_summary.total_time_in_seconds;
      int gp_num_params = gp_summary.num_parameters;
      bool gp_pos_eval, gp_mp_eval;
      double gp_pos_cost, gp_mp_cost;
      gp_pos_eval = gp_est->get_pos_resid_cost(gp_pos_cost);
      gp_mp_eval = gp_est->get_mp_resid_cost(gp_mp_cost);

      // get optimized gp
      std::shared_ptr<gp::RnGP<Eigen::Map<vec2>, 2>> gp = gp_est->get_gp();

      // sample gp estimated trajectory
      std::vector<double> t_gp_opt;
      std::vector<vec2> x_gp_opt, v_gp_opt, a_gp_opt;
      // cut off first and last cut_off_time seconds to avoid wierd anomalies at trajectory ends
      double cut_off_time = 0.25;
      for(double t_samp = cut_off_time; t_samp < sim_end_time - cut_off_time; t_samp += sim_time_step)
      {
        // only look at times where the gp is valid
        if(!gp->is_time_valid(t_samp)) continue;

        vec2 x_samp, v_samp, a_samp;
        gp->eval(t_samp, &x_samp, &v_samp, (model_type == gp::ModelType::JERK) ? &a_samp : nullptr);
        t_gp_opt.push_back(t_samp);
        x_gp_opt.push_back(x_samp);
        v_gp_opt.push_back(v_samp);
        a_gp_opt.push_back((model_type == gp::ModelType::JERK) ? a_samp : vec2::Zero());
      }

      // run spline estimator
      estimator::ModelType spl_model_type = static_cast<estimator::ModelType>(model_type);
      std::unique_ptr<linear_example::SplineLinEst<2>> spline_est(new linear_example::SplineLinEst<2>(spline_k, spline_dt, spline_use_motion_priors, spline_motion_prior_dt, spl_model_type, Q));
      spline_est->add_pos_measurements(z_pos, pos_sqrt_W);
      ceres::Solver::Summary spline_summary = spline_est->solve(print_est_prog);
      double spline_solve_time = spline_summary.total_time_in_seconds;
      int spline_num_params = spline_summary.num_parameters;
      bool spline_pos_eval, spline_mp_eval;
      double spline_pos_cost, spline_mp_cost;
      spline_pos_eval = spline_est->get_pos_resid_cost(spline_pos_cost);
      spline_mp_eval = spline_est->get_mp_resid_cost(spline_mp_cost);

      // get optimized spline
      std::shared_ptr<spline::RnSpline<Eigen::Map<vec2>, 2>> spline = spline_est->get_spline();

      // sample spline estimated trajectory
      std::vector<double> t_spl_opt;
      std::vector<vec2> x_spl_opt, v_spl_opt, a_spl_opt;
      // cut off first and last cut_off_times to avoid wierd anomalies at trajectory ends
      for(double t_samp = cut_off_time; t_samp < sim_end_time - cut_off_time; t_samp += sim_time_step)
      {
        vec2 x_samp, v_samp, a_samp;
        spline->eval(t_samp, &x_samp, &v_samp, &a_samp);
        t_spl_opt.push_back(t_samp);
        x_spl_opt.push_back(x_samp);
        v_spl_opt.push_back(v_samp);
        a_spl_opt.push_back(a_samp);
      }

      // compute errors
      double x_gp_err_sum = 0.0;
      double v_gp_err_sum = 0.0;
      double a_gp_err_sum = 0.0;
      double x_spl_err_sum = 0.0;
      double v_spl_err_sum = 0.0;
      double a_spl_err_sum = 0.0;

      int i_start = cut_off_time / sim_time_step;
      for(int i = 0; i < t_gp_opt.size(); ++i)
      {
        x_gp_err_sum += std::pow((x_gp_opt[i] - x_true[i+i_start]).norm(), 2.0);
        v_gp_err_sum += std::pow((v_gp_opt[i] - v_true[i+i_start]).norm(), 2.0);
        a_gp_err_sum += std::pow((a_gp_opt[i] - a_true[i+i_start]).norm(), 2.0);
      }
      for(int i = 0; i < t_spl_opt.size(); ++i)
      {
        x_spl_err_sum += std::pow((x_spl_opt[i] - x_true[i+i_start]).norm(), 2.0);
        v_spl_err_sum += std::pow((v_spl_opt[i] - v_true[i+i_start]).norm(), 2.0);
        a_spl_err_sum += std::pow((a_spl_opt[i] - a_true[i+i_start]).norm(), 2.0);
      }
      double x_gp_rmse = std::sqrt(x_gp_err_sum)/t_gp_opt.size();
      double v_gp_rmse = std::sqrt(v_gp_err_sum)/t_gp_opt.size();
      double a_gp_rmse = std::sqrt(a_gp_err_sum)/t_gp_opt.size();

      double x_spl_rmse = std::sqrt(x_spl_err_sum)/t_spl_opt.size();
      double v_spl_rmse = std::sqrt(v_spl_err_sum)/t_spl_opt.size();
      double a_spl_rmse = std::sqrt(a_spl_err_sum)/t_spl_opt.size();

      // print rmse
      if(print_mc_run)
      {
        std::cout << "\tMC run: " << mm << "\n";
        std::cout << "\tGP: x: " << x_gp_rmse << ", v: " << v_gp_rmse << ", a: " << a_gp_rmse << ", time: " << gp_solve_time << "\n";
        std::cout << "\tSpline: x: " << x_spl_rmse << ", v: " << v_spl_rmse << ", a: " << a_spl_rmse << ", time: " << spline_solve_time << "\n";
      }

      // skip first monte carlo run. Ceres has to "warm up" on first solve for some reason
      if(num_monte_carlo_runs > 1 && first_run)
      {
        first_run = false;
        mm--;
        continue;
      }

      // stash mc run results
      mc_spline_x_rmse.push_back(x_spl_rmse);
      mc_spline_v_rmse.push_back(v_spl_rmse);
      mc_spline_solve_time.push_back(spline_solve_time);
      mc_spline_num_params.push_back(spline_num_params);
      mc_gp_x_rmse.push_back(x_gp_rmse);
      mc_gp_v_rmse.push_back(v_gp_rmse);
      mc_gp_solve_time.push_back(gp_solve_time);
      mc_gp_num_params.push_back(gp_num_params);
      if(gp_pos_eval) mc_gp_pos_cost.push_back(gp_pos_cost);
      if(gp_mp_eval) mc_gp_mp_cost.push_back(gp_mp_cost);
      if(spline_pos_eval) mc_spline_pos_cost.push_back(spline_pos_cost);
      if(spline_mp_eval) mc_spline_mp_cost.push_back(spline_mp_cost);

      // only log if we aren't doing a monte carlo experiment, otherwise it takes too long
      if(log_results && num_monte_carlo_runs == 1)
      {
        boost::filesystem::path dir(log_directory);
        if(!boost::filesystem::exists(dir))
        {
          std::cout << "Creating folder " << log_directory << "\n";
          if(!boost::filesystem::create_directory(log_directory))
            std::cout << "Couldn't create the folder!\n";
        }

        // truth data
        std::ofstream truth_file(log_directory + "truth.txt");
        for(int i = 0; i < t.size(); ++i)
        {
          truth_file << t[i] << " " << x_true[i](0) << " " << x_true[i](1) << " " << v_true[i](0) << " " << v_true[i](1) << " "
                    << a_true[i](0) << " " << a_true[i](1) << "\n";
        }

        // gp opt data
        std::ofstream gp_opt_file(log_directory + "gp_opt.txt");
        for(int i = 0; i < t_gp_opt.size(); ++i)
        {
          gp_opt_file << t_gp_opt[i] << " " << x_gp_opt[i](0) << " " << x_gp_opt[i](1) << " " << v_gp_opt[i](0) << " " << v_gp_opt[i](1) << " "
                    << a_gp_opt[i](0) << " " << a_gp_opt[i](1) << "\n";
        }

        // spline opt data
        std::ofstream spl_opt_file(log_directory + "spl_opt.txt");
        for(int i = 0; i < t_spl_opt.size(); ++i)
        {
          spl_opt_file << t_spl_opt[i] << " " << x_spl_opt[i](0) << " " << x_spl_opt[i](1) << " " << v_spl_opt[i](0) << " " << v_spl_opt[i](1) << " "
                    << a_spl_opt[i](0) << " " << a_spl_opt[i](1) << "\n";
        }
      }

    }

    std::cout << "\n\tMC medians: \n\tx_spline: " << compute_median(mc_spline_x_rmse) << ", v_spline: " << compute_median(mc_spline_v_rmse) << ", spline time: " << compute_median(mc_spline_solve_time)
                  << ", spline pos cost: " << (mc_spline_pos_cost.size() ? compute_median(mc_spline_pos_cost) : 0) << ", spline mp cost: " << (mc_spline_mp_cost.size() ? compute_median(mc_spline_mp_cost) : 0)
              << "\n\tx_gp: " << compute_median(mc_gp_x_rmse) << ", v_gp: " << compute_median(mc_gp_v_rmse) << ", gp time: " << compute_median(mc_gp_solve_time) 
                  << ", gp pos cost: " << (mc_gp_pos_cost.size() ? compute_median(mc_gp_pos_cost) : 0) << ", gp mp cost: " << (mc_gp_mp_cost.size() ? compute_median(mc_gp_mp_cost) : 0) << "\n";

    if(log_monte_carlo)
    {
      std::string mc_directory = log_directory + "mc_sweep/";
      boost::filesystem::path dir(mc_directory);
      if(!boost::filesystem::exists(dir))
      {
        std::cout << "Creating folder " << mc_directory << "\n";
        if(!boost::filesystem::create_directory(mc_directory))
          std::cout << "Couldn't create the folder!\n";
      }

      std::stringstream gp_filename, spline_filename;
      gp_filename.precision(3);
      spline_filename.precision(3);
      int traj_num = (sim_traj_type) == 0 ? static_cast<int>(model_type) : 2;
      gp_filename << std::fixed << "gp_" << traj_num << "_" << static_cast<int>(model_type) << "_" << gp_n << "_" << gp_use_motion_priors << ".txt";
      spline_filename << std::fixed << "spline_" << traj_num << "_" << static_cast<int>(model_type) << "_" << spline_k << "_" << spline_dt << "_" << spline_use_motion_priors << "_" << spline_motion_prior_dt << ".txt";

      std::ofstream gp_mc_file(mc_directory + gp_filename.str());
      std::ofstream spline_mc_file(mc_directory + spline_filename.str());
      for(int i = 0; i < num_monte_carlo_runs; ++i) gp_mc_file << mc_gp_x_rmse[i] << " " << mc_gp_v_rmse[i] << " " << mc_gp_solve_time[i] << " " << mc_gp_num_params[i] << "\n";
      for(int i = 0; i < num_monte_carlo_runs; ++i) spline_mc_file << mc_spline_x_rmse[i] << " " << mc_spline_v_rmse[i] << " " << mc_spline_solve_time[i] << " " << mc_spline_num_params[i] << "\n";
    }

    if(log_mc_sweep)
    {
      gp_mc_sweep_file << compute_median(mc_gp_x_rmse) << " " << compute_median(mc_gp_v_rmse) << " " << compute_median(mc_gp_solve_time) << " " << mc_gp_num_params.back() << " " << gp_n << " ";
      if(mc_gp_pos_cost.size()) gp_mc_sweep_file << compute_median(mc_gp_pos_cost) << " ";
      if(mc_gp_mp_cost.size()) gp_mc_sweep_file << compute_median(mc_gp_mp_cost) << " ";
      gp_mc_sweep_file << "\n";

      spline_mc_sweep_file << compute_median(mc_spline_x_rmse) << " " << compute_median(mc_spline_v_rmse) << " " << compute_median(mc_spline_solve_time) << " " << mc_spline_num_params.back() << " " << spline_dt << " ";
      if(mc_spline_pos_cost.size()) spline_mc_sweep_file << compute_median(mc_spline_pos_cost) << " ";
      if(mc_spline_mp_cost.size()) spline_mc_sweep_file << compute_median(mc_spline_mp_cost) << " ";
      spline_mc_sweep_file << "\n";
    }

    // terminate when we've run on the end value for both
    if (gp_n >= gp_n_end && spline_dt >= spline_dt_end) break;

    // apply a sort of logarithmic sweep (20 points per decade)
    int gp_n_step;
    double spline_dt_step;
    if(gp_n < 20)
      gp_n_step = 1;
    else if(gp_n < 200)
      gp_n_step = 10;
    else if(gp_n < 2000)
      gp_n_step = 100;
    else
      gp_n_step = 1000;

    if(spline_dt < 0.2)
      spline_dt_step = 0.01;
    else if(spline_dt < 2.0)
      spline_dt_step = 0.1;
    else if(spline_dt < 20.0)
      spline_dt_step = 0.5;
    else
      spline_dt_step = 10.0;

    gp_n = std::min(gp_n_end, gp_n + gp_n_step);
    spline_dt = std::min(spline_dt_end, spline_dt + spline_dt_step);
  }
}