#include <vector>
#include <memory>
#include <string>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>
#include <chrono>

#include <boost/filesystem.hpp>

#include <Eigen/Dense>

#include "yaml-cpp/yaml.h"

#include "lie_groups/so3.hpp"
#include "lie_groups/se3.hpp"
#include "lie_groups/sonxrn.hpp"
#include "spline/lie_spline.hpp"
#include "spline/sonxrn_spline.hpp"
#include "spline/rn_spline.hpp"
#include "gp/lie_gp.hpp"
#include "gp/sonxrn_gp.hpp"
#include "utils/multivariate_gaussian.hpp"
#include "utils/sinusoid.h"
#include "utils/line.h"
#include "sim/sim_params.h"
#include "estimator/estimator_base.h"
#include "estimator/spline_estimator.hpp"
#include "estimator/gp_se3_estimator.h"
#include "estimator/gp_so3xr3_estimator.h"

#define ESTTYPE 2

using vec3 = Eigen::Vector3d;
using vec6 = Eigen::Matrix<double, 6, 1>;
using mat2 = Eigen::Matrix2d;
using mat3 = Eigen::Matrix3d;
using mat6 = Eigen::Matrix<double, 6, 6>;
using so3 = lie_groups::SO3d;
using se3 = lie_groups::SE3d;
using so3xr3 = lie_groups::SO3xR3d;
#if ESTTYPE == 0
using se3spline = spline::LieSpline<lie_groups::Map<se3>>;
#elif ESTTYPE == 1
using se3spline = spline::SOnxRnSpline<lie_groups::Map<so3xr3>>;
#elif ESTTYPE == 2
using se3gp = gp::LieGP<lie_groups::Map<se3>, Eigen::Map<vec6>>;
using gpest = estimator::GPSE3Estimator;
#else
using se3gp = gp::SOnxRnGP<lie_groups::Map<so3>, Eigen::Map<vec3>, Eigen::Map<vec3>>;
using gpest = estimator::GPSO3xR3Estimator;
#endif
using biasspline = spline::RnSpline<Eigen::Map<vec3>, 3>;

template <typename T>
T find_median(std::vector<T> vec)
{
  std::nth_element(vec.begin(), vec.begin() + vec.size()/2, vec.end());
  return vec[vec.size()/2];
}

template <typename T>
double find_mean(std::vector<T> vec)
{
  T sum = 0;
  for(auto const& v : vec) sum += v;
  return sum/vec.size();
}

int main()
{
  // load parameters
  YAML::Node yaml = YAML::LoadFile("../params/sim.yaml");

  sim::SimParams sim_params;

  // sim trajectory params
  sim_params.sim_traj_type = yaml["sim_traj_type"].as<int>();

  // wnoj trajectory parameters
  std::vector<double> sim_x_init_data = yaml["sim_wnoj_x_init"].as<std::vector<double>>();
  std::vector<double> sim_v_init_data = yaml["sim_wnoj_v_init"].as<std::vector<double>>();
  std::vector<double> sim_a_init_data = yaml["sim_wnoj_a_init"].as<std::vector<double>>();
  sim_params.T_init = se3(vec6(sim_x_init_data.data()));
  sim_params.twist_init = vec6(sim_v_init_data.data());
  sim_params.twist_dot_init = vec6(sim_a_init_data.data());

  // sinusoidal trajectory parameters
  sim_params.x_amplitude = yaml["x_amplitude"].as<double>();
  sim_params.x_period = yaml["x_period"].as<double>();
  sim_params.x_offset = yaml["x_offset"].as<double>();
  sim_params.x_phase = yaml["x_phase"].as<double>();
  sim_params.y_amplitude = yaml["y_amplitude"].as<double>();
  sim_params.y_period = yaml["y_period"].as<double>();
  sim_params.y_offset = yaml["y_offset"].as<double>();
  sim_params.y_phase = yaml["y_phase"].as<double>();
  sim_params.z_amplitude = yaml["z_amplitude"].as<double>();
  sim_params.z_period = yaml["z_period"].as<double>();
  sim_params.z_offset = yaml["z_offset"].as<double>();
  sim_params.z_phase = yaml["z_phase"].as<double>();
  sim_params.roll_amplitude = yaml["roll_amplitude"].as<double>();
  sim_params.roll_period = yaml["roll_period"].as<double>();
  sim_params.roll_offset = yaml["roll_offset"].as<double>();
  sim_params.roll_phase = yaml["roll_phase"].as<double>();
  sim_params.pitch_amplitude = yaml["pitch_amplitude"].as<double>();
  sim_params.pitch_period = yaml["pitch_period"].as<double>();
  sim_params.pitch_offset = yaml["pitch_offset"].as<double>();
  sim_params.pitch_phase = yaml["pitch_phase"].as<double>();
  sim_params.yaw_slope = yaml["yaw_slope"].as<double>();
  sim_params.yaw_offset = yaml["yaw_offset"].as<double>();

  // IMU parameters
  sim_params.imu_frequency = yaml["imu_frequency"].as<double>();
  sim_params.gyro_stdev = yaml["gyro_stdev"].as<double>();
  sim_params.accel_stdev = yaml["accel_stdev"].as<double>();
  sim_params.gyro_walk_stdev = yaml["gyro_walk_stdev"].as<double>();
  sim_params.accel_walk_stdev = yaml["accel_walk_stdev"].as<double>();
  sim_params.g = yaml["g"].as<double>();

  // camera parameters
  sim_params.camera_frequency = yaml["camera_frequency"].as<double>();
  sim_params.pix_stdev = yaml["pix_stdev"].as<double>();
  sim_params.pos_stdev = yaml["pos_stdev"].as<double>();
  sim_params.rot_stdev = yaml["rot_stdev"].as<double>();
  sim_params.sim_constant_cov = yaml["sim_constant_cov"].as<bool>();
  sim_params.est_constant_cov = yaml["est_constant_cov"].as<bool>();
  double fx = yaml["fx"].as<double>();
  double fy = yaml["fy"].as<double>();
  double cx = yaml["cx"].as<double>();
  double cy = yaml["cy"].as<double>();
  sim_params.cam_mat = (mat3() << fx, 0.0, cx,
                                  0.0, fy, cy,
                                  0.0, 0.0, 1.0).finished();
  std::vector<double> r_imu_c_vec = yaml["r_imu_c"].as<std::vector<double>>();
  std::vector<double> p_imu_c_vec = yaml["p_imu_c"].as<std::vector<double>>();
  vec3 r_imu_c = M_PI / 180.0 * (vec3() << r_imu_c_vec[0], r_imu_c_vec[1], r_imu_c_vec[2]).finished();
  vec3 p_imu_c = (vec3() << p_imu_c_vec[0], p_imu_c_vec[1], p_imu_c_vec[2]).finished();
  sim_params.T_imu_c = se3(so3(r_imu_c), p_imu_c);

  // camera field of view
  sim_params.max_range = yaml["max_range"].as<double>();
  sim_params.max_azimuth = M_PI / 180.0 * yaml["max_azimuth"].as<double>();
  sim_params.max_elevation = M_PI / 180.0 * yaml["max_elevation"].as<double>();
  sim_params.assert_facing_camera = yaml["assert_facing_camera"].as<bool>();
  sim_params.apriltag_side_length = yaml["apriltag_side_length"].as<double>();

  // sim parameters
  sim_params.sim_end_time = yaml["sim_end_time"].as<double>();
  sim_params.sim_period = yaml["sim_period"].as<double>();

  // spline estimator parameters
  sim_params.spline_k = yaml["spline_k"].as<int>();
  sim_params.spline_dt = yaml["spline_dt"].as<double>();
  sim_params.spline_son_mp_dt = yaml["spline_son_mp_dt"].as<double>();
  sim_params.spline_rn_mp_dt = yaml["spline_rn_mp_dt"].as<double>();
  sim_params.spline_son_mp_type = yaml["spline_son_mp_type"].as<int>();
  sim_params.spline_rn_mp_type = yaml["spline_rn_mp_type"].as<int>();  

  // gp estimator parameters
  sim_params.q_gp_rot = yaml["q_gp_rot"].as<double>();
  sim_params.q_gp_pos = yaml["q_gp_pos"].as<double>();
  sim_params.n_gp = yaml["n_gp"].as<int>();
  sim_params.gp_son_type = yaml["gp_son_type"].as<int>();
  mat6 Q_gp = mat6::Zero();
  Q_gp.block<3,3>(0,0) = sim_params.q_gp_pos * mat3::Identity();
  Q_gp.block<3,3>(3,3) = sim_params.q_gp_rot * mat3::Identity();

  // initial biases
  std::vector<double> gyro_bias_init_vec = yaml["gyro_bias_init"].as<std::vector<double>>();
  std::vector<double> accel_bias_init_vec = yaml["accel_bias_init"].as<std::vector<double>>();
  vec3 gyro_bias_init = (vec3() << gyro_bias_init_vec[0], gyro_bias_init_vec[1], gyro_bias_init_vec[2]).finished();
  vec3 accel_bias_init = (vec3() << accel_bias_init_vec[0], accel_bias_init_vec[1], accel_bias_init_vec[2]).finished();

  // residuals to use
  sim_params.use_imu = yaml["use_imu"].as<bool>();
  sim_params.use_mp = yaml["use_mp"].as<bool>();

  // max num iterations
  sim_params.max_num_iterations = yaml["max_num_iterations"].as<int>();

  // monte carlo
  sim_params.num_monte_carlo_runs = yaml["num_monte_carlo_runs"].as<int>();
  sim_params.seed_random = yaml["seed_random"].as<bool>();

  if(sim_params.seed_random) srand((unsigned)time(NULL));
  else srand(100);

  // logging parameters (only log if only doing one run)
  sim_params.log_results = yaml["log_results"].as<bool>() && (sim_params.num_monte_carlo_runs == 1);
  sim_params.log_directory = yaml["log_directory"].as<std::string>();
  sim_params.print_est_prog = yaml["print_est_prog"].as<bool>();

  // create motion parameters
  utils::Sinusoid x_motion = utils::Sinusoid(1.0/sim_params.x_period, sim_params.x_amplitude, M_PI * sim_params.x_phase, sim_params.x_offset);
  utils::Sinusoid y_motion = utils::Sinusoid(1.0/sim_params.y_period, sim_params.y_amplitude, M_PI * sim_params.y_phase, sim_params.y_offset);
  utils::Sinusoid z_motion = utils::Sinusoid(1.0/sim_params.z_period, sim_params.z_amplitude, M_PI * sim_params.z_phase, sim_params.z_offset);
  utils::Sinusoid roll_motion = utils::Sinusoid(1.0/sim_params.roll_period, sim_params.roll_amplitude, M_PI * sim_params.roll_phase, M_PI * sim_params.roll_offset);
  utils::Sinusoid pitch_motion = utils::Sinusoid(1.0/sim_params.pitch_period, sim_params.pitch_amplitude, M_PI * sim_params.pitch_phase, M_PI * sim_params.pitch_offset);
  utils::Line yaw_motion = utils::Line(M_PI * sim_params.yaw_slope, sim_params.yaw_offset, true);

  // imu noise
  mat3 sigma_gyro = std::pow(sim_params.gyro_stdev, 2.0) * mat3::Identity();
  mat3 sigma_accel = std::pow(sim_params.accel_stdev, 2.0) * mat3::Identity();
  mat3 sigma_gyro_walk = std::pow(sim_params.gyro_walk_stdev, 2.0) * mat3::Identity();
  mat3 sigma_accel_walk = std::pow(sim_params.accel_walk_stdev, 2.0) * mat3::Identity();

  utils::MultivariateGaussian<3> gyro_dist(vec3::Zero(), sigma_gyro, sim_params.seed_random);
  utils::MultivariateGaussian<3> accel_dist(vec3::Zero(), sigma_accel, sim_params.seed_random);
  utils::MultivariateGaussian<3> gyro_walk_dist(vec3::Zero(), sigma_gyro_walk, sim_params.seed_random);
  utils::MultivariateGaussian<3> accel_walk_dist(vec3::Zero(), sigma_accel_walk, sim_params.seed_random);

  // wnoj traj sim noise
  utils::MultivariateGaussian<6> wnoj_traj_dist(vec6::Zero(), Q_gp, sim_params.seed_random);

  // apriltag pose noise
  mat6 sigma_apriltag_const = mat6::Zero();
  sigma_apriltag_const.block<3,3>(0,0) = std::pow(sim_params.pos_stdev, 2.0) * mat3::Identity();
  sigma_apriltag_const.block<3,3>(3,3) = std::pow(sim_params.rot_stdev, 2.0) * mat3::Identity();
  mat2 sigma_pix = std::pow(sim_params.pix_stdev, 2.0) * mat2::Identity();

  // create vector of apriltag corners in apriltag frame
  std::vector<vec3> apriltag_corners;
  apriltag_corners.push_back(sim_params.apriltag_side_length/2.0 * (vec3() << -1.0, 1.0, 0.0).finished());
  apriltag_corners.push_back(sim_params.apriltag_side_length/2.0 * (vec3() << 1.0, 1.0, 0.0).finished());
  apriltag_corners.push_back(sim_params.apriltag_side_length/2.0 * (vec3() << 1.0, -1.0, 0.0).finished());
  apriltag_corners.push_back(sim_params.apriltag_side_length/2.0 * (vec3() << -1.0, -1.0, 0.0).finished());

  // this is only used if sim_params.est_constant_covariance == true
  Eigen::LLT<mat6> apriltag_chol(sigma_apriltag_const.inverse());
  assert(apriltag_chol.info() == Eigen::Success);
  mat6 sqrt_W_apriltag_const = apriltag_chol.matrixL().transpose();

  Eigen::LLT<mat6> imu_chol((mat6() << sigma_gyro.inverse(), Eigen::Matrix3d::Zero(),
                                       Eigen::Matrix3d::Zero(), sigma_accel.inverse()).finished());
  assert(imu_chol.info() == Eigen::Success);
  mat6 sqrt_W_imu = imu_chol.matrixL().transpose();

  mat2 W_cam = sigma_pix.inverse();
  
  // set up estimator
  estimator::EstimatorParams est_params;
  est_params.g = sim_params.g;
  est_params.T_b_c = sim_params.T_imu_c;
  est_params.use_bias_splines = yaml["use_bias_splines"].as<bool>();
  est_params.bias_spline_k = yaml["bias_spline_k"].as<int>();
  est_params.bias_spline_dt = yaml["bias_spline_dt"].as<double>();
  est_params.accel_bias = vec3::Zero();
  est_params.gyro_bias = vec3::Zero();
  est_params.cam_mat = sim_params.cam_mat;
  est_params.apriltag_side_length = sim_params.apriltag_side_length;


  // monte carlo results
  std::vector<double> mc_pos_rmse, mc_rot_rmse, mc_twist_rmse, mc_solve_time, mc_time_iter, mc_lin_sol_time, mc_jac_eval_time, mc_sample_time;
  std::vector<int> mc_iters;

  // monte carlo loop
  for(int mc = 0; mc < sim_params.num_monte_carlo_runs; ++mc)
  {

    // create truth vectors
    std::vector<double> t_truth;
    std::vector<se3> Tib_truth;
    std::vector<vec6> twist_bi_b_truth;
    std::vector<vec3> gyro_bias_truth;
    std::vector<vec3> accel_bias_truth;

    // create measurement vectors
    std::shared_ptr<std::vector<std::pair<double, vec6>>> z_imu(new std::vector<std::pair<double, vec6>>);
    std::shared_ptr<std::vector<std::pair<double, std::vector<std::pair<int, se3>>>>> z_cam(new std::vector<std::pair<double, std::vector<std::pair<int, se3>>>>);

    // add initial biases
    gyro_bias_truth.push_back(gyro_bias_init);
    accel_bias_truth.push_back(accel_bias_init);

    // create fiducial marker
    std::vector<se3> lm;
    lm.push_back(se3(so3((vec3() << M_PI, 0.0, 0.0).finished()), vec3::Zero()));

    // last measurement times
    double last_imu_time = -1.0;
    double last_cam_time = -1.0;

    // initialize wnoj trajectory (if applicable)
    se3 Tib_last = sim_params.T_init;
    vec6 twist_last = sim_params.twist_init;
    vec6 twist_dot_last = sim_params.twist_dot_init;

    // start simulation
    for(double sim_time = 0.0; sim_time < sim_params.sim_end_time; sim_time += sim_params.sim_period)
    {
      se3 Tib;
      vec6 twist_bi_b;
      vec3 a_bi_b;

      if(sim_params.sim_traj_type == 0) // wnoj trajectory
      {
        vec6 twist_dot_bi_b = twist_dot_last + wnoj_traj_dist.sample() * sim_params.sim_period;
        twist_bi_b = twist_last + twist_dot_bi_b * sim_params.sim_period;
        Tib = se3::Exp(-twist_bi_b * sim_params.sim_period) * Tib_last;

        a_bi_b = twist_dot_bi_b.head<3>();

        Tib_last = Tib;
        twist_last = twist_bi_b;
        twist_dot_last = twist_dot_bi_b;
      }
      else // sinusoidal trajectory
      {
        // sample trajectory derivatives
        std::vector<double> x_derivs = x_motion.sample(sim_time, std::vector<int>({0,1,2,3}));
        std::vector<double> y_derivs = y_motion.sample(sim_time, std::vector<int>({0,1,2,3}));
        std::vector<double> z_derivs = z_motion.sample(sim_time, std::vector<int>({0,1,2,3}));
        vec3 pos = (vec3() << x_derivs[0], y_derivs[0], z_derivs[0]).finished();
        vec3 vel = (vec3() << x_derivs[1], y_derivs[1], z_derivs[1]).finished();
        vec3 acc = (vec3() << x_derivs[2], y_derivs[2], z_derivs[2]).finished();

        double phi = roll_motion.sample(sim_time);
        double phidot = roll_motion.sample_d1(sim_time);
        double theta = pitch_motion.sample(sim_time);
        double thetadot = pitch_motion.sample_d1(sim_time);
        double psi = yaw_motion.sample(sim_time);
        double psidot = yaw_motion.sample_d1(sim_time);

        // form pose and twist
        Tib = se3(so3(phi, theta, psi), pos).inverse();
        twist_bi_b = (vec6() << Tib.rotation() * vel, 
                                (vec3() << phidot - std::sin(theta)*psidot,
                                           std::cos(phi)*thetadot + std::sin(phi)*std::cos(theta)*psidot,
                                           -std::sin(phi)*thetadot + std::cos(phi)*std::cos(theta)*psidot).finished()
                      ).finished();
        
        a_bi_b = Tib.rotation() * acc - lie_groups::SO3d::hat(twist_bi_b.tail<3>()) * twist_bi_b.head<3>();
      }
      
      // update biases
      vec3 gyro_bias = gyro_bias_truth.back() + sim_params.sim_period * gyro_walk_dist.sample();
      vec3 accel_bias = accel_bias_truth.back() + sim_params.sim_period * accel_walk_dist.sample();

      // simulate apriltag measurement
      // if traj type is wnoj, then apriltag is always seen with constant covariance and does not need to be in fov
      if(sim_time - last_cam_time > 1.0/sim_params.camera_frequency)
      {
        // check if tag is in field of view
        se3 Tac = sim_params.T_imu_c * Tib * lm[0].inverse();
        vec3 trans = Tac.translation();
        double dist = trans.norm();
        double azimuth = std::atan2(trans(0), trans(2));
        double elevation = std::atan2(trans(1), trans(2));
        if(((dist < sim_params.max_range) && (std::abs(azimuth) < sim_params.max_azimuth) && (std::abs(elevation) < sim_params.max_elevation)) || (sim_params.sim_traj_type == 0))
        {
          if((!sim_params.assert_facing_camera || Tac.rotation().matrix()(2,2) < 0.0) || (sim_params.sim_traj_type == 0))
          {
            mat6 apriltag_covariance;
            if(sim_params.sim_constant_cov || (sim_params.sim_traj_type == 0)) apriltag_covariance = sigma_apriltag_const;
            else
            {
              Eigen::Matrix<double, 8, 6> J_apriltag_est;
              Eigen::Matrix<double, 8, 8> W_cam;
              W_cam.setZero();
              for(int i = 0; i < apriltag_corners.size(); ++i)
              {
                vec3 corner_c = Tac * apriltag_corners[i];
                Eigen::Matrix<double, 2, 3> dpi_dcorner = (Eigen::Matrix<double, 2, 3>() << 1.0/corner_c(2), 0.0, -corner_c(0)/std::pow(corner_c(2), 2),
                                                                                            0.0, 1.0/corner_c(2), -corner_c(1)/std::pow(corner_c(2), 2)).finished();
                Eigen::Matrix<double, 3, 6> dcorner_dT;
                dcorner_dT.block<3,3>(0,0).setIdentity();
                dcorner_dT.block<3,3>(0,3) = -lie_groups::SO3d::hat(corner_c);
                J_apriltag_est.block<2,6>(2*i, 0) = sim_params.cam_mat.block<2,2>(0,0) * dpi_dcorner * dcorner_dT;

                W_cam.block<2,2>(2*i, 2*i) = sigma_pix.inverse();
              }
              apriltag_covariance = (J_apriltag_est.transpose() * W_cam * J_apriltag_est).inverse();
            }

            utils::MultivariateGaussian<6> apriltag_dist(vec6::Zero(), apriltag_covariance, sim_params.seed_random);
            std::vector<std::pair<int, se3>> image;
            image.push_back(std::make_pair(0, se3::Exp(apriltag_dist.sample()) * Tac));
            z_cam->push_back(std::make_pair(sim_time, image));
          }
        }

        last_cam_time = sim_time;
      }

      // simulate imu measurement
      if(sim_time - last_imu_time > 1.0/sim_params.imu_frequency)
      {
        vec3 z_gyro = twist_bi_b.tail<3>() + gyro_bias + gyro_dist.sample();
        vec3 z_accel = a_bi_b - sim_params.g * Tib.rotation().matrix().col(2) + accel_bias + accel_dist.sample();
        z_imu->push_back(std::make_pair(sim_time, (vec6() << z_gyro, z_accel).finished()));

        last_imu_time = sim_time;
      }

      // save truth data
      t_truth.push_back(sim_time);
      Tib_truth.push_back(Tib);
      twist_bi_b_truth.push_back(twist_bi_b);
      gyro_bias_truth.push_back(gyro_bias);
      accel_bias_truth.push_back(accel_bias);
    }

    // discard imu measurements that were obtained after the last camera measurement
    while(z_imu->back().first >= z_cam->back().first) z_imu->pop_back();

    std::unique_ptr<estimator::EstimatorBase> estimator;
  #if ESTTYPE < 2
    estimator.reset(new estimator::SplineEstimator<se3spline>(est_params));
  #elif ESTTYPE == 2
    estimator.reset(new gpest(est_params));
  #elif ESTTYPE == 3
    estimator.reset(new gpest(est_params));
  #endif

    estimator->set_landmark_truth(lm);

    if(sim_params.est_constant_cov || (sim_params.sim_traj_type == 0)) estimator->add_lm_pose_measurements(z_cam, sqrt_W_apriltag_const);
    else estimator->add_lm_pose_measurements(z_cam, W_cam);
    // estimator->add_imu_measurements(z_imu, sqrt_W_imu);

  #if ESTTYPE == 0
    static_cast<estimator::SplineEstimator<se3spline>*>(estimator.get())->init_spline(sim_params.spline_dt, sim_params.spline_k);
    static_cast<estimator::SplineEstimator<se3spline>*>(estimator.get())->set_up_motion_priors(static_cast<estimator::ModelType>(sim_params.spline_son_mp_type), 
                            sim_params.spline_son_mp_dt, Q_gp);
  #elif ESTTYPE == 1
    static_cast<estimator::SplineEstimator<se3spline>*>(estimator.get())->init_spline(sim_params.spline_dt, sim_params.spline_k);
    static_cast<estimator::SplineEstimator<se3spline>*>(estimator.get())->set_up_motion_priors(static_cast<estimator::ModelType>(sim_params.spline_son_mp_type), 
                            static_cast<estimator::ModelType>(sim_params.spline_rn_mp_type), sim_params.spline_son_mp_dt, sim_params.spline_rn_mp_dt, Q_gp);
  #elif ESTTYPE == 2
    static_cast<gpest*>(estimator.get())->init_gp(Q_gp, sim_params.n_gp);
  #elif ESTTYPE == 3
    static_cast<gpest*>(estimator.get())->init_gp(Q_gp, sim_params.n_gp, static_cast<gp::ModelType>(sim_params.gp_son_type));
  #endif

    // solve
    ceres::Solver::Summary summary1 = estimator->solve(sim_params.print_est_prog, false, sim_params.max_num_iterations);
    double solve_time = summary1.total_time_in_seconds;
    double eval_time = summary1.residual_evaluation_time_in_seconds + summary1.jacobian_evaluation_time_in_seconds;
    double lin_sol_time = summary1.total_time_in_seconds - eval_time;
    int iters = summary1.iterations.size();

    if(sim_params.use_imu || sim_params.use_mp) 
    {
      if(sim_params.use_imu) estimator->add_imu_measurements(z_imu, sqrt_W_imu);
      ceres::Solver::Summary summary2 = estimator->solve(sim_params.print_est_prog, sim_params.use_mp, sim_params.max_num_iterations);
      solve_time += summary2.total_time_in_seconds;
      eval_time += summary2.residual_evaluation_time_in_seconds + summary2.jacobian_evaluation_time_in_seconds;
      lin_sol_time += summary2.total_time_in_seconds - (summary2.residual_evaluation_time_in_seconds + summary2.jacobian_evaluation_time_in_seconds);
      iters = summary2.iterations.size();
    }

    // create time vector for sampling results
    std::pair<double, double> estimator_time_bounds = estimator->get_start_and_end_time();
    std::vector<double> t_samp_vec;
    t_samp_vec.push_back(estimator_time_bounds.first);
    while(t_samp_vec.back() < estimator_time_bounds.second - sim_params.sim_period)
      t_samp_vec.push_back(t_samp_vec.back() + sim_params.sim_period);


  const auto start_st = std::chrono::high_resolution_clock::now();
  #if ESTTYPE < 2
    std::shared_ptr<se3spline> spline = static_cast<estimator::SplineEstimator<se3spline>*>(estimator.get())->get_spline();
    spline->pre_eval(t_samp_vec, true, true, false, false, false, false);
  #else
    std::shared_ptr<se3gp> gp = static_cast<gpest*>(estimator.get())->get_gp();
    gp->pre_eval(t_samp_vec, true, true, false, false, false, false);
  #endif
    const auto stop_st = std::chrono::high_resolution_clock::now();
    double sampling_time = std::chrono::duration<double, std::milli>(stop_st - start_st).count() / 1000.0;

    // sample optimized trajectory
    std::vector<se3> T_opt;
    std::vector<vec6> twist_opt;
    for(int i = 0; i < t_samp_vec.size(); ++i)
    {
      se3 T_samp;
      vec6 twist_samp;
  #if ESTTYPE == 0
      spline->get_pre_eval(i, &T_samp, &twist_samp);
  #elif ESTTYPE == 1
      so3xr3 R_i_bxp;
      vec6 v_i_om;
      spline->get_pre_eval(i, &R_i_bxp, &v_i_om);
      T_samp = se3(R_i_bxp.rotation(), -(R_i_bxp.rotation() * R_i_bxp.translation()));
      twist_samp = (vec6() << -(R_i_bxp.rotation() * v_i_om.head<3>()), v_i_om.tail<3>()).finished();
  #elif ESTTYPE == 2
      gp->get_pre_eval(i, &T_samp, &twist_samp);
  #elif ESTTYPE == 3
      so3 R;
      vec3 p, v, om;
      gp->get_pre_eval(i, &p, &v, nullptr, &R, &om);
      T_samp = se3(R, -(R * p));
      twist_samp = (vec6() << -(R * v), om).finished();
  #endif
      T_opt.push_back(T_samp);
      twist_opt.push_back(-twist_samp);
    }

    // get optimized imu biases
    std::vector<vec3> gyro_bias_opt;
    std::vector<vec3> accel_bias_opt;
    vec3 gyro_bias_const, accel_bias_const;
    std::shared_ptr<biasspline> gyro_bias_spline, accel_bias_spline;
    if(sim_params.use_imu)
    {
      if(!est_params.use_bias_splines)
      {
        estimator->get_gyro_bias(&gyro_bias_const);
        estimator->get_accel_bias(&accel_bias_const);
      }
      else
      {
    #if ESTTYPE < 2
        static_cast<estimator::SplineEstimator<se3spline>*>(estimator.get())->get_gyro_bias(gyro_bias_spline);
        static_cast<estimator::SplineEstimator<se3spline>*>(estimator.get())->get_accel_bias(accel_bias_spline);

        gyro_bias_spline->pre_eval(t_samp_vec, true, false, false, false, false, false);
        accel_bias_spline->pre_eval(t_samp_vec, true, false, false, false, false, false);
    #endif
      }
    }

    // sample imu biases
    for(int i = 0; i < t_samp_vec.size(); ++i)
    {
      vec3 gyro_bias_samp, accel_bias_samp;
      if(sim_params.use_imu)
      {
        if(est_params.use_bias_splines)
        {
          gyro_bias_spline->get_pre_eval(i, &gyro_bias_samp);
          accel_bias_spline->get_pre_eval(i, &accel_bias_samp);
        }
        else
        {
          gyro_bias_samp = gyro_bias_const;
          accel_bias_samp = accel_bias_const;
        }
        gyro_bias_opt.push_back(gyro_bias_samp);
        accel_bias_opt.push_back(accel_bias_samp);
      }
      else
      {
        gyro_bias_opt.push_back(vec3::Zero());
        accel_bias_opt.push_back(vec3::Zero());
      }
    }

    // compute estimated imu measurements
    std::vector<vec6> est_imu;
    for(int i = 0; i < t_samp_vec.size(); ++i)
    {
      vec3 a_bi_b;
  #if ESTTYPE == 0
      vec6 a;
      spline->eval(t_samp_vec[i], nullptr, nullptr, &a);
      a_bi_b = -a.head<3>();
  #elif ESTTYPE == 1
      vec6 a_i_omdot;
      spline->eval(t_samp_vec[i], nullptr, nullptr, &a_i_omdot);
      a_bi_b = T_opt[i].rotation() * a_i_omdot.head<3>();
  #elif ESTTYPE == 2
      vec6 a;
      gp->eval(t_samp_vec[i], nullptr, nullptr, &a);
      a_bi_b = -a.head<3>();
  #elif ESTTYPE == 3
      vec3 a;
      gp->eval(t_samp_vec[i], nullptr, nullptr, &a);
      a_bi_b = T_opt[i].rotation() * a;
  #endif
      vec3 est_gyro = twist_opt[i].tail<3>() + gyro_bias_opt[i];
      vec3 est_accel = a_bi_b - sim_params.g * T_opt[i].rotation().matrix().col(2) + accel_bias_opt[i];
      est_imu.push_back((vec6() << est_gyro, est_accel).finished());
    }

    // compute trajectory rmse
    double position_error_sum = 0.0;
    double rotation_error_sum = 0.0;
    double twist_error_sum = 0.0;
    for(int i = 0; i < t_samp_vec.size(); ++i)
    {
      se3 T_be_bt = Tib_truth[i] * T_opt[i].inverse();
      position_error_sum += std::pow((T_opt[i].inverse().translation() - Tib_truth[i].inverse().translation()).norm(), 2.0);
      rotation_error_sum += std::pow((T_opt[i].rotation().inverse() * Tib_truth[i].rotation()).Log().norm(), 2.0);
      twist_error_sum += std::pow((twist_bi_b_truth[i] - T_be_bt.Ad() * twist_opt[i]).norm(), 2.0);
    }
    double position_rmse = std::sqrt(position_error_sum/t_samp_vec.size());
    double rotation_rmse = std::sqrt(rotation_error_sum/t_samp_vec.size());
    double twist_rmse = std::sqrt(twist_error_sum/t_samp_vec.size());

    std::cout << "Run " << mc << ". Pos. rmse, rot. rmse, twist rmse, solve time, iters, time/iter: " << position_rmse << ", " << rotation_rmse << ", " << twist_rmse << ", " 
                                                                                    << solve_time << ", " << iters << ", " << solve_time/iters << "\n";

    // store mc results
    mc_pos_rmse.push_back(position_rmse);
    mc_rot_rmse.push_back(rotation_rmse);
    mc_twist_rmse.push_back(twist_rmse);
    mc_solve_time.push_back(solve_time);
    mc_iters.push_back(iters);
    mc_time_iter.push_back(solve_time/iters);
    mc_lin_sol_time.push_back(lin_sol_time);
    mc_jac_eval_time.push_back(eval_time);
    mc_sample_time.push_back(sampling_time);

    // log stuff
    if(sim_params.log_results)
    {
      boost::filesystem::path dir(sim_params.log_directory);
      if(!boost::filesystem::exists(dir))
      {
        std::cout << "Creating folder " << sim_params.log_directory << "\n";
        if(!boost::filesystem::create_directory(sim_params.log_directory))
          std::cout << "Couldn't create the folder!\n";
      }

      // truth data
      std::ofstream truth_file(sim_params.log_directory + "truth.txt");
      for(int i = 0; i < t_truth.size(); ++i)
      {
        vec3 p_bi_i = Tib_truth[i].inverse().translation();
        vec3 rib = Tib_truth[i].rotation().Log();
        truth_file << t_truth[i] << " " << p_bi_i(0) << " " << p_bi_i(1) << " " << p_bi_i(2) << " " << rib(0) << " " << rib(1) << " " << rib(2) << " "
          << twist_bi_b_truth[i](0) << " " << twist_bi_b_truth[i](1) << " " << twist_bi_b_truth[i](2) << " " << twist_bi_b_truth[i](3) << " " << twist_bi_b_truth[i](4) << " " << twist_bi_b_truth[i](5) << " "
          << gyro_bias_truth[i+1](0) << " " << gyro_bias_truth[i+1](1) << " " << gyro_bias_truth[i+1](2) << " " << accel_bias_truth[i+1](0) << " " << accel_bias_truth[i+1](1) << " " << accel_bias_truth[i+1](2) << "\n";
      }

      // est data
      std::ofstream opt_file(sim_params.log_directory + "opt.txt");
      for(int i = 0; i < t_samp_vec.size(); ++i)
      {
        vec3 p_bi_i = T_opt[i].inverse().translation();
        vec3 r_ib = T_opt[i].rotation().Log();
        opt_file << t_samp_vec[i] << " " << p_bi_i(0) << " " << p_bi_i(1) << " " << p_bi_i(2) << " " << r_ib(0) << " " << r_ib(1) << " " << r_ib(2) << " "
          << twist_opt[i](0) << " " << twist_opt[i](1) << " " << twist_opt[i](2) << " " << twist_opt[i](3) << " " << twist_opt[i](4) << " " << twist_opt[i](5) << " "
          << gyro_bias_opt[i](0) << " " << gyro_bias_opt[i](1) << " " << gyro_bias_opt[i](2) << " " << accel_bias_opt[i](0) << " " << accel_bias_opt[i](1) << " " << accel_bias_opt[i](2) << "\n";
      }

      // imu data
      std::ofstream imu_file(sim_params.log_directory + "imu.txt");
      for(const auto& meas : *z_imu)
      {
        imu_file << meas.first << " " << meas.second(0) << " " << meas.second(1) << " " << meas.second(2) << " " << meas.second(3) << " " << meas.second(4) << " " << meas.second(5) << "\n";
      }

      // // estimated imu data
      std::ofstream est_imu_file(sim_params.log_directory + "est_imu.txt");
      for(int i = 0; i < t_samp_vec.size(); ++i)
      {
        est_imu_file << t_samp_vec[i] << " " << est_imu[i](0) << " " << est_imu[i](1) << " " << est_imu[i](2) << " " << est_imu[i](3) << " " << est_imu[i](4) << " " << est_imu[i](5) << "\n"; 
      }
    }

  } // end monte carlo loop


  // find medians and print results
  std::cout << "\nMedian MC run vals\nPos rmse, rot rmse, twist rmse, solve time, iters, time/iter, jac eval time, lin sol time, sample time: " << find_median<double>(mc_pos_rmse) << ", " << find_median<double>(mc_rot_rmse) << ", " 
                                                    << find_median<double>(mc_twist_rmse) << ", " << find_median<double>(mc_solve_time) << ", " << find_median<int>(mc_iters) << ", "
                                                    << find_median<double>(mc_time_iter) << ", " << find_median<double>(mc_jac_eval_time) << ", " << find_median<double>(mc_lin_sol_time) << ", " << find_median<double>(mc_sample_time) << "\n";

  std::cout << "\nMean MC run vals\nPos rmse, rot rmse, twist rmse, solve time, iters, time/iter, jac eval time, lin sol time: " << find_mean<double>(mc_pos_rmse) << ", " << find_mean<double>(mc_rot_rmse) << ", " 
                                                    << find_mean<double>(mc_twist_rmse) << ", " << find_mean<double>(mc_solve_time) << ", " << find_mean<int>(mc_iters) << ", "
                                                    << find_mean<double>(mc_time_iter) << ", " << find_mean<double>(mc_jac_eval_time) << ", " << find_mean<double>(mc_lin_sol_time) << ", " << find_mean<double>(mc_sample_time) << "\n";
}