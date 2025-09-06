#include <vector>
#include <memory>
#include <string>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>

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
#include "hw/hw_params.h"
#include "estimator/estimator_base.h"
#include "estimator/spline_estimator.hpp"
#include "estimator/gp_se3_estimator.h"
#include "estimator/gp_so3xr3_estimator.h"

#define ESTTYPE 2

using vec3 = Eigen::Vector3d;
using vec6 = Eigen::Matrix<double, 6, 1>;
using mat2 = Eigen::Matrix2d;
using mat3 = Eigen::Matrix3d;
using mat4 = Eigen::Matrix<double, 4, 4>;
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

int main()
{
  // load parameters
  YAML::Node yaml = YAML::LoadFile("../params/hw.yaml");
  hw::HwParams hw_params;

  // sensor parameters
  double fx = yaml["fx"].as<double>();
  double fy = yaml["fy"].as<double>();
  double cx = yaml["cx"].as<double>();
  double cy = yaml["cy"].as<double>();
  hw_params.cam_mat = (mat3() << fx, 0.0, cx,
                                  0.0, fy, cy,
                                  0.0, 0.0, 1.0).finished();
  std::vector<double> T_imu_cam_params = yaml["T_imu_cam"].as<std::vector<double>>();
  mat4 T_imu_cam_mat = Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(T_imu_cam_params.data());
  hw_params.T_imu_cam = se3(T_imu_cam_mat);
  hw_params.timeshift_cam_imu = yaml["timeshift_cam_imu"].as<double>();
  hw_params.use_constant_pose_covariance = yaml["use_constant_pose_covariance"].as<bool>();
  hw_params.sigma_april_trans = yaml["sigma_april_trans"].as<double>();
  hw_params.sigma_april_rot = yaml["sigma_april_rot"].as<double>();
  hw_params.sigma_pix = yaml["sigma_pix"].as<double>();
  hw_params.sigma_gyro = yaml["sigma_gyro"].as<double>();
  hw_params.sigma_accel = yaml["sigma_accel"].as<double>();
  hw_params.g = yaml["g"].as<double>();
  hw_params.apriltag_side_length = yaml["apriltag_side_length"].as<double>();

  // data parameters
  hw_params.data_path = yaml["data_path"].as<std::string>();
  hw_params.data_start_times = yaml["data_start_times"].as<std::vector<double>>();
  hw_params.data_end_times = yaml["data_end_times"].as<std::vector<double>>();
  hw_params.apriltag_ids = yaml["apriltag_ids"].as<std::vector<int>>();

  // residuals to use
  hw_params.use_imu = yaml["use_imu"].as<bool>();
  hw_params.use_mp = yaml["use_mp"].as<bool>();

  // logging
  hw_params.log_results = yaml["log_results"].as<bool>();
  hw_params.log_directory = yaml["log_directory"].as<std::string>();
  hw_params.print_est_prog = yaml["print_est_prog"].as<bool>();
  hw_params.max_num_iterations = yaml["max_num_iterations"].as<int>();
  hw_params.sample_period = yaml["sample_period"].as<double>();

  // spline estimation parameters
  hw_params.spline_k = yaml["spline_k"].as<int>();
  hw_params.spline_dt = yaml["spline_dt"].as<double>();
  hw_params.spline_son_mp_dt = yaml["spline_son_mp_dt"].as<double>();
  hw_params.spline_rn_mp_dt = yaml["spline_rn_mp_dt"].as<double>();
  hw_params.spline_son_mp_type = yaml["spline_son_mp_type"].as<int>();
  hw_params.spline_rn_mp_type = yaml["spline_rn_mp_type"].as<int>();

  // gp estimation parameters
  hw_params.q_gp_rot = yaml["q_gp_rot"].as<double>();
  hw_params.q_gp_pos = yaml["q_gp_pos"].as<double>();
  hw_params.n_gp = yaml["n_gp"].as<int>();
  hw_params.gp_son_type = yaml["gp_son_type"].as<int>();
  mat6 Q_gp = mat6::Zero();
  Q_gp.block<3,3>(0,0) = hw_params.q_gp_pos * mat3::Identity();
  Q_gp.block<3,3>(3,3) = hw_params.q_gp_rot * mat3::Identity();

  // mocap to imu calibration
  mat4 T_imu_mocap_mat = Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(yaml["T_imu_mocap"].as<std::vector<double>>().data());
  hw_params.T_imu_mocap = se3(T_imu_mocap_mat);
  hw_params.timeshift_mocap_cam = yaml["timeshift_mocap_cam"].as<double>();
  hw_params.calibrate_mocap_tx = yaml["calibrate_mocap_tx"].as<bool>();
  hw_params.low_pass_filter_mocap_orientation = yaml["low_pass_filter_mocap_orientation"].as<bool>();
  hw_params.mocap_lpf_alpha = yaml["mocap_lpf_alpha"].as<double>();  

  // estimator params
  estimator::EstimatorParams est_params;
  est_params.g = hw_params.g;
  est_params.T_b_c = hw_params.T_imu_cam;
  est_params.use_bias_splines = yaml["use_bias_splines"].as<bool>();
  est_params.bias_spline_k = yaml["bias_spline_k"].as<int>();
  est_params.bias_spline_dt = yaml["bias_spline_dt"].as<double>();
  est_params.accel_bias = vec3::Zero();
  est_params.gyro_bias = vec3::Zero();
  est_params.cam_mat = hw_params.cam_mat;
  est_params.apriltag_side_length = hw_params.apriltag_side_length;

  // create sqrt_W matrices
  mat6 sigma_apriltag_const = mat6::Identity();
  sigma_apriltag_const.block<3,3>(0,0) *= std::pow(hw_params.sigma_april_trans, 2.0);
  sigma_apriltag_const.block<3,3>(3,3) *= std::pow(hw_params.sigma_april_rot, 2.0);
  Eigen::LLT<mat6> apriltag_const_chol(sigma_apriltag_const.inverse());
  mat6 sqrt_W_apriltag_const = apriltag_const_chol.matrixL().transpose();

  mat6 sigma_imu = mat6::Identity();
  sigma_imu.block<3,3>(0,0) *= std::pow(hw_params.sigma_gyro, 2.0);
  sigma_imu.block<3,3>(3,3) *= std::pow(hw_params.sigma_accel, 2.0);
  Eigen::LLT<mat6> imu_chol(sigma_imu.inverse());
  mat6 sqrt_W_imu = imu_chol.matrixL().transpose();

  mat2 sigma_pix = std::pow(hw_params.sigma_pix, 2.0) * mat2::Identity();

  // loop through each data segment
  for(int dst = 0; dst < hw_params.data_start_times.size(); ++dst)
  {

    // instantiate measurement buffers
    std::shared_ptr<std::vector<std::pair<double, vec6>>> z_imu(new std::vector<std::pair<double, vec6>>);
    std::shared_ptr<std::vector<std::pair<double, std::vector<std::pair<int, se3>>>>> z_cam(new std::vector<std::pair<double, std::vector<std::pair<int, se3>>>>);
    std::vector<se3> apriltag_poses;
    std::vector<std::pair<double, se3>> mocap_poses;

    // load mocap data
    std::fstream mocap_file(hw_params.data_path + "hw_rover_truth.txt");
    if(mocap_file)
    {
      so3 R_prev;
      bool start = true;
      std::string line;
      while(std::getline(mocap_file, line))
      {
        double t, x, y, z, qw, qx, qy, qz;
        std::stringstream ss(line);
        ss >> t >> x >> y >> z >> qw >> qx >> qy >> qz;
        if(t < hw_params.data_start_times[dst] || t > hw_params.data_end_times[dst]) continue;

        Eigen::Quaterniond quat(qw, qx, qy, qz);
        vec3 trans = (vec3() << x, y, z).finished();
        se3 T_i_m = se3(so3(quat), trans).inverse(); // here m is the frame that the mocap system assigned to the rover. Does not coincide with camera or imu

        if(hw_params.low_pass_filter_mocap_orientation)
        {
          if(start)
          {
            R_prev = T_i_m.rotation();
            start = false;
          }
          else
          {
            vec3 trans = T_i_m.inverse().translation();
            so3 R_lpf = so3::Exp(hw_params.mocap_lpf_alpha * (R_prev * T_i_m.rotation().inverse()).Log()) * T_i_m.rotation();
            T_i_m = se3(R_lpf, -(R_lpf * trans));
            R_prev = R_lpf;
          }
        }

        mocap_poses.push_back(std::make_pair(t, T_i_m));
      }
    }
    else std::cout << "File " + hw_params.data_path + "hw_rover_truth.txt does not exist. Will not plot truth.\n";

    // load apriltag mocap poses
    for(const int id : hw_params.apriltag_ids)
    {
      std::fstream apriltag_mocap_file(hw_params.data_path + "hw_april" + std::to_string(id) + ".txt");
      if(apriltag_mocap_file)
      {
        // only use the first mocap update to determine apriltag pose
        std::string line;
        std::getline(apriltag_mocap_file, line);
        std::stringstream ss(line);
        double t, x, y, z, qw, qx, qy, qz;
        ss >> t >> x >> y >> z >> qw >> qx >> qy >> qz;
        Eigen::Quaterniond quat(qw, qx, qy, qz);
        vec3 trans = (vec3() << x, y, z).finished();
        // motion capture screws with apriltag coordinate axes before publishing. Have to undo
        // apriltag_poses.push_back(se3(so3(M_PI, 0.0, M_PI/2.0), vec3::Zero()).inverse() * se3(so3(quat), trans).inverse());
        apriltag_poses.push_back(se3(so3(quat), trans).inverse());
      }
      else std::cout << "File " + hw_params.data_path + "hw_april" + std::to_string(id) + ".txt does not exist!\n";
    }

    // load imu data
    std::fstream imu_file(hw_params.data_path + "hw_imu.txt");
    if(imu_file)
    {
      std::string line;
      while(std::getline(imu_file, line))
      {
        double t, ax, ay, az, gx, gy, gz;
        std::stringstream ss(line);
        ss >> t >> ax >> ay >> az >> gx >> gy >> gz;
        double t_shift = t - hw_params.timeshift_cam_imu;
        if(t_shift < hw_params.data_start_times[dst] || t_shift > hw_params.data_end_times[dst]) continue;

        z_imu->push_back(std::make_pair(t_shift, (vec6() << gx, gy, gz, ax, ay, az).finished()));
      }
    }
    else std::cout << "File " + hw_params.data_path + "hw_imu.txt does not exist!\n";

    // load apriltag data
    std::fstream tag_file(hw_params.data_path + "hw_apriltag.txt");
    if(tag_file)
    {
      std::string line;
      while(std::getline(tag_file, line))
      {
        int id;
        double t, x, y, z, qw, qx, qy, qz;
        std::stringstream ss(line);
        ss >> id >> t >> x >> y >> z >> qw >> qx >> qy >> qz;
        if(t < hw_params.data_start_times[dst] || t > hw_params.data_end_times[dst]) continue;
        
        Eigen::Quaterniond quat(qw, qx, qy, qz);
        vec3 trans = (vec3() << x, y, z).finished();
        se3 T_a_c = se3(so3(quat), trans);

        int vec_id = std::find(hw_params.apriltag_ids.begin(), hw_params.apriltag_ids.end(), id) - hw_params.apriltag_ids.begin();
        if(vec_id >= hw_params.apriltag_ids.size()) continue;

        if(z_cam->size() && std::abs(t - z_cam->back().first) < 1e-6) z_cam->back().second.push_back(std::make_pair(vec_id, T_a_c));
        else z_cam->push_back(std::make_pair(t, std::vector<std::pair<int, se3>>({std::make_pair(vec_id, T_a_c)})));
      }
    }
    else std::cout << "File " + hw_params.data_path + "hw_apriltag.txt does not exist!\n";

    // instantiate estimator
    std::unique_ptr<estimator::EstimatorBase> estimator;
  #if ESTTYPE < 2
    estimator.reset(new estimator::SplineEstimator<se3spline>(est_params));
  #elif ESTTYPE == 2
    estimator.reset(new gpest(est_params));
  #elif ESTTYPE == 3
    estimator.reset(new gpest(est_params));
  #endif

    // solve
    estimator->set_landmark_truth(apriltag_poses);
    if(hw_params.use_constant_pose_covariance) estimator->add_lm_pose_measurements(z_cam, sqrt_W_apriltag_const);
    else estimator->add_lm_pose_measurements(z_cam, sigma_pix.inverse().eval());

  #if ESTTYPE == 0
    static_cast<estimator::SplineEstimator<se3spline>*>(estimator.get())->init_spline(hw_params.spline_dt, hw_params.spline_k);
    static_cast<estimator::SplineEstimator<se3spline>*>(estimator.get())->set_up_motion_priors(static_cast<estimator::ModelType>(hw_params.spline_son_mp_type), 
                            hw_params.spline_son_mp_dt, Q_gp);
  #elif ESTTYPE == 1
    static_cast<estimator::SplineEstimator<se3spline>*>(estimator.get())->init_spline(hw_params.spline_dt, hw_params.spline_k);
    static_cast<estimator::SplineEstimator<se3spline>*>(estimator.get())->set_up_motion_priors(static_cast<estimator::ModelType>(hw_params.spline_son_mp_type), 
                            static_cast<estimator::ModelType>(hw_params.spline_rn_mp_type), hw_params.spline_son_mp_dt, hw_params.spline_rn_mp_dt, Q_gp);
  #elif ESTTYPE == 2
    static_cast<gpest*>(estimator.get())->init_gp(Q_gp, hw_params.n_gp);
  #elif ESTTYPE == 3
    static_cast<gpest*>(estimator.get())->init_gp(Q_gp, hw_params.n_gp, static_cast<gp::ModelType>(hw_params.gp_son_type));
  #endif

    ceres::Solver::Summary summary1 = estimator->solve(hw_params.print_est_prog, false, hw_params.max_num_iterations);
    double solve_time = summary1.total_time_in_seconds;
    double eval_time = summary1.residual_evaluation_time_in_seconds + summary1.jacobian_evaluation_time_in_seconds;
    double lin_sol_time = summary1.total_time_in_seconds - eval_time;
    int iters = summary1.iterations.size();

    if(hw_params.use_imu || hw_params.use_mp) 
    {
      if(hw_params.use_imu) estimator->add_imu_measurements(z_imu, sqrt_W_imu);
      ceres::Solver::Summary summary2 = estimator->solve(hw_params.print_est_prog, hw_params.use_mp, hw_params.max_num_iterations);
      solve_time += summary2.total_time_in_seconds;
      eval_time += summary2.residual_evaluation_time_in_seconds + summary2.jacobian_evaluation_time_in_seconds;
      lin_sol_time += summary2.total_time_in_seconds - (summary2.residual_evaluation_time_in_seconds + summary2.jacobian_evaluation_time_in_seconds);
      iters = summary2.iterations.size();
    }

    // create time vector for sampling results
    std::pair<double, double> estimator_time_bounds = estimator->get_start_and_end_time();
    std::vector<double> t_samp_vec;
    t_samp_vec.push_back(estimator_time_bounds.first);
    while(t_samp_vec.back() < estimator_time_bounds.second - hw_params.sample_period)
      t_samp_vec.push_back(t_samp_vec.back() + hw_params.sample_period);

  #if ESTTYPE < 2
    std::shared_ptr<se3spline> spline = static_cast<estimator::SplineEstimator<se3spline>*>(estimator.get())->get_spline();
    spline->pre_eval(t_samp_vec, true, true, false, false, false, false);
  #else
    std::shared_ptr<se3gp> gp = static_cast<gpest*>(estimator.get())->get_gp();
    gp->pre_eval(t_samp_vec, true, true, false, false, false, false);
  #endif

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
    if(hw_params.use_imu)
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
      if(hw_params.use_imu)
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
      vec3 est_accel = a_bi_b - hw_params.g * T_opt[i].rotation().matrix().col(2) + accel_bias_opt[i];
      est_imu.push_back((vec6() << est_gyro, est_accel).finished());
    }

    // calibrate mocap transformation
    std::pair<double, lie_groups::SE3d> T_bm = std::make_pair(hw_params.timeshift_mocap_cam, hw_params.T_imu_mocap);
  #if ESTTYPE == 0
    if(hw_params.calibrate_mocap_tx)
      T_bm = static_cast<estimator::SplineEstimator<se3spline>*>(estimator.get())->calibrate_mocap_transformation(std::make_shared<std::vector<std::pair<double, se3>>>(mocap_poses), 
        hw_params.T_imu_mocap, hw_params.timeshift_mocap_cam);
  #endif

    // log stuff
    if(hw_params.log_results)
    {
      boost::filesystem::path dir(hw_params.log_directory);
      if(!boost::filesystem::exists(dir))
      {
        std::cout << "Creating folder " << hw_params.log_directory << "\n";
        if(!boost::filesystem::create_directory(hw_params.log_directory))
          std::cout << "Couldn't create the folder!\n";
      }

      // mocap data
      std::ofstream mocap_file(hw_params.log_directory + "mocap.txt");
      for(auto const& pose : mocap_poses)
      {
        se3 T_ib_truth = T_bm.second.inverse() * pose.second;
        vec3 p_bi_i = T_ib_truth.inverse().translation();
        vec3 rib = T_ib_truth.rotation().Log();
        mocap_file << pose.first + T_bm.first << " " << p_bi_i(0) << " " << p_bi_i(1) << " " << p_bi_i(2) << " " << rib(0) << " " << rib(1) << " " << rib(2) << "\n";
      }

      // apriltag mocap data
      std::ofstream apriltag_mocap_file(hw_params.log_directory + "apriltag_mocap.txt");
      for(auto const& pose : apriltag_poses)
      {
        vec3 p_fi_i = pose.inverse().translation();
        vec3 rif = pose.rotation().Log();
        apriltag_mocap_file << p_fi_i(0) << " " << p_fi_i(1) << " " << p_fi_i(2) << " " << rif(0) << " " << rif(1) << " " << rif(2) << "\n";
      }

      // est data
      std::ofstream opt_file(hw_params.log_directory + "opt.txt");
      for(int i = 0; i < t_samp_vec.size(); ++i)
      {
        vec3 p_bi_i = T_opt[i].inverse().translation();
        vec3 r_ib = T_opt[i].rotation().Log();
        opt_file << t_samp_vec[i] << " " << p_bi_i(0) << " " << p_bi_i(1) << " " << p_bi_i(2) << " " << r_ib(0) << " " << r_ib(1) << " " << r_ib(2) << " "
          << twist_opt[i](0) << " " << twist_opt[i](1) << " " << twist_opt[i](2) << " " << twist_opt[i](3) << " " << twist_opt[i](4) << " " << twist_opt[i](5) << " "
          << gyro_bias_opt[i](0) << " " << gyro_bias_opt[i](1) << " " << gyro_bias_opt[i](2) << " " << accel_bias_opt[i](0) << " " << accel_bias_opt[i](1) << " " << accel_bias_opt[i](2) << "\n";
      }

      // imu data
      std::ofstream imu_file(hw_params.log_directory + "imu.txt");
      for(const auto& meas : *z_imu)
      {
        imu_file << meas.first << " " << meas.second(0) << " " << meas.second(1) << " " << meas.second(2) << " " << meas.second(3) << " " << meas.second(4) << " " << meas.second(5) << "\n";
      }

      // // estimated imu data
      std::ofstream est_imu_file(hw_params.log_directory + "est_imu.txt");
      for(int i = 0; i < t_samp_vec.size(); ++i)
      {
        est_imu_file << t_samp_vec[i] << " " << est_imu[i](0) << " " << est_imu[i](1) << " " << est_imu[i](2) << " " << est_imu[i](3) << " " << est_imu[i](4) << " " << est_imu[i](5) << "\n"; 
      }
    }

    // compute rmse
    double pos_err = 0.0;
    double rot_err = 0.0;
    int np = 0;
    for(auto const& T_im : mocap_poses)
    {
      se3 T_ib_true = T_bm.second.inverse() * T_im.second;
      double t = T_bm.first + T_im.first;
      if(t < estimator->get_start_and_end_time().first || t > estimator->get_start_and_end_time().second) continue;
      se3 T_ib_est;
  #if ESTTYPE == 0
      spline->eval(t, &T_ib_est);
  #elif ESTTYPE == 1
      so3xr3 R_i_bxp;
      spline->eval(t, &R_i_bxp);
      T_ib_est = se3(R_i_bxp.rotation(), -(R_i_bxp.rotation() * R_i_bxp.translation()));
  #elif ESTTYPE == 2
      gp->eval(t, &T_ib_est);
  #elif ESTTYPE == 3
      so3 R;
      vec3 p;
      gp->eval(t, &p, nullptr, nullptr, &R);
      T_ib_est = se3(R, -(R * p));
  #endif
      double r_err = std::pow((T_ib_est.rotation().inverse() * T_ib_true.rotation()).Log().norm(), 2.0);
      if(r_err > 0.1) 
      {
        std::cout << "Rot. error unnaturally high at t=" << t << ": " << r_err << ". Skipping.\n";
        continue;
      }
      pos_err += std::pow((T_ib_est.inverse().translation() - T_ib_true.inverse().translation()).norm(), 2.0);
      rot_err += r_err;
      np++;
    }

    std::cout << hw_params.data_start_times[dst] << "-" << hw_params.data_end_times[dst] << ". Pos rmse, rot rmse, solve time, num iters: " 
              << std::sqrt(pos_err/np) << ", " << std::sqrt(rot_err/np) << ", " << solve_time << ", " << iters << "\n";
  
  }
}