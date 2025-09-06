cmake_minimum_required(VERSION 3.5)
project(gp_spline)

add_compile_options(-std=c++17 -fvisibility=hidden)
set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_FLAGS "-fopenmp")

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(GTest REQUIRED)
find_package(Ceres REQUIRED)
find_package(yaml-cpp REQUIRED)

set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.45.0 COMPONENTS filesystem REQUIRED)

include_directories(include
  ${catkin_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIRS})

add_executable(lie_spline_jac_test
  src/test/lie_spline_jac_test.cpp)
target_link_libraries(lie_spline_jac_test
  ${GTEST_LIBRARIES} 
  ${GTEST_MAIN_LIBRARIES} 
  pthread)

add_executable(lie_gp_jac_test
  src/test/lie_gp_jac_test.cpp)
target_link_libraries(lie_gp_jac_test
  ${GTEST_LIBRARIES} 
  ${GTEST_MAIN_LIBRARIES} 
  pthread)

add_executable(rn_gp_resid_test
  src/test/rn_gp_resid_test.cpp)
target_link_libraries(rn_gp_resid_test
  ${GTEST_LIBRARIES} 
  ${GTEST_MAIN_LIBRARIES} 
  pthread)

add_executable(rn_spline_motion_prior_jac_test
  src/test/rn_spline_motion_prior_jac_test.cpp)
target_link_libraries(rn_spline_motion_prior_jac_test
  ${GTEST_LIBRARIES} 
  ${GTEST_MAIN_LIBRARIES} 
  pthread)

add_executable(lie_spline_motion_prior_jac_test
  src/test/lie_spline_motion_prior_jac_test.cpp)
target_link_libraries(lie_spline_motion_prior_jac_test
  ${GTEST_LIBRARIES} 
  ${GTEST_MAIN_LIBRARIES} 
  pthread)

add_executable(lie_gp_test
  src/test/lie_gp_test.cpp)

add_executable(lie_spline_motion_prior_opt_test
  src/test/lie_spline_motion_prior_opt_test.cpp)
target_link_libraries(lie_spline_motion_prior_opt_test
  ${CERES_LIBRARIES})

add_executable(sim
  src/sim/sim.cpp
  src/estimator/estimator_base.cpp
  src/estimator/gp_se3_estimator.cpp
  src/estimator/gp_so3xr3_estimator.cpp
  src/utils/line.cpp
  src/utils/sinusoid.cpp)
target_link_libraries(sim
  ${CERES_LIBRARIES}
  ${Boost_LIBRARIES}
  ${YAML_CPP_LIBRARIES})

add_executable(hw
  src/hw/hw.cpp
  src/estimator/estimator_base.cpp
  src/estimator/gp_se3_estimator.cpp
  src/estimator/gp_so3xr3_estimator.cpp)
target_link_libraries(hw
  ${CERES_LIBRARIES}
  ${Boost_LIBRARIES}
  ${YAML_CPP_LIBRARIES})

add_executable(linear_sim
  src/linear_example/linear_sim.cpp
  src/utils/sinusoid.cpp)
target_link_libraries(linear_sim
  ${CERES_LIBRARIES}
  ${Boost_LIBRARIES}
  ${YAML_CPP_LIBRARIES})