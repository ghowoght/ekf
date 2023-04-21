cmake_minimum_required(VERSION 3.0.4)
project(EKF)

add_compile_options(-std=c++17)
# Release build
set(CMAKE_BUILD_TYPE Release)

find_package(Eigen3)

include_directories(
    include
    ${EIGEN3_INCLUDE_DIR}
)

add_executable(ekf_robot src/ekf_robot.cpp)