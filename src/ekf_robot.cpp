/**
 * @file ekf_robot.cpp
 * @author Linfu Wei (ghowoght@qq.com)
 * @brief ADIS16465 + 里程计 测试 (大车数据)
 * @version 2.0
 * @date 2023-04-21
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include "ekf.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>

int main(int argc, char **argv) {

    ModelParam model_param;
    model_param.ARW                  = 0.1 / 60.0 * SO3Math::D2R;  // deg/sqrt(hr)
    model_param.VRW                  = 0.1 / 60.0;        // m/s/sqrt(hr)
    model_param.gyro_bias_std        = 50 * SO3Math::D2R / 3600.0; // deg/hr
    model_param.gyro_bias_corr_time  = 1 * 3600.0;
    model_param.accel_bias_std       = 200 * 1e-5; // mGal 1mGal=1e-5m/s^2
    model_param.accel_bias_corr_time = 1 * 3600.0;

    model_param.odom_std << 0.03, 0.05, 0.05; // odom观测噪声标准差

    model_param.odom_lever_arm << 0, 0, 1.099; // m
    model_param.imu_mount_angle << 0, 0.3, 0.31; // roll, pitch, yaw (deg)
    model_param.imu_mount_angle *= SO3Math::D2R;

    IMUProcess imu_process(model_param);

    ImuData imudata;
    V3D odomdata = V3D::Zero();
    MeasureData measure;
    
    std::ifstream data_fs(
        "/home/ghowoght/workspace/ekf/data/HL_INSPROBE_9_VEL_IMU_ODO.txt");

    data_fs >> imudata.time
            >> imudata.gyro_rps[0] >> imudata.gyro_rps[1] >> imudata.gyro_rps[2]
            >> imudata.accel_mpss[0] >> imudata.accel_mpss[1] >> imudata.accel_mpss[2]
            >> odomdata[0] >> odomdata[1];

    double last_time = imudata.time;
    int cnt = 0;
    while (!data_fs.eof()){
        data_fs >> imudata.time
                >> imudata.gyro_rps[0] >> imudata.gyro_rps[1] >> imudata.gyro_rps[2]
                >> imudata.accel_mpss[0] >> imudata.accel_mpss[1] >> imudata.accel_mpss[2]
                >> odomdata[0] >> odomdata[1];

        measure.imu_queue.push_back(imudata);
        measure.odom_queue.push_back(odomdata);
        
        if(imudata.time - last_time > 0.1)
        {
            last_time = imudata.time;
            imu_process.process(measure);
            measure.imu_queue.clear();
            measure.odom_queue.clear();
        }

    }
}