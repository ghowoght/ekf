/**
 * @file ekf2.hpp
 * @author Linfu Wei (ghowoght@qq.com)
 * @brief 
 * @version 1.0
 * @date 2023-04-08
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#ifndef IMU_PROCESS_HPP
#define IMU_PROCESS_HPP

#include <Eigen/Core>
#include <Eigen/Dense>

#include <vector>
#include <queue>
#include <filesystem>
#include <iostream>

#include "so3_math.hpp"
#include "file_helper.hpp"

#define DIM 15 // 状态维度

using V3D = Eigen::Vector3d;
using M3D = Eigen::Matrix3d;
using QD = Eigen::Quaterniond;
using MD_DIM = Eigen::Matrix<double, DIM, DIM>;

struct StateEKF{
    double time;
    QD rot;
    V3D pos;
    V3D vel;
    V3D bg;
    V3D ba;
    StateEKF(){
        time = 0;
        rot = QD::Identity();
        pos = V3D::Zero();
        vel = V3D::Zero();
        bg = V3D::Zero();
        ba = V3D::Zero();
    }
    StateEKF(const StateEKF& s){
        time = s.time;
        rot = s.rot;
        pos = s.pos;
        vel = s.vel;
        bg = s.bg;
        ba = s.ba;
    }
    void operator=(const StateEKF& other){
        time = other.time;
        rot = other.rot;
        pos = other.pos;
        vel = other.vel;
        bg = other.bg;
        ba = other.ba;
    }
};

struct ImuData{
    double time;
    Eigen::Vector3d accel_mpss;
    Eigen::Vector3d gyro_rps;
    ImuData(){
        accel_mpss = Eigen::Vector3d::Zero();
        gyro_rps = Eigen::Vector3d::Zero();
        time = 0;
    }
    ImuData(double time_, Eigen::Vector3d accel_mpss_, Eigen::Vector3d gyro_rps_){
        time = time_;
        accel_mpss = accel_mpss_;
        gyro_rps = gyro_rps_;
    }
    ImuData(const ImuData& other){
        accel_mpss = other.accel_mpss;
        gyro_rps = other.gyro_rps;
        time = other.time;
    }
};

// IMU 和 里程计 数据打包
struct MeasureData{
    std::vector<ImuData> imu_queue;
    std::vector<V3D> odom_queue;
    MeasureData()=default;
    ~MeasureData()=default;
};

struct ModelParam{
    double ARW;                             // 角度随机游走
    double VRW;                             // 速度随机游走
    double gyro_bias_std;                   // 陀螺仪零偏标准差
    double gyro_bias_corr_time;             // 陀螺仪零偏相关时间
    double accel_bias_std;                  // 加速度计零偏标准差
    double accel_bias_corr_time;            // 加速度计零偏相关时间
    V3D odom_std;                           // odom观测噪声
    V3D odom_lever_arm;                     // odom安装杆臂
    V3D imu_mount_angle;                    // IMU安装角度
    M3D r_b_v;                              // 从body系到vehicle系的旋转矩阵

    void init_r_b_v(const V3D& euler_angle){
        r_b_v = SO3Math::euler2dcm(euler_angle);
    }
};

class IMUProcess{
private:
    StateEKF state_last_;
    std::vector<StateEKF> state_queue_;
    bool is_initialized_ = false;
    ModelParam model_param_;
    V3D grav;
    
    // kalman相关
    Eigen::Matrix<double, DIM, 1> delta_x_; // p v phi bg ba g
    MD_DIM Pmat_;

    Eigen::Matrix<double, DIM, 12> Gmat_; // 噪声输入映射矩阵 noise-input mapping matrix
    Eigen::Matrix<double, DIM, 12> Gmat_last_;

    M3D Rmat_;
    Eigen::Matrix<double, Eigen::Dynamic, 1> delta_z_;
    Eigen::Matrix<double, Eigen::Dynamic, DIM> Hmat_;
    Eigen::Matrix<double, DIM, Eigen::Dynamic> Kmat_;

    FileWriterPtr fw_ptr_;

public:
    IMUProcess()=default;
    IMUProcess(const ModelParam& model_param) : model_param_(model_param){
        state_last_ = StateEKF();
        
        delta_x_.setZero();

        // 初始化P
        Pmat_.setIdentity();
        Pmat_(0,0) = Pmat_(1,1) = Pmat_(2,2) = 0.0;
        Pmat_(3,3) = Pmat_(4,4) = Pmat_(5,5) = 1e-9;
        Pmat_(6,6) = Pmat_(7,7) = Pmat_(8,8) = 1e-9;
        Pmat_(9,9) = Pmat_(10,10) = Pmat_(11,11) = 1e-9;
        Pmat_(12,12) = Pmat_(13,13) = Pmat_(14,14) = 1e-9;
        // 初始化R
        Rmat_ = model_param_.odom_std
                .cwiseProduct(model_param_.odom_std)
                .asDiagonal();

        grav = V3D(0, 0, -9.7936);

        // 初始化安装角矩阵
        model_param_.init_r_b_v(model_param_.imu_mount_angle);
        
        fw_ptr_ = FileWriter::create("/home/ghowoght/workspace/ekf/result/result_ekf2.txt");
    }
    ~IMUProcess()=default;

private:
    void forward_propagation(MeasureData& measure){
        state_queue_.clear();
        if(state_queue_.empty()){
            state_queue_.push_back(state_last_);
        }
        auto size = measure.imu_queue.size();
        for(int i = 0; i < size; i++){
            auto imu = measure.imu_queue[i];
            StateEKF state_curr = state_queue_.back();

            double dt = imu.time - state_curr.time;
            state_curr.time = imu.time;

            //////////////// 机械编排 ////////////////
            // 姿态更新
            state_curr.rot = state_curr.rot * SO3Math::Exp(imu.gyro_rps * dt);
            state_curr.rot.normalize();
            // 速度更新
            state_curr.vel += (state_curr.rot * imu.accel_mpss - grav) * dt;
            // 位置更新
            state_curr.pos += state_curr.vel * dt;

            state_queue_.push_back(state_curr);

            //////////////// 噪声传播 ////////////////
            const M3D I_33 = M3D::Identity();
            // 计算状态转移矩阵Φ
            MD_DIM PHImat;
            PHImat.setIdentity();
            // p
            PHImat.block<3, 3>(0,  3) = I_33 * dt;  
            // v
            PHImat.block<3, 3>(3,  6) = -(state_curr.rot * SO3Math::get_skew_symmetric(imu.accel_mpss * dt)); 
            PHImat.block<3, 3>(3, 12) = -state_curr.rot.toRotationMatrix() * dt;
            // phi
            PHImat.block<3, 3>(6,  6) = SO3Math::Exp(-imu.gyro_rps * dt);
            PHImat.block<3, 3>(6,  9) = -SO3Math::J_l(imu.gyro_rps * dt).transpose() * dt;
            // PHImat.block<3, 3>(6,  6) = I_33 + SO3Math::get_skew_symmetric(-imu.gyro_rps * dt);  // 近似
            // PHImat.block<3, 3>(6,  9) = -I_33 * dt;                                              // 近似

            PHImat.block<3, 3>(9, 9) = (1 - dt / model_param_.gyro_bias_corr_time) * I_33;
            PHImat.block<3, 3>(12, 12) = (1 - dt / model_param_.accel_bias_corr_time) * I_33;

            // 计算状态转移噪声协方差矩阵Q
            Eigen::Matrix<double, 12, 12> qmat;
            qmat.setZero();
            double item[] = {   model_param_.VRW * model_param_.VRW,
                                model_param_.ARW * model_param_.ARW,
                                2 * model_param_.gyro_bias_std * model_param_.gyro_bias_std / model_param_.gyro_bias_corr_time,
                                2 * model_param_.accel_bias_std * model_param_.accel_bias_std / model_param_.accel_bias_corr_time,
                            };
            for(int i = 0; i < 4; i++){
                qmat.block<3, 3>(3 * i,  3 * i) = item[i] * M3D::Identity();
            }
            Gmat_.setZero();

            // 梯形积分
            Gmat_.block<3, 3>( 3, 0) = -state_curr.rot.toRotationMatrix();
            Gmat_.block<3, 3>( 6, 3) = -SO3Math::J_l(imu.gyro_rps * dt).transpose();
            // Gmat_.block<3, 3>( 3, 0) = -I_33; // 近似
            // Gmat_.block<3, 3>( 6, 3) = -I_33; // 近似
            Gmat_.block<3, 3>( 9, 6) = I_33;
            Gmat_.block<3, 3>(12, 9) = I_33;
            MD_DIM Qmat = 
                0.5 * dt * (PHImat * Gmat_ * qmat * Gmat_.transpose() + Gmat_ * qmat * Gmat_.transpose() * PHImat.transpose());

            // 状态转移
            MD_DIM P_ = PHImat * Pmat_ * PHImat.transpose() + Qmat;
            Pmat_ = P_;
        }
    }

    void update(const MeasureData& measure){
        // 计算平均速度
        V3D ave_odom = V3D::Zero();
        for(auto& odom : measure.odom_queue){
            ave_odom += odom;
        }
        ave_odom /= measure.odom_queue.size();
        double ave_vel = (ave_odom[0] + ave_odom[1]) / 2.0;

        StateEKF state_curr = state_queue_.back();

        const M3D   r_v_b       = model_param_.r_b_v.transpose();
        const V3D&  lodom       = model_param_.odom_lever_arm;
        const QD&   r_b_n       = state_curr.rot;
        const V3D&  gyro_rps    = measure.imu_queue.back().gyro_rps;
        // 计算观测矩阵H
        Eigen::Matrix<double, 3, DIM> Hmat;
        Hmat.setZero();
        Hmat.block<3, 3>(0, 3) = M3D::Identity();
        Hmat.block<3, 3>(0, 6) = r_b_n * r_v_b * SO3Math::get_skew_symmetric(lodom.cross(gyro_rps));
        Hmat.block<3, 3>(0, 9) = r_b_n * r_v_b * SO3Math::get_skew_symmetric(lodom);

        V3D z_k(ave_vel, 0, 0);
        // 新息
        V3D delta_z = r_b_n * r_v_b * z_k - 
                    (state_curr.vel - r_b_n * r_v_b * lodom.cross(gyro_rps));
        // std::cout << "z_k: " << z_k[0] << " delta_z: " << delta_z.transpose() << std::endl;

        // 卡尔曼增益
        Eigen::Matrix<double, DIM, 3> Kmat = Pmat_ * Hmat.transpose() 
                            * (Hmat * Pmat_ * Hmat.transpose() 
                            + Rmat_).inverse();
                            
        // 更新状态
        delta_x_ = Kmat * (delta_z - Hmat * delta_x_);
        state_curr.pos += delta_x_.block<3, 1>(0, 0);
        state_curr.vel += delta_x_.block<3, 1>(3, 0);
        state_curr.rot = state_curr.rot * SO3Math::Exp(delta_x_.block<3, 1>(6, 0));
        state_curr.rot.normalize();
        state_curr.bg += delta_x_.block<3, 1>(9, 0);
        state_curr.ba += delta_x_.block<3, 1>(12, 0);

        // 状态反馈后，清空delta_x_
        delta_x_.setZero();
        
        // 更新状态协方差矩阵
        Pmat_ = (MD_DIM::Identity() - Kmat * Hmat) * Pmat_;

        state_last_ = state_curr;
    }
    
public:
    bool init_imu(MeasureData& measure){
        const int imu_buffer_size = 1000;
        static int cnt = 0;
        static bool init_flag = false;
        static std::vector<ImuData> imu_queue;
        if(cnt < imu_buffer_size){
            for(auto& imu : measure.imu_queue){
                imu_queue.push_back(imu);
                state_last_.ba += imu.accel_mpss;
                state_last_.bg += imu.gyro_rps;
                cnt++;
            }
        }
        else if(cnt >= imu_buffer_size && !init_flag){
            state_last_.bg /= (double)cnt;
            
            // 加速度计测量值
            const V3D accel = state_last_.ba / (double)cnt;
            
            // 轴线转换为旋转矩阵
            auto axis_to_matrix = [](const V3D& axis, M3D& R){
                R.col(0) = V3D(1, 0, 0);
                R.col(1) = V3D(0, 1, 0);
                R.col(2) = -axis;
                R.col(0) -= R.col(0).dot(R.col(2)) / R.col(2).dot(R.col(2)) * R.col(2);
                R.col(0).normalize();
                R.col(1) -= R.col(1).dot(R.col(2)) / R.col(2).dot(R.col(2)) * R.col(2);
                R.col(1) -= R.col(1).dot(R.col(0)) / R.col(0).dot(R.col(0)) * R.col(0);
                R.col(1).normalize();
            };
            auto accel_normed = accel / accel.norm();
            M3D R;
            axis_to_matrix(accel_normed, R);
            state_last_.rot = R.transpose();
            state_last_.ba = accel - state_last_.rot.conjugate() * grav;

            std::cout << "bg: " << state_last_.bg.transpose() << std::endl;
            std::cout << "ba: " << state_last_.ba.transpose() << std::endl;

            V3D bias = state_last_.rot * (accel - state_last_.ba) - grav;
            init_flag = true;
            std::cout << "bias: " << bias.transpose() << std::endl;

            // 求imu观测方差
            V3D var_accel = V3D::Zero();
            V3D var_gyro = V3D::Zero();
            for(auto& imu : imu_queue){
                var_accel += imu.accel_mpss.cwiseProduct(state_last_.rot * (imu.accel_mpss - state_last_.ba) - grav);
                var_gyro += imu.gyro_rps.cwiseProduct(imu.gyro_rps - state_last_.bg);
            }
            var_accel /= (double)cnt;
            var_gyro /= (double)cnt;
            std::cout << "var_accel: " << var_accel.transpose() << std::endl;
            std::cout << "var_gyro: " << var_gyro.transpose() << std::endl;
            var_accel = var_accel.cwiseSqrt() * 1e5;
            var_gyro = var_gyro.cwiseSqrt() * SO3Math::R2D * 3600;
        }
        return init_flag;
    }
    void process(MeasureData& measure){
        if(!is_initialized_){
            if(init_imu(measure)){
                is_initialized_ = true;
                state_last_.time = measure.imu_queue.back().time;
                std::cout << "init success! " << state_last_.time << std::endl;
            }
            return;
        }
        // 原始数据补偿
        for(auto& imu : measure.imu_queue){
            imu.accel_mpss  -= state_last_.ba;
            imu.gyro_rps    -= state_last_.bg;
        }
        // 前向传播
        forward_propagation(measure);

        // 状态更新
        update(measure);

        // 保存状态
        std::stringstream ss;
        ss << std::fixed << std::setprecision(9) << state_last_.time << " ";
        ss << std::fixed << std::setprecision(9) << state_last_.pos.x() << " ";
        ss << std::fixed << std::setprecision(9) << state_last_.pos.y() << " ";
        ss << std::fixed << std::setprecision(9) << state_last_.pos.z() << " ";
        ss << std::fixed << std::setprecision(9) << state_last_.vel.x() << " ";
        ss << std::fixed << std::setprecision(9) << state_last_.vel.y() << " ";
        ss << std::fixed << std::setprecision(9) << state_last_.vel.z() << " ";
        // 四元数转欧拉角
        V3D euler = SO3Math::quat2euler(state_last_.rot);
        ss << std::fixed << std::setprecision(9) << euler.x() * SO3Math::R2D << " ";
        ss << std::fixed << std::setprecision(9) << euler.y() * SO3Math::R2D << " ";
        ss << std::fixed << std::setprecision(9) << euler.z() * SO3Math::R2D << " ";
        ss << std::endl;
        fw_ptr_->write_txt(ss.str());
    }
};

#endif // IMU_PROCESS_HPP