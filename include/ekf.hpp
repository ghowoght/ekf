/**
 * @file ekf.hpp
 * @author Linfu Wei (ghowoght@qq.com)
 * @brief IMU & 里程计松组合(右乘模型)
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
using MD_DIM = Eigen::Matrix<double, DIM, DIM>;

struct StateEKF{
    double time;
    M3D rot;
    V3D pos;
    V3D vel;
    V3D bg;
    V3D ba;
    StateEKF(){
        time = 0;
        rot = M3D::Identity();
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
    double accel_bias_std;                  // 加速度计零偏标准差
    double corr_time;                       // 零偏相关时间
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

    MD_DIM PHImat_;
    Eigen::Matrix<double, DIM, 12> Gmat_; // 噪声输入映射矩阵 noise-input mapping matrix
    Eigen::Matrix<double, 12, 12> qmat_;

    M3D Rmat_;
    Eigen::Matrix<double, Eigen::Dynamic, 1> delta_z_;
    Eigen::Matrix<double, Eigen::Dynamic, DIM> Hmat_;
    Eigen::Matrix<double, DIM, Eigen::Dynamic> Kmat_;

    FileWriterPtr fw_ptr_;

public:
    IMUProcess()=default;
    IMUProcess(const ModelParam& model_param) : model_param_(model_param){
        state_last_ = StateEKF();
        // 误差状态置零        
        delta_x_.setZero();
        // 初始化P
        Pmat_.setIdentity();
        Pmat_(0,0) = Pmat_(1,1) = Pmat_(2,2) = 0.0;
        Pmat_(3,3) = Pmat_(4,4) = Pmat_(5,5) = 1e-9;
        Pmat_(6,6) = Pmat_(7,7) = Pmat_(8,8) = 1e-9;
        Pmat_(9,9) = Pmat_(10,10) = Pmat_(11,11) = 1e-9;
        Pmat_(12,12) = Pmat_(13,13) = Pmat_(14,14) = 1e-9;
        // 初始化q
        qmat_.setZero();
        double item[] = {   model_param_.VRW * model_param_.VRW,
                            model_param_.ARW * model_param_.ARW,
                            2 * model_param_.gyro_bias_std * model_param_.gyro_bias_std / model_param_.corr_time,
                            2 * model_param_.accel_bias_std * model_param_.accel_bias_std / model_param_.corr_time,
                        };
        for(int i = 0; i < 4; i++){
            qmat_.block<3, 3>(3 * i,  3 * i) = item[i] * M3D::Identity();
        }

        // 初始化R
        Rmat_ = model_param_.odom_std
                .cwiseProduct(model_param_.odom_std)
                .asDiagonal();

        grav = V3D(0, 0, -9.7936);

        // 初始化安装角矩阵
        model_param_.init_r_b_v(model_param_.imu_mount_angle);
        
        fw_ptr_ = FileWriter::create("/home/ghowoght/workspace/ekf_plus/result/result_ekf2.txt");
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
            // 速度更新
            state_curr.vel += (state_curr.rot * imu.accel_mpss - grav) * dt;
            // 位置更新
            state_curr.pos += state_curr.vel * dt;

            state_queue_.push_back(state_curr);

            //////////////// 噪声传播 ////////////////
            const M3D I_33 = M3D::Identity();
            // 计算状态转移矩阵Φ
            PHImat_.setIdentity();
            // p
            PHImat_.block<3, 3>(0,  3) = I_33 * dt;  
            // v
            PHImat_.block<3, 3>(3,  6) = -(state_curr.rot * SO3Math::get_skew_symmetric(imu.accel_mpss * dt)); 
            PHImat_.block<3, 3>(3, 12) = -state_curr.rot * dt;
            // phi
            // PHImat_.block<3, 3>(6,  6) = SO3Math::Exp(-imu.gyro_rps * dt);
            // PHImat_.block<3, 3>(6,  9) = -SO3Math::J_l(imu.gyro_rps * dt).transpose() * dt;
            PHImat_.block<3, 3>(6,  6) = I_33 + SO3Math::get_skew_symmetric(-imu.gyro_rps * dt);  // 近似
            PHImat_.block<3, 3>(6,  9) = -I_33 * dt;                                              // 近似

            PHImat_.block<3, 3>(9, 9) = (1 - dt / model_param_.corr_time) * I_33;
            PHImat_.block<3, 3>(12, 12) = (1 - dt / model_param_.corr_time) * I_33;

            // 计算状态转移噪声协方差矩阵Q
            Gmat_.setZero();
            // Gmat_.block<3, 3>( 3, 0) = -state_curr.rot.toRotationMatrix();
            // Gmat_.block<3, 3>( 6, 3) = -SO3Math::J_l(imu.gyro_rps * dt).transpose();
            Gmat_.block<3, 3>( 3, 0) = -I_33; // 近似
            Gmat_.block<3, 3>( 6, 3) = -I_33; // 近似
            Gmat_.block<3, 3>( 9, 6) = I_33;
            Gmat_.block<3, 3>(12, 9) = I_33;
            // 梯形积分
            MD_DIM Qmat = 
                0.5 * dt * (PHImat_ * Gmat_ * qmat_ * Gmat_.transpose() 
                + Gmat_ * qmat_ * Gmat_.transpose() * PHImat_.transpose());

            // 状态转移
            Pmat_ = PHImat_ * Pmat_ * PHImat_.transpose() + Qmat;
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
        const M3D&   r_b_n       = state_curr.rot;
        const V3D&  gyro_rps    = measure.imu_queue.back().gyro_rps;
        // 计算观测矩阵H
        Eigen::Matrix<double, 3, DIM> Hmat;
        Hmat.setZero();
        Hmat.block<3, 3>(0, 3) = M3D::Identity();
        Hmat.block<3, 3>(0, 6) = r_b_n * SO3Math::get_skew_symmetric(lodom.cross(gyro_rps));
        Hmat.block<3, 3>(0, 9) = r_b_n * SO3Math::get_skew_symmetric(lodom);

        V3D vel_v(ave_vel, 0, 0); // v系速度观测
        // 新息
        V3D delta_z = r_b_n * r_v_b * vel_v - 
                    (state_curr.vel - r_b_n * lodom.cross(gyro_rps));

        // 卡尔曼增益
        Eigen::Matrix<double, DIM, 3> Kmat = Pmat_ * Hmat.transpose() 
                            * (Hmat * Pmat_ * Hmat.transpose() 
                            + Rmat_).inverse();
        // 更新误差状态
        delta_x_ = Kmat * (delta_z - Hmat * delta_x_);
        // 状态反馈
        state_curr.pos += delta_x_.block<3, 1>(0, 0);
        state_curr.vel += delta_x_.block<3, 1>(3, 0);
        state_curr.rot = state_curr.rot * SO3Math::Exp(delta_x_.block<3, 1>(6, 0));
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
        const int imu_buffer_size = 100;
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
            auto axis_to_matrix = [](const V3D& axis){
                M3D R;
                R.col(0) = V3D(1, 0, 0);
                R.col(1) = V3D(0, 1, 0);
                R.col(2) = -axis;
                R.col(0) -= R.col(0).dot(R.col(2)) / R.col(2).dot(R.col(2)) * R.col(2);
                R.col(0).normalize();
                R.col(1) -= R.col(1).dot(R.col(2)) / R.col(2).dot(R.col(2)) * R.col(2);
                R.col(1) -= R.col(1).dot(R.col(0)) / R.col(0).dot(R.col(0)) * R.col(0);
                R.col(1).normalize();
                return R;
            };
            // 使用归一化的加速度计测量值作为重力方向
            M3D R = axis_to_matrix(accel / accel.norm());
            state_last_.rot = R.transpose();
            state_last_.ba = accel - state_last_.rot.transpose() * grav;

            std::cout << "bg: " << state_last_.bg.transpose() << std::endl;
            std::cout << "ba: " << state_last_.ba.transpose() << std::endl;

            init_flag = true;
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
        Eigen::Quaterniond q(state_last_.rot);
        ss << std::fixed << std::setprecision(9) << q.x() << " ";
        ss << std::fixed << std::setprecision(9) << q.y() << " ";
        ss << std::fixed << std::setprecision(9) << q.z() << " ";
        ss << std::fixed << std::setprecision(9) << q.w() << " ";
        ss << std::endl;
        fw_ptr_->write_txt(ss.str());

        // for(auto& state : state_queue_){
        //     std::cout << "time: " << state.time << " ";
        //     std::cout << "pos: " << state.pos.transpose() << " ";
        //     std::cout << std::endl;
        // }
    }
};

#endif // IMU_PROCESS_HPP