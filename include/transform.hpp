#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/quaternion.hpp>
#include <mutex>
#include <memory>
#include "PnPSolver.hpp"
namespace helios_cv {

class TransformManager {
public:
    static TransformManager& getInstance() {
        static TransformManager instance;
        return instance;
    }


    void updateGimbalAngles(double yaw, double pitch, double roll = 0.0) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        current_yaw_ = yaw;
        current_pitch_ = pitch;
        current_roll_ = roll;
        
        updateTransforms();
    }
    
    // 设置相机安装参数 - 从URDF的camera_joint获取
    void setCameraParams(
        const Eigen::Vector3d& camera_translation = Eigen::Vector3d(0.125, 0, -0.035),
        const Eigen::Vector3d& camera_rpy = Eigen::Vector3d(0, 0, 0)
    ) {
        std::lock_guard<std::mutex> lock(mutex_);
        camera_translation_ = camera_translation;
        camera_rpy_ = camera_rpy;
        updateTransforms();
    }
    
    // 获取从世界坐标系(odoom)到相机光学坐标系的旋转
    cv::Quatd getOdom2CamRotation() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return odom2cam_rotation_;
    }
    
    // 获取从相机光学坐标系到世界坐标系(odoom)的旋转
    cv::Quatd getCam2OdomRotation() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return cam2odom_rotation_;
    }
    
    // 获取云台偏航角
    double getGimbalYaw() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return current_yaw_;
    }
    
    // 获取云台俯仰角
    double getGimbalPitch() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return current_pitch_;
    }
    
    // 获取云台横滚角
    double getGimbalRoll() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return current_roll_;
    }

private:
    TransformManager() {
        // 初始化变换
        updateTransforms();
    }
    

    void updateTransforms() {
        // ROS是Z轴向上，X轴向前，Y轴向左
        Eigen::AngleAxisd yaw_rotation(current_yaw_, Eigen::Vector3d::UnitZ());
        Eigen::AngleAxisd pitch_rotation(current_pitch_, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd roll_rotation(current_roll_, Eigen::Vector3d::UnitX());
        
        // 世界到云台的变换
        Eigen::Quaterniond world_to_gimbal = yaw_rotation * pitch_rotation * roll_rotation;
        
        // 云台到相机的变换
        Eigen::AngleAxisd camera_yaw(camera_rpy_[2], Eigen::Vector3d::UnitZ());
        Eigen::AngleAxisd camera_pitch(camera_rpy_[1], Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd camera_roll(camera_rpy_[0], Eigen::Vector3d::UnitX());
        Eigen::Quaterniond gimbal_to_camera = camera_yaw * camera_pitch * camera_roll;
        
        // 相机到光学坐标系的变换
        Eigen::Quaterniond camera_to_optical = 
            Eigen::AngleAxisd(-M_PI/2, Eigen::Vector3d::UnitZ()) * 
            Eigen::AngleAxisd(-M_PI/2, Eigen::Vector3d::UnitX());
        
        // 计算世界到光学坐标系的完整变换链
        Eigen::Quaterniond world_to_optical = gimbal_to_camera * world_to_gimbal;
        world_to_optical = camera_to_optical * world_to_optical;
        
        // 确保四元数是标准化的
        world_to_optical.normalize();
        
        // 设置odom2cam四元数
        odom2cam_rotation_ = cv::Quatd(
            world_to_optical.w(), 
            world_to_optical.x(), 
            world_to_optical.y(), 
            world_to_optical.z()
        );
        
        // 光学坐标系到世界的变换
        Eigen::Quaterniond optical_to_world = world_to_optical.conjugate();
        
        // 设置cam2odom四元数
        cam2odom_rotation_ = cv::Quatd(
            optical_to_world.w(),
            optical_to_world.x(),
            optical_to_world.y(),
            optical_to_world.z()
        );
    }
    
    
    double current_yaw_ = 0.0;
    double current_pitch_ = 0.0;
    double current_roll_ = 0.0;
    
    
    Eigen::Vector3d camera_translation_{0.125, 0, -0.035};  // camera_joint xyz
    Eigen::Vector3d camera_rpy_{0, 0, 0};                  // camera_joint rpy
    
    // 坐标变换四元数
    cv::Quatd odom2cam_rotation_;
    cv::Quatd cam2odom_rotation_;
    

    mutable std::mutex mutex_;
};


inline std::shared_ptr<ArmorTransformInfo> createArmorTransformInfo() {
    auto& transform_manager = TransformManager::getInstance();
    return std::make_shared<ArmorTransformInfo>(
        transform_manager.getOdom2CamRotation(),
        transform_manager.getCam2OdomRotation()
    );
}


inline std::shared_ptr<EnergyTransformInfo> createEnergyTransformInfo() {
    auto& transform_manager = TransformManager::getInstance();
    return std::make_shared<EnergyTransformInfo>(
        transform_manager.getOdom2CamRotation(),
        transform_manager.getCam2OdomRotation(),
        transform_manager.getGimbalYaw()
    );
}

} // namespace helios_cv