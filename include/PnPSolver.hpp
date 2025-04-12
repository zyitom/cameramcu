// Modified by removing ROS dependencies
// Submodule of HeliosRobotSystem
#pragma once

// Standard includes
#include <array>
#include <vector>
#include <iostream>
#include <cmath>

// Eigen
#include <Eigen/Core>
#include <Eigen/Dense>

// OpenCV
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/quaternion.hpp>
#include <opencv2/imgproc.hpp>

// Ceres
#include <ceres/ceres.h>

#include "Armor.hpp"

namespace helios_cv {

// Constants
constexpr double small_armor_width = 135.0;
constexpr double small_armor_height = 55.0;
constexpr double large_armor_width = 225.0;
constexpr double large_armor_height = 55.0;

constexpr double energy_armor_height = 300.0;
constexpr double energy_armor_width_ = 300.0;
constexpr double energy_fan_width_ = 568.48 / 1000.0; // Unit: m

// Define a simple logger replacement for ROS
class Logger {
public:
    Logger(const std::string& name) : name_(name) {}
    
    void debug(const std::string& message) {
        std::cout << "[DEBUG][" << name_ << "]: " << message << std::endl;
    }
    
    void info(const std::string& message) {
        std::cout << "[INFO][" << name_ << "]: " << message << std::endl;
    }
    
    void warn(const std::string& message) {
        std::cout << "[WARN][" << name_ << "]: " << message << std::endl;
    }
    
    void error(const std::string& message) {
        std::cerr << "[ERROR][" << name_ << "]: " << message << std::endl;
    }
    
private:
    std::string name_;
};

// Base information for coordinate transform
struct BaseTransformInfo {
    virtual ~BaseTransformInfo() = default;
};

// Armor specific transform information
struct ArmorTransformInfo : BaseTransformInfo {
    ArmorTransformInfo(cv::Quatd odom2cam, cv::Quatd cam2odom)
        : odom2cam_r(odom2cam), cam2odom_r(cam2odom) {}

    cv::Quatd odom2cam_r;
    cv::Quatd cam2odom_r;
};

// Energy fan transform information
struct EnergyTransformInfo : BaseTransformInfo {
    EnergyTransformInfo(cv::Quatd odom2cam, cv::Quatd cam2odom, double gimbal_yaw)
        : odom2cam_r(odom2cam), cam2odom_r(cam2odom), gimbal_yaw(gimbal_yaw) {}

    cv::Quatd odom2cam_r;
    cv::Quatd cam2odom_r;
    double gimbal_yaw;
};

// Base PnP Solver class
class PnPSolver {
public:
    PnPSolver(const std::array<double, 9>& camera_matrix,
              const std::vector<double>& distortion_coefficients);

    // Get 3d position
    virtual bool solve_pose(const Armor& armor, cv::Mat& rvec, cv::Mat& tvec);

    virtual void update_transform_info(BaseTransformInfo* transform_info) {}

    virtual void draw_projection_points(cv::Mat& image) {}

    // Calculate the distance between armor center and image center
    float calculateDistanceToCenter(const cv::Point2f& image_point);

    virtual void set_use_projection(bool use_projection) = 0;

    virtual bool use_projection() = 0;

protected:
    bool solve_pnp(const Armor& armor, cv::Mat& rvec, cv::Mat& tvec);

    // Four vertices of armor in 3d
    std::vector<cv::Point3f> small_armor_points_;
    std::vector<cv::Point3f> large_armor_points_;
    std::vector<std::vector<cv::Point3f>> energy_fan_points_;
    std::vector<cv::Point2f> image_fans_points_;
    std::vector<cv::Point3f> object_points_;

    std::vector<cv::Point2f> image_armor_points_;
    std::vector<cv::Point3f> energy_armor_points_;

    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;

private:
    Logger logger_{"PnPSolver"};
};

// Armor projection with yaw optimization
class ArmorProjectYaw : public PnPSolver {
public:
    explicit ArmorProjectYaw(const std::array<double, 9>& camera_matrix,
                             const std::vector<double>& dist_coeffs);

    bool solve_pose(const Armor& armor, cv::Mat& rvec, cv::Mat& tvec) override;

    void update_transform_info(BaseTransformInfo* transform_info) override;

    void draw_projection_points(cv::Mat& image) override;

    void set_use_projection(bool use_projection) override;

    bool use_projection() override;

private:
    bool use_projection_ = true;
    
    // Cost function for CERES optimizer
    typedef struct CostFunctor {
        template <typename T>
        bool operator()(const T* const yaw, T* residual) const;
    } CostFunctor;
    
    bool is_transform_info_updated_ = false;

    cv::Matx33d odom2cam_r_;
    cv::Matx33d cam2odom_r_;

    // self pointer to make self pointer accessible in ceres callback
    inline static ArmorProjectYaw* pthis_;

    double diff_function(double yaw);

    [[maybe_unused]] double phi_optimization(double left, double right, double eps);

    void get_rotation_matrix(double yaw, cv::Mat& rotation_mat) const;

    std::vector<cv::Point2f> projected_points_;
    cv::Mat tvec_;
    
    // The pitch and roll of armor are fixed for target
    double roll_ = 0, pitch_ = M_PI/12.0; // 15 degrees in radians
    double armor_angle_;

    Logger logger_{"ArmorProjectYaw"};
};

// Energy target projection with roll optimization
class EnergyProjectRoll : public PnPSolver {
public:
    explicit EnergyProjectRoll(const std::array<double, 9>& camera_matrix,
                               const std::vector<double>& dist_coeffs);

    bool solve_pose(const Armor& armor, cv::Mat& rvec, cv::Mat& tvec) override;

    void draw_projection_points(cv::Mat& image) override;

    void update_transform_info(BaseTransformInfo* transform_info) override;

    void set_use_projection(bool use_projection) override;

    bool use_projection() override;

private:
    bool use_projection_ = true;

    // self pointer to make self pointer accessible in ceres callback
    inline static EnergyProjectRoll* pthis_;
    double yaw_;

    struct CostFunctor {
        template <typename T>
        bool operator()(const T* const roll, T* residual) const;
    };

    bool is_transform_info_updated_ = false;

    cv::Matx33d odom2cam_r_;
    cv::Matx33d cam2odom_r_;

    double pitch_ = 0;
    std::vector<cv::Point2f> projected_points_;
    cv::Mat tvec_;

    double diff_function(double roll);

    void get_rotation_matrix(double roll, cv::Mat& rotation_mat) const;

    Logger logger_{"EnergyProjectRoll"};
};

} // namespace helios_cv