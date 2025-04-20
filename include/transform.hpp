#include <opencv2/core/mat.hpp>
#include <opencv2/core/quaternion.hpp>
#include <opencv2/calib3d.hpp>
#include <mutex>

class SimpleTransformManager {
private:
    cv::Quatd cam2odom_rotation;
    cv::Vec3d cam2odom_translation;
    cv::Quatd odom2cam_rotation;
    cv::Vec3d odom2cam_translation;
    std::mutex mutex;
    
    // Helper method to rotate a vector using quaternion
    cv::Vec3d rotateVector(const cv::Quatd& q, const cv::Vec3d& v) {
        // Implement quaternion rotation: q * v * q^-1
        double w = q.w;
        cv::Vec3d u(q.x, q.y, q.z);
        
        // Formula: v' = v + 2 * cross(u, cross(u, v) + w*v)
        cv::Vec3d uCrossV = u.cross(v);
        cv::Vec3d term = u.cross(uCrossV + w*v);
        return v + 2.0 * term;
    }
    
    // Helper method to convert Euler angles to quaternion
    cv::Quatd eulerToQuaternion(double roll, double pitch, double yaw) {
        // Convert Euler angles to quaternion using the ZYX convention
        double cy = cos(yaw * 0.5);
        double sy = sin(yaw * 0.5);
        double cp = cos(pitch * 0.5);
        double sp = sin(pitch * 0.5);
        double cr = cos(roll * 0.5);
        double sr = sin(roll * 0.5);
        
        cv::Quatd q;
        q.w = cr * cp * cy + sr * sp * sy;
        q.x = sr * cp * cy - cr * sp * sy;
        q.y = cr * sp * cy + sr * cp * sy;
        q.z = cr * cp * sy - sr * sp * cy;
        
        return q;
    }
    
    // Helper method to convert rotation matrix to quaternion
    cv::Quatd matrixToQuaternion(const cv::Mat& rotMatrix) {
        // Ensure we have a proper rotation matrix
        CV_Assert(rotMatrix.type() == CV_64F && rotMatrix.rows == 3 && rotMatrix.cols == 3);
        
        // Using the algorithm from OpenCV's own implementation
        double trace = rotMatrix.at<double>(0,0) + rotMatrix.at<double>(1,1) + rotMatrix.at<double>(2,2);
        cv::Quatd q;
        
        if (trace > 0) {
            double s = 0.5 / sqrt(trace + 1.0);
            q.w = 0.25 / s;
            q.x = (rotMatrix.at<double>(2,1) - rotMatrix.at<double>(1,2)) * s;
            q.y = (rotMatrix.at<double>(0,2) - rotMatrix.at<double>(2,0)) * s;
            q.z = (rotMatrix.at<double>(1,0) - rotMatrix.at<double>(0,1)) * s;
        } else {
            if (rotMatrix.at<double>(0,0) > rotMatrix.at<double>(1,1) && 
                rotMatrix.at<double>(0,0) > rotMatrix.at<double>(2,2)) {
                double s = 2.0 * sqrt(1.0 + rotMatrix.at<double>(0,0) - 
                                      rotMatrix.at<double>(1,1) - rotMatrix.at<double>(2,2));
                q.w = (rotMatrix.at<double>(2,1) - rotMatrix.at<double>(1,2)) / s;
                q.x = 0.25 * s;
                q.y = (rotMatrix.at<double>(0,1) + rotMatrix.at<double>(1,0)) / s;
                q.z = (rotMatrix.at<double>(0,2) + rotMatrix.at<double>(2,0)) / s;
            } else if (rotMatrix.at<double>(1,1) > rotMatrix.at<double>(2,2)) {
                double s = 2.0 * sqrt(1.0 + rotMatrix.at<double>(1,1) - 
                                      rotMatrix.at<double>(0,0) - rotMatrix.at<double>(2,2));
                q.w = (rotMatrix.at<double>(0,2) - rotMatrix.at<double>(2,0)) / s;
                q.x = (rotMatrix.at<double>(0,1) + rotMatrix.at<double>(1,0)) / s;
                q.y = 0.25 * s;
                q.z = (rotMatrix.at<double>(1,2) + rotMatrix.at<double>(2,1)) / s;
            } else {
                double s = 2.0 * sqrt(1.0 + rotMatrix.at<double>(2,2) - 
                                      rotMatrix.at<double>(0,0) - rotMatrix.at<double>(1,1));
                q.w = (rotMatrix.at<double>(1,0) - rotMatrix.at<double>(0,1)) / s;
                q.x = (rotMatrix.at<double>(0,2) + rotMatrix.at<double>(2,0)) / s;
                q.y = (rotMatrix.at<double>(1,2) + rotMatrix.at<double>(2,1)) / s;
                q.z = 0.25 * s;
            }
        }
        
        return q;
    }

public:
    SimpleTransformManager() {
        resetTransforms();
    }

    void resetTransforms() {
        std::lock_guard<std::mutex> lock(mutex);
        cam2odom_rotation = cv::Quatd(1, 0, 0, 0);  // Identity quaternion
        cam2odom_translation = cv::Vec3d(0, 0, 0);
        odom2cam_rotation = cv::Quatd(1, 0, 0, 0);  // Identity quaternion
        odom2cam_translation = cv::Vec3d(0, 0, 0);
    }

    // 更新变换，基于当前的偏航角和俯仰角
    void updateTransforms(double yaw, double pitch) {
        std::lock_guard<std::mutex> lock(mutex);
        
        // 固定的相机到pitch_link的变换 (来自URDF中的camera_joint)
        cv::Vec3d camera_translation(0.125, 0.0, -0.035);
        cv::Quatd camera_rotation(1, 0, 0, 0); // Identity quaternion
        
        // 固定的相机光学坐标系到相机的变换 (来自URDF中的camera_optical_joint)
        cv::Quatd optical_rotation = eulerToQuaternion(-M_PI/2, 0, -M_PI/2);
        
        // 可变的yaw变换 (odoom到yaw_link)
        cv::Quatd yaw_rotation = eulerToQuaternion(0, 0, -yaw);
        
        // 可变的pitch变换 (yaw_link到pitch_link)
        cv::Quatd pitch_rotation = eulerToQuaternion(0, -pitch, 0);
        
        // 计算完整的变换链：optical_frame -> camera -> pitch -> yaw -> odom
        // 注意：旋转的组合顺序是反的，因为我们是从相机到odom
        cam2odom_rotation = yaw_rotation * pitch_rotation * camera_rotation * optical_rotation;
        
        // 对于平移，先执行旋转再加上平移
        cv::Vec3d rotated_camera_translation = rotateVector(pitch_rotation, camera_translation);
        cam2odom_translation = rotated_camera_translation;  // 实际情况下可能还需要加上yaw和pitch的平移偏移
        
        // 计算逆变换：odom -> optical_frame
        odom2cam_rotation.w = cam2odom_rotation.w;
        odom2cam_rotation.x = -cam2odom_rotation.x;
        odom2cam_rotation.y = -cam2odom_rotation.y;
        odom2cam_rotation.z = -cam2odom_rotation.z;
        
        // v' = -(q^-1 * v * q)
        odom2cam_translation = -rotateVector(odom2cam_rotation, cam2odom_translation);
    }

    // 获取相机到odom的变换
    void getCam2Odom(cv::Quatd& rotation, cv::Vec3d& translation) {
        std::lock_guard<std::mutex> lock(mutex);
        rotation = cam2odom_rotation;
        translation = cam2odom_translation;
    }

    // 获取odom到相机的变换
    void getOdom2Cam(cv::Quatd& rotation, cv::Vec3d& translation) {
        std::lock_guard<std::mutex> lock(mutex);
        rotation = odom2cam_rotation;
        translation = odom2cam_translation;
    }
};