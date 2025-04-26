#include <opencv2/core/mat.hpp>
#include <opencv2/core/quaternion.hpp>
#include <map>
#include <mutex>
#include <algorithm>
#include <chrono>
#include <iostream>

class TimestampedTransformManager {
private:
    struct TransformData {
        double yaw;
        double pitch;
        cv::Quatd cam2odom_rotation;
        cv::Vec3d cam2odom_translation;
        cv::Quatd odom2cam_rotation;
        cv::Vec3d odom2cam_translation;
    };
    
    // Store timestamp (nanoseconds) -> transform data
    std::map<int64_t, TransformData> transform_cache;
    mutable std::mutex mutex; // Make mutex mutable so it can be locked in const methods
    
    // Maximum cache size - set to 10 as requested
    const size_t MAX_CACHE_SIZE = 60;
    
    // Helper method to rotate a vector using quaternion
    cv::Vec3d rotateVector(const cv::Quatd& q, const cv::Vec3d& v) const {
        double w = q.w;
        cv::Vec3d u(q.x, q.y, q.z);
        
        // Formula: v' = v + 2 * cross(u, cross(u, v) + w*v)
        cv::Vec3d uCrossV = u.cross(v);
        cv::Vec3d term = u.cross(uCrossV + w*v);
        return v + 2.0 * term;
    }
    
    // Helper method to convert Euler angles to quaternion
    cv::Quatd eulerToQuaternion(double roll, double pitch, double yaw) const {
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
    
    // Calculate and store transforms for given yaw and pitch
    void calculateTransforms(TransformData& data) {
        // Fixed camera to pitch_link transform (from URDF)
        cv::Vec3d camera_translation(0.125, 0.0, -0.035);
        cv::Quatd camera_rotation(1, 0, 0, 0); // Identity quaternion
        
        // Fixed optical frame to camera transform (from URDF)
        cv::Quatd optical_rotation = eulerToQuaternion(-M_PI/2, 0, -M_PI/2);
        
        // Variable yaw transform (odoom to yaw_link)
        cv::Quatd yaw_rotation = eulerToQuaternion(0, 0, -data.yaw);
        
        // Variable pitch transform (yaw_link to pitch_link)
        cv::Quatd pitch_rotation = eulerToQuaternion(0, -data.pitch, 0);
        
        // Complete transform chain: optical_frame -> camera -> pitch -> yaw -> odom
        data.cam2odom_rotation = yaw_rotation * pitch_rotation * camera_rotation * optical_rotation;
        
        // For translation, rotate first then add translation
        cv::Vec3d rotated_camera_translation = rotateVector(pitch_rotation, camera_translation);
        data.cam2odom_translation = rotated_camera_translation;
        
        // Calculate inverse transform: odom -> optical_frame
        data.odom2cam_rotation.w = data.cam2odom_rotation.w;
        data.odom2cam_rotation.x = -data.cam2odom_rotation.x;
        data.odom2cam_rotation.y = -data.cam2odom_rotation.y;
        data.odom2cam_rotation.z = -data.cam2odom_rotation.z;
        
        data.odom2cam_translation = -rotateVector(data.odom2cam_rotation, data.cam2odom_translation);
    }
    
    // Clean up old entries from the cache
    void cleanupCache() {
        // Keep only the MAX_CACHE_SIZE most recent entries
        if (transform_cache.size() > MAX_CACHE_SIZE) {
            size_t to_remove = transform_cache.size() - MAX_CACHE_SIZE;
            auto it = transform_cache.begin();
            for (size_t i = 0; i < to_remove && it != transform_cache.end(); ++i) {
                it = transform_cache.erase(it);
            }
        }
    }

public:
    TimestampedTransformManager() {}
    
    // Update the transform cache with new data
    void updateTransform(int64_t timestamp_ns, double yaw, double pitch) {
        std::lock_guard<std::mutex> lock(mutex);
        
        // Create and calculate new transform data
        TransformData data;
        data.yaw = yaw;
        data.pitch = pitch;
        calculateTransforms(data);
        
        // Store in cache
        transform_cache[timestamp_ns] = data;
        
        // Clean up cache to maintain size limit
        cleanupCache();
    }
    

    bool getCam2Odom(int64_t timestamp_ns, cv::Quatd& rotation, cv::Vec3d& translation) {
        std::lock_guard<std::mutex> lock(mutex);
        
        if (transform_cache.empty()) {
            std::cout << "Error: Transform cache is empty. No transforms available." << std::endl;
            return false;
        }
    
        auto exact_match = transform_cache.find(timestamp_ns);
        if (exact_match != transform_cache.end()) {
            rotation = exact_match->second.cam2odom_rotation;
            translation = exact_match->second.cam2odom_translation;
            return true;
        }
        
        std::cout << "Error: No exact transform match for timestamp " << timestamp_ns 
                << " nanoseconds. Aborting transform lookup." << std::endl;
        
        return false;
    }


    bool getOdom2Cam(int64_t timestamp_ns, cv::Quatd& rotation, cv::Vec3d& translation) {
        std::lock_guard<std::mutex> lock(mutex);
        
        if (transform_cache.empty()) {
            std::cout << "Error: Transform cache is empty. No transforms available." << std::endl;
            return false;
        }
        
     
        auto exact_match = transform_cache.find(timestamp_ns);
        if (exact_match != transform_cache.end()) {
            rotation = exact_match->second.odom2cam_rotation;
            translation = exact_match->second.odom2cam_translation;
            return true;
        }
        
        std::cout << "Error: No exact transform match for timestamp " << timestamp_ns 
                << " nanoseconds. Aborting transform lookup." << std::endl;
        
        return false;
    }
    

    bool getTransforms(int64_t timestamp_ns, 
                    cv::Quatd& cam2odom_rotation, cv::Vec3d& cam2odom_translation,
                    cv::Quatd& odom2cam_rotation, cv::Vec3d& odom2cam_translation) {
        std::lock_guard<std::mutex> lock(mutex);
        
        if (transform_cache.empty()) {
            std::cout << "Error: Transform cache is empty. No transforms available." << std::endl;
            return false;
        }
        

        auto exact_match = transform_cache.find(timestamp_ns);
        if (exact_match != transform_cache.end()) {
            cam2odom_rotation = exact_match->second.cam2odom_rotation;
            cam2odom_translation = exact_match->second.cam2odom_translation;
            odom2cam_rotation = exact_match->second.odom2cam_rotation;
            odom2cam_translation = exact_match->second.odom2cam_translation;
            return true;
        }
        

        std::cout << "Error: No exact transform match for timestamp " << timestamp_ns 
                << " nanoseconds. Aborting transform lookup." << std::endl;
        

        if (!transform_cache.empty()) {
            auto first = transform_cache.begin();
            auto last = std::prev(transform_cache.end());
            std::cout << "Available timestamp range: " << first->first 
                    << " to " << last->first << " nanoseconds" << std::endl;
        }
        
        return false;
    }
    
    // Get the size of the cache
    size_t cacheSize() const {
        std::lock_guard<std::mutex> lock(mutex); // This works now with mutable mutex
        return transform_cache.size();
    }
    
    // Get the timestamp range in the cache
    bool getTimestampRange(int64_t& oldest_ts, int64_t& newest_ts) const {
        std::lock_guard<std::mutex> lock(mutex); // This works now with mutable mutex
        if (transform_cache.empty()) {
            return false;
        }
        
        oldest_ts = transform_cache.begin()->first;
        newest_ts = transform_cache.rbegin()->first;
        return true;
    }
};