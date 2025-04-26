#include <opencv2/core/mat.hpp>
#include <opencv2/core/quaternion.hpp>
#define BOOST_BIND_GLOBAL_PLACEHOLDERS 
#include <serial/serial.h>
#include "Protocol.hpp"
#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <memory>
#include <atomic>
#include <cmath>
#include "hikvision_camera.h"
#include "ovdetector.hpp"
#include "optional"
#include "Serial.hpp"
#include <boost/thread/mutex.hpp>
#include <boost/thread/lock_guard.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/lockfree/queue.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/thread/thread.hpp>
#include <boost/bind.hpp>
#include <boost/asio.hpp>
#include <boost/asio/serial_port.hpp>
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <future> 
#include <fcntl.h>  
#include "PnPSolver.hpp"
#include "transform.hpp"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <rclcpp/rclcpp.hpp>
#include <signal.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include "autoaim_interfaces/msg/armor.hpp"
#include "autoaim_interfaces/msg/armors.hpp"
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

std::shared_ptr<tf2_ros::Buffer> tf2_buffer;
std::shared_ptr<tf2_ros::TransformListener> tf2_listener;
rclcpp::Publisher<autoaim_interfaces::msg::Armors>::SharedPtr armors_pub_;
rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
visualization_msgs::msg::Marker armor_marker_;
visualization_msgs::msg::Marker text_marker_;
std::string node_namespace_;
std::unique_ptr<TimestampedTransformManager> transform_manager = 
    std::make_unique<TimestampedTransformManager>();
std::atomic<bool> sync_started(false);
std::atomic<uint64_t> sync_frame_id(0);
std::atomic<uint64_t> sync_gyro_id(0);
std::atomic<bool> ready_for_sync(false);
std::mutex latest_gyro_mutex;
helios::MCUPacket latest_gyro_data;
std::chrono::high_resolution_clock::time_point latest_gyro_timestamp;

bool has_new_gyro_data = false;

std::atomic<bool> first_serial_received(false);
std::mutex initial_sync_mutex;
std::condition_variable initial_sync_cv;

// Modified update_latest_gyro_data function to signal first data arrival
void update_latest_gyro_data(const helios::MCUPacket& gyro_data) {
    {
        std::lock_guard<std::mutex> lock(latest_gyro_mutex);
        latest_gyro_data = gyro_data;
        latest_gyro_timestamp = std::chrono::high_resolution_clock::now();
        has_new_gyro_data = true;
    }
    
    // Signal that we've received the first serial data
    // Make sure this is outside of the previous lock to avoid potential deadlocks
    if (!first_serial_received.load()) {
        std::unique_lock<std::mutex> initial_lock(initial_sync_mutex);
        first_serial_received.store(true);
        initial_sync_cv.notify_all(); // Notify all waiting threads
        std::cout << "First serial data received, notifying waiting threads" << std::endl;
    }
}
// 初始化时间点，用于确保程序启动3秒后才开始同步
std::chrono::high_resolution_clock::time_point program_start_time;

// 在main函数中初始化

constexpr int MAX_QUEUE_SIZE = 30;         // 减少队列大小以降低内存占用
// constexpr float TIME_MATCH_MIN_MS = 8;     // 时间戳匹配最小值 (ms)
// constexpr float TIME_MATCH_MAX_MS = 8;     // 时间戳匹配最大值 (ms)
constexpr size_t BUFFER_SIZE = 32;         // 串口缓冲区大小，提高为2的幂次方以优化内存对齐

FrameData detector_input;
// 核心状态控制
std::atomic<bool> running(true);
std::atomic<bool> camera_initialized(false);

// 使用Boost线程锁，减少全局锁争用
boost::mutex cout_mutex;
boost::mutex frame_queue_mutex;
boost::mutex gyro_queue_mutex;

std::atomic<bool> new_frame_available(false);
std::atomic<bool> new_gyro_available(false);

// 数据结构优化：使用内存对齐且尽量避免虚拟函数以减少缓存未命中
// 使用struct而非class，避免不必要的封装开销
std::unique_ptr<helios_cv::ArmorProjectYaw> armor_pnp_solver;
std::unique_ptr<helios_cv::EnergyProjectRoll> energy_pnp_solver;
boost::mutex pnp_mutex;
// 将int64_t时间戳(毫秒)转换为time_point
inline std::chrono::high_resolution_clock::time_point int64ToTimePoint(int64_t timestamp_ms) {
    return std::chrono::high_resolution_clock::time_point(
        std::chrono::milliseconds(timestamp_ms));
}

// 将time_point转换为int64_t时间戳(毫秒)
inline int64_t timePointToInt64(const std::chrono::high_resolution_clock::time_point& tp) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        tp.time_since_epoch()).count();
}
struct alignas(64) TimestampedFrame {    
    cv::Mat frame;
    int64_t timestamp;
    uint64_t sync_id;  // 同步ID

    TimestampedFrame() 
        : frame(), timestamp(-1), sync_id(UINT64_MAX) {}

    TimestampedFrame(const cv::Mat& f, int64_t ts, uint64_t id) 
        : frame(f), timestamp(ts), sync_id(id) {}
};

// 修改TimestampedPacket结构
struct alignas(64) TimestampedPacket {
    helios::MCUPacket packet;
    std::chrono::high_resolution_clock::time_point timestamp;
    uint64_t sync_id;  // 同步ID

    TimestampedPacket() 
        : packet(), timestamp(std::chrono::high_resolution_clock::time_point::min()), sync_id(UINT64_MAX) {}

    TimestampedPacket(const helios::MCUPacket& p, std::chrono::high_resolution_clock::time_point ts_ms, uint64_t id) 
        : packet(p), timestamp(ts_ms), sync_id(id) {}
};


boost::circular_buffer<TimestampedFrame> frame_queue(MAX_QUEUE_SIZE);
boost::circular_buffer<TimestampedPacket> gyro_queue(MAX_QUEUE_SIZE);
int last_timestamp;


boost::mutex detector_mutex;

std::unique_ptr<OVnetDetector> global_detector;

bool initialize_detector(const std::string& model_path, bool is_blue = true) {
    try {
        boost::lock_guard<boost::mutex> lock(detector_mutex);
        global_detector = std::make_unique<OVnetDetector>(model_path, is_blue);
        return global_detector->initialize();
    } catch (const std::exception& e) {
        boost::lock_guard<boost::mutex> lock(cout_mutex);
        std::cerr << "Failed to initialize detector: " << e.what() << std::endl;
        return false;
    }
}

bool initialize_pnp_solvers(const std::array<double, 9>& camera_matrix, const std::vector<double>& dist_coeffs) {
    try {
        boost::lock_guard<boost::mutex> lock(pnp_mutex);
        

        armor_pnp_solver = std::make_unique<helios_cv::ArmorProjectYaw>(camera_matrix, dist_coeffs);
        armor_pnp_solver->set_use_projection(true); 
        
        energy_pnp_solver = std::make_unique<helios_cv::EnergyProjectRoll>(camera_matrix, dist_coeffs);
        energy_pnp_solver->set_use_projection(true); 
        
        return true;
    } catch (const std::exception& e) {
        boost::lock_guard<boost::mutex> lock(cout_mutex);
        std::cerr << "Failed to initialize PnP solvers: " << e.what() << std::endl;
        return false;
    }
}

constexpr int MAX_PENDING_RESULTS = 5; 
boost::mutex armor_pending_results_mutex;
void start_sync() {
    // 等待相机和串口都初始化完成
    while (running && (!camera_initialized.load(std::memory_order_acquire) || !first_serial_received.load())) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    // 确保两个设备都准备好再开始同步
    {
        boost::lock_guard<boost::mutex> lock(cout_mutex);
        std::cout << "===== 相机和串口都已初始化完成 =====" << std::endl;
        std::cout << "开始硬件同步计数！时间: " 
                  << std::chrono::time_point_cast<std::chrono::milliseconds>(
                       std::chrono::high_resolution_clock::now()).time_since_epoch().count() 
                  << "ms" << std::endl;
    }
    
    // 重置同步ID
    sync_frame_id.store(0, std::memory_order_release);
    sync_gyro_id.store(0, std::memory_order_release);
    
    // 清空现有队列，以避免旧数据干扰
    {
        boost::lock_guard<boost::mutex> frame_lock(frame_queue_mutex);
        frame_queue.clear();
    }
    
    {
        boost::lock_guard<boost::mutex> gyro_lock(gyro_queue_mutex);
        gyro_queue.clear();
    }
    
    // 添加短暂延迟，确保所有线程都准备好同步状态
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    // 激活同步信号
    ready_for_sync.store(true, std::memory_order_release);
    sync_started.store(true, std::memory_order_release);
}

std::deque<std::tuple<std::chrono::high_resolution_clock::time_point, helios::MCUPacket, std::shared_future<FrameData>>> armor_pending_results;
// Function to visualize armor detections in odom frame
// 三维坐标系可视化函数
void visualizeArmorsIn3DOdomFrame(
    const std::vector<autoaim_interfaces::msg::Armor>& armors,
    const cv::Quatd& cam2odom_rotation,  // 相机到odom的四元数旋转
    int wait_key = 1)
{
    // 创建三个视图：顶视图(XY)、前视图(XZ)和侧视图(YZ)
    const int view_size = 400;  // 每个视图的大小
    cv::Mat top_view = cv::Mat::zeros(view_size, view_size, CV_8UC3);
    cv::Mat front_view = cv::Mat::zeros(view_size, view_size, CV_8UC3);
    cv::Mat side_view = cv::Mat::zeros(view_size, view_size, CV_8UC3);
    
    // 坐标系中心点
    cv::Point2i center(view_size/2, view_size/2);
    
    // 缩放因子
    const double scale = 100.0;
    
    // 绘制坐标轴
    // 顶视图 (XY平面)
    cv::line(top_view, center, center + cv::Point2i(scale, 0), cv::Scalar(0, 0, 255), 2);  // X轴 (红色)
    cv::line(top_view, center, center + cv::Point2i(0, -scale), cv::Scalar(0, 255, 0), 2);  // Y轴 (绿色)
    cv::putText(top_view, "X", center + cv::Point2i(scale+10, 0), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    cv::putText(top_view, "Y", center + cv::Point2i(0, -scale-10), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    cv::putText(top_view, "Top View (XY)", cv::Point(20, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    // 前视图 (XZ平面)
    cv::line(front_view, center, center + cv::Point2i(scale, 0), cv::Scalar(0, 0, 255), 2);  // X轴 (红色)
    cv::line(front_view, center, center + cv::Point2i(0, -scale), cv::Scalar(255, 0, 0), 2);  // Z轴 (蓝色)
    cv::putText(front_view, "X", center + cv::Point2i(scale+10, 0), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    cv::putText(front_view, "Z", center + cv::Point2i(0, -scale-10), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
    cv::putText(front_view, "Front View (XZ)", cv::Point(20, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    // 侧视图 (YZ平面)
    cv::line(side_view, center, center + cv::Point2i(scale, 0), cv::Scalar(0, 255, 0), 2);  // Y轴 (绿色)
    cv::line(side_view, center, center + cv::Point2i(0, -scale), cv::Scalar(255, 0, 0), 2);  // Z轴 (蓝色)
    cv::putText(side_view, "Y", center + cv::Point2i(scale+10, 0), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    cv::putText(side_view, "Z", center + cv::Point2i(0, -scale-10), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
    cv::putText(side_view, "Side View (YZ)", cv::Point(20, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    // 绘制网格线 (每10单位一条线)
    const int grid_step = 50;
    const cv::Scalar grid_color(30, 30, 30);
    
    for (int i = -4; i <= 4; i++) {
        if (i == 0) continue;  // 跳过坐标轴
        int pos = center.x + i * grid_step;
        
        // 顶视图网格
        cv::line(top_view, cv::Point(pos, 0), cv::Point(pos, view_size), grid_color, 1);
        cv::line(top_view, cv::Point(0, center.y + i * grid_step), 
                 cv::Point(view_size, center.y + i * grid_step), grid_color, 1);
        
        // 前视图网格
        cv::line(front_view, cv::Point(pos, 0), cv::Point(pos, view_size), grid_color, 1);
        cv::line(front_view, cv::Point(0, center.y + i * grid_step), 
                 cv::Point(view_size, center.y + i * grid_step), grid_color, 1);
        
        // 侧视图网格
        cv::line(side_view, cv::Point(pos, 0), cv::Point(pos, view_size), grid_color, 1);
        cv::line(side_view, cv::Point(0, center.y + i * grid_step), 
                 cv::Point(view_size, center.y + i * grid_step), grid_color, 1);
    }
    
    // 将四元数转换为旋转矩阵
    cv::Mat rotation_matrix;
    cv::Mat(cam2odom_rotation.toRotMat3x3()).convertTo(rotation_matrix, CV_64F);
    
    // 绘制每个装甲板
    const cv::Scalar colors[] = {
        cv::Scalar(255, 0, 0),    // 蓝色
        cv::Scalar(0, 255, 0),    // 绿色
        cv::Scalar(0, 0, 255),    // 红色
        cv::Scalar(255, 255, 0),  // 青色
        cv::Scalar(255, 0, 255),  // 洋红色
        cv::Scalar(0, 255, 255)   // 黄色
    };
    
    for (size_t i = 0; i < armors.size(); ++i) {
        const auto& armor = armors[i];
        cv::Scalar color = colors[i % 6];
        
        // 转换相机坐标系下的位置到odom坐标系
        cv::Mat pos_camera(3, 1, CV_64F);
        pos_camera.at<double>(0) = armor.pose.position.x;
        pos_camera.at<double>(1) = armor.pose.position.y;
        pos_camera.at<double>(2) = armor.pose.position.z;
        
        // 应用旋转变换到odom坐标系
        cv::Mat pos_odom = rotation_matrix * pos_camera;
        
        // 提取三维坐标
        double x = pos_odom.at<double>(0);
        double y = pos_odom.at<double>(1);
        double z = pos_odom.at<double>(2);
        
        // 计算各视图中的2D位置
        cv::Point2i pos_top(center.x + x * scale, center.y - y * scale);  // 顶视图 (XY)
        cv::Point2i pos_front(center.x + x * scale, center.y - z * scale);  // 前视图 (XZ)
        cv::Point2i pos_side(center.x + y * scale, center.y - z * scale);  // 侧视图 (YZ)
        
        // 绘制装甲板点
        const int point_size = 5;
        cv::circle(top_view, pos_top, point_size, color, -1);
        cv::circle(front_view, pos_front, point_size, color, -1);
        cv::circle(side_view, pos_side, point_size, color, -1);
        
        // 绘制从原点到装甲板的线段
        cv::line(top_view, center, pos_top, color, 1);
        cv::line(front_view, center, pos_front, color, 1);
        cv::line(side_view, center, pos_side, color, 1);
        
        // 显示装甲板编号
        std::string number_text = armor.number;
        cv::putText(top_view, number_text, pos_top + cv::Point2i(10, 0), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        cv::putText(front_view, number_text, pos_front + cv::Point2i(10, 0), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        cv::putText(side_view, number_text, pos_side + cv::Point2i(10, 0), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }
    
    // 合并三个视图为一个窗口
    cv::Mat visualization;
    cv::Mat top_row, bottom_row;
    cv::hconcat(top_view, front_view, top_row);
    
    // 创建一个信息面板
    cv::Mat info_panel = cv::Mat::zeros(view_size, view_size, CV_8UC3);
    cv::putText(info_panel, "3D Armor Visualization", cv::Point(60, 50), 
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    cv::putText(info_panel, "Color Legend:", cv::Point(30, 100), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
    
    // 装甲板颜色图例
    for (int i = 0; i < std::min(6, (int)armors.size()); i++) {
        cv::circle(info_panel, cv::Point(50, 130 + i * 30), 5, colors[i], -1);
        std::string armor_info = "Armor " + armors[i].number + 
                                 " (Type: " + std::to_string(armors[i].type) + ")";
        cv::putText(info_panel, armor_info, cv::Point(70, 135 + i * 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1);
    }
    
    // 添加坐标系信息
    cv::putText(info_panel, "Coordinate System:", cv::Point(30, 280), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
    cv::line(info_panel, cv::Point(50, 310), cv::Point(80, 310), cv::Scalar(0, 0, 255), 2);
    cv::putText(info_panel, "X axis", cv::Point(90, 315), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    cv::line(info_panel, cv::Point(50, 340), cv::Point(80, 340), cv::Scalar(0, 255, 0), 2);
    cv::putText(info_panel, "Y axis", cv::Point(90, 345), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    cv::line(info_panel, cv::Point(50, 370), cv::Point(80, 370), cv::Scalar(255, 0, 0), 2);
    cv::putText(info_panel, "Z axis", cv::Point(90, 375), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
    
    cv::hconcat(side_view, info_panel, bottom_row);
    cv::vconcat(top_row, bottom_row, visualization);
    
    // 显示可视化结果
    cv::imshow("3D Armor Visualization in Odom Frame", visualization);
    cv::waitKey(wait_key);
}
void visualize_armors_in_odom(const FrameData& result, 
                             const cv::Quatd& cam2odom_rotation, const cv::Vec3d& cam2odom_translation,
                             const cv::Mat& frame, std::chrono::high_resolution_clock::time_point timestamp) {
    // Get the latest gyro data
    helios::MCUPacket gyro_data;
    {
        std::lock_guard<std::mutex> lock(latest_gyro_mutex);
        gyro_data = latest_gyro_data;
    }
    
    // Create output visualization image
    cv::Mat vis_img = frame.clone();
    auto ts_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(timestamp).time_since_epoch().count();
    auto gyro_ts_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(latest_gyro_timestamp).time_since_epoch().count();
    
    // Add timestamp and gyro data to visualization
    cv::putText(vis_img, "Camera TS: " + std::to_string(ts_ms), cv::Point(10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    cv::putText(vis_img, "Gyro TS: " + std::to_string(gyro_ts_ms), cv::Point(10, 60), 
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    cv::putText(vis_img, "Yaw: " + std::to_string(gyro_data.yaw) + ", Pitch: " + 
                std::to_string(gyro_data.pitch), cv::Point(10, 90), 
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    
    // Rest of the visualization code as before...
    // [The 3D visualization code here remains the same as in my previous message]
}
void process_detection_result(const FrameData& result, std::chrono::high_resolution_clock::time_point timestamp) {
    auto timestamp_ns = std::chrono::time_point_cast<std::chrono::nanoseconds>(timestamp).time_since_epoch().count();
    
    cv::Quatd cam2odom_rotation;
    cv::Vec3d cam2odom_translation;
    cv::Quatd odom2cam_rotation;
    cv::Vec3d odom2cam_translation;
    
    bool transform_success = transform_manager->getTransforms(
        timestamp_ns,
        cam2odom_rotation, cam2odom_translation,
        odom2cam_rotation, odom2cam_translation
    );
    
    if (!transform_success) {
        std::cout << "Failed to get transforms for timestamp: " << timestamp_ns << std::endl;
        return;
    }
    
    // Create ROS time from timestamp
    rclcpp::Time ros_time(timestamp_ns);
    
    // Create Armors message
    auto armors_msg = std::make_unique<autoaim_interfaces::msg::Armors>();
    armors_msg->header.stamp = ros_time;
    armors_msg->header.frame_id = "camera_optical_frame";
    
    // Create markers for visualization
    visualization_msgs::msg::MarkerArray marker_array;
    
    if (result.has_valid_results()) {
        boost::lock_guard<boost::mutex> lock(pnp_mutex);
        if (armor_pnp_solver) {
            // Check if we have new gyro data
            {
                std::lock_guard<std::mutex> lock(latest_gyro_mutex);
                if (!has_new_gyro_data) {
                    std::cout << "No gyro data available yet, skipping visualization" << std::endl;
                    return;
                }
            }
            
            // Prepare transform info for PnP solver
            helios_cv::ArmorTransformInfo armor_transform_info(odom2cam_rotation, cam2odom_rotation);
            armor_pnp_solver->update_transform_info(&armor_transform_info);
            
            // Process each armor
            for (const auto& armor : result.armors) {
                cv::Mat rvec, tvec;
                bool pose_solved = armor_pnp_solver->solve_pose(armor, rvec, tvec);
                
                if (pose_solved) {
                    autoaim_interfaces::msg::Armor armor_msg;
                    
                    armor_msg.type = static_cast<int>(armor.type);
                    armor_msg.number = armor.number;
                    
                    armor_msg.pose.position.x = tvec.at<double>(0);
                    armor_msg.pose.position.y = tvec.at<double>(1);
                    armor_msg.pose.position.z = tvec.at<double>(2);
                    
                    cv::Mat rotation_matrix;
                    if (armor_pnp_solver->use_projection()) {
                        rotation_matrix = rvec;
                    } else {
                        cv::Rodrigues(rvec, rotation_matrix);
                    }
                    
                    tf2::Matrix3x3 tf2_rotation_matrix(
                        rotation_matrix.at<double>(0, 0),
                        rotation_matrix.at<double>(0, 1),
                        rotation_matrix.at<double>(0, 2),
                        rotation_matrix.at<double>(1, 0),
                        rotation_matrix.at<double>(1, 1),
                        rotation_matrix.at<double>(1, 2),
                        rotation_matrix.at<double>(2, 0),
                        rotation_matrix.at<double>(2, 1),
                        rotation_matrix.at<double>(2, 2)
                    );
                    
                    tf2::Quaternion tf2_quaternion;
                    tf2_rotation_matrix.getRotation(tf2_quaternion);
                    armor_msg.pose.orientation = tf2::toMsg(tf2_quaternion);
                    armor_msg.distance_to_image_center = 
                        armor_pnp_solver->calculateDistanceToCenter(armor.center);
                    
                    armors_msg->armors.push_back(armor_msg);
                }
            }
            
            // Create and publish markers if we have valid armors
            if (!armors_msg->armors.empty()) {
                // Show the armor projections in visualization
                cv::Mat draw = result.frame.clone();
                armor_pnp_solver->draw_projection_points(draw);
                cv::imshow("Armor Projection", result.frame);
                cv::waitKey(1);
                
                // Create visualization markers
                marker_array.markers.clear();
                int marker_id = 0;
                
                // Template for armor markers
                visualization_msgs::msg::Marker armor_marker;
                armor_marker.ns = "armors";
                armor_marker.action = visualization_msgs::msg::Marker::ADD;
                armor_marker.type = visualization_msgs::msg::Marker::CUBE;
                armor_marker.scale.x = 0.05;
                armor_marker.scale.z = 0.125;
                armor_marker.color.a = 1.0;
                armor_marker.color.g = 0.5;
                armor_marker.color.b = 1.0;
                armor_marker.lifetime = rclcpp::Duration::from_seconds(0.1);
                
                // Template for text markers
                visualization_msgs::msg::Marker text_marker;
                text_marker.ns = "classification";
                text_marker.action = visualization_msgs::msg::Marker::ADD;
                text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
                text_marker.scale.z = 0.1;
                text_marker.color.a = 1.0;
                text_marker.color.r = 1.0;
                text_marker.color.g = 1.0;
                text_marker.color.b = 1.0;
                text_marker.lifetime = rclcpp::Duration::from_seconds(0.1);
                
                for (const auto& armor : armors_msg->armors) {
                    // Armor cube marker
                    armor_marker.id = marker_id++;
                    armor_marker.header.stamp = ros_time;
                    armor_marker.header.frame_id = "camera_optical_frame";
                    armor_marker.scale.y = (armor.type == 0) ? 0.135 : 0.23;
                    armor_marker.pose = armor.pose;
                    marker_array.markers.push_back(armor_marker);
                    
                    // Armor number text marker
                    text_marker.id = marker_id++;
                    text_marker.header.stamp = ros_time;
                    text_marker.header.frame_id = "camera_optical_frame";
                    text_marker.pose.position = armor.pose.position;
                    text_marker.pose.position.y -= 0.1;
                    text_marker.text = armor.number;
                    marker_array.markers.push_back(text_marker);
                }
            } else {
                // If no armors detected, send a marker to clear previous markers
                marker_array.markers.clear();
                visualization_msgs::msg::Marker clear_marker;
                clear_marker.ns = "armors";
                clear_marker.action = visualization_msgs::msg::Marker::DELETE;
                clear_marker.id = 0;
                clear_marker.header.stamp = ros_time;
                clear_marker.header.frame_id = "camera_optical_frame";
                marker_array.markers.push_back(clear_marker);
            }
            
            // Use our custom visualization
            visualize_armors_in_odom(result, cam2odom_rotation, cam2odom_translation,
                                    result.frame, timestamp);
        }
    }
    
    // Publish armor messages and markers
    armors_pub_->publish(*armors_msg);
    marker_pub_->publish(marker_array);
}

void camera_thread_func() {
    pthread_t this_thread = pthread_self();
    struct sched_param params;
    params.sched_priority = sched_get_priority_max(SCHED_FIFO);
    pthread_setschedparam(this_thread, SCHED_FIFO, &params);
    
    camera::HikCamera hikCam;
    // Remove the unused variable warning
    std::string camera_config_path = HIK_CONFIG_FILE_PATH"/camera_config.yaml";
    
    try {
        hikCam.Init(true, camera_config_path, false);
    } catch (const std::exception& e) {
        boost::lock_guard<boost::mutex> lock(cout_mutex);
        std::cerr << "相机初始化失败: " << e.what() << std::endl;
        running = false;
        return;
    }
    
    {
        boost::lock_guard<boost::mutex> lock(cout_mutex);
        std::cout << "海康相机初始化成功" << std::endl;
    }
    
    // 清空缓存的图像
    cv::Mat temp_frame;
    uint64_t temp_device_ts;
    int64_t temp_host_ts;
    while (hikCam.ReadImg(temp_frame, &temp_device_ts, &temp_host_ts)) {
        // 持续读取直到没有新图像
    }
    
    camera_initialized.store(true, std::memory_order_release);

// Wait for first serial data with timeout
{
    std::unique_lock<std::mutex> lock(initial_sync_mutex);
    if (!first_serial_received.load()) {
        std::cout << "Camera initialized, waiting for first serial data (timeout: 10s)..." << std::endl;
        
        // Wait with a timeout of 10 seconds
        if (!initial_sync_cv.wait_for(lock, std::chrono::seconds(10), 
                                     []{ return first_serial_received.load(); })) {
            // If timeout occurred, we'll still proceed but log a warning
            std::cout << "WARNING: Timeout waiting for serial data. Proceeding anyway..." << std::endl;
        } else {
            std::cout << "First serial data received, starting camera processing." << std::endl;
        }
    }
}
    
    // Main processing loop
    cv::Mat frame;
    uint64_t device_ts;
    int64_t host_ts;
    
    while (running) {
        bool hasNew = hikCam.ReadImg(frame, &device_ts, &host_ts);
        
        if (hasNew && !frame.empty() && sync_started.load(std::memory_order_acquire)) {
            // 获取当前的同步ID
            uint64_t current_sync_id = sync_frame_id.fetch_add(1, std::memory_order_relaxed);
            
            boost::lock_guard<boost::mutex> lock(frame_queue_mutex);
            if (frame_queue.empty() || host_ts > frame_queue.back().timestamp) {
                frame_queue.push_back(TimestampedFrame(frame, host_ts, current_sync_id));
                new_frame_available.store(true, std::memory_order_release);
                
                // 增强输出信息
                std::cout << "[FRAME] ID: " << std::setw(5) << current_sync_id 
                          << " | 时间戳: " << std::setw(13) << host_ts 
                          << " | 系统时间: " << std::chrono::time_point_cast<std::chrono::milliseconds>(
                                                  std::chrono::high_resolution_clock::now()).time_since_epoch().count() 
                          << "ms" << std::endl;
            } else {
                std::cerr << "[ERROR] 收到乱序帧，丢弃。同步ID: " 
                          << current_sync_id << ", 时间戳: " << host_ts << std::endl;
            }
            
            // 可视化ID和时间戳
            cv::Mat debug_frame = frame.clone();
            cv::putText(debug_frame, "Frame ID: " + std::to_string(current_sync_id),
                       cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
            cv::putText(debug_frame, "Timestamp: " + std::to_string(host_ts),
                       cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
            cv::imshow("Synchronized Frame", debug_frame);
            cv::waitKey(1);
        }
    }
    
    {
        boost::lock_guard<boost::mutex> lock(cout_mutex);
        std::cout << "相机线程已结束" << std::endl;
    }
}





void serial_thread_func(const std::string& port_name, int baud_rate, std::shared_ptr<rclcpp::Node> node) {
    {
        boost::lock_guard<boost::mutex> lock(cout_mutex);
        std::cout << "Opening serial port: " << port_name << " at " << baud_rate << " baud" << std::endl;
    }
    

    if (!running) {
        return;
    }
    
    // 设置线程优先级
    pthread_t this_thread = pthread_self();
    struct sched_param params;
    params.sched_priority = sched_get_priority_max(SCHED_FIFO);
    pthread_setschedparam(this_thread, SCHED_FIFO, &params);
    
    bool port_open = false;
    boost::asio::io_service io_service;
    boost::asio::serial_port serial_port(io_service);
    std::chrono::time_point<std::chrono::steady_clock> last_reconnect_attempt;
    
    // 创建数据缓冲区
    std::vector<uint8_t> receive_buffer;
    std::vector<uint8_t> temp_buffer(BUFFER_SIZE);
    
    // 打开串口的函数
    auto open_serial = [&]() -> bool {
        if (serial_port.is_open()) {
            try {
                serial_port.close();
            } catch (...) {
                // 忽略关闭错误
            }
        }
        
        boost::system::error_code ec;
        serial_port.open(port_name, ec);
        
        if (ec) {
            return false;
        }
        

        serial_port.set_option(boost::asio::serial_port_base::baud_rate(baud_rate), ec);
        serial_port.set_option(boost::asio::serial_port_base::character_size(8), ec);
        serial_port.set_option(boost::asio::serial_port_base::parity(boost::asio::serial_port_base::parity::none), ec);
        serial_port.set_option(boost::asio::serial_port_base::stop_bits(boost::asio::serial_port_base::stop_bits::one), ec);
        serial_port.set_option(boost::asio::serial_port_base::flow_control(boost::asio::serial_port_base::flow_control::none), ec);
        

        try {
            int fd = serial_port.native_handle();
            int flags = fcntl(fd, F_GETFL, 0);
            if (flags != -1) {
                fcntl(fd, F_SETFL, flags | O_NONBLOCK);
            }
        } catch (...) {
            // 忽略设置非阻塞模式的错误
        }
        
        return true;
    };
    
    // 首次尝试打开串口
    port_open = open_serial();
    if (port_open) {
        boost::lock_guard<boost::mutex> lock(cout_mutex);
        std::cout << "串口打开成功，开始读取数据" << std::endl;
    } else {
        boost::lock_guard<boost::mutex> lock(cout_mutex);
        std::cout << "串口打开失败，将在运行中尝试重连" << std::endl;
    }
    
    last_reconnect_attempt = std::chrono::steady_clock::now();
    
    // 主循环
    while (running) {
        // 检查是否需要重新连接
        if (!port_open) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_reconnect_attempt).count();
            
            if (elapsed >= 1) { // 每秒尝试重连一次
                port_open = open_serial();
                if (port_open) {
                    boost::lock_guard<boost::mutex> lock(cout_mutex);
                    std::cout << "串口重连成功" << std::endl;
                }
                last_reconnect_attempt = now;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        
        // 尝试读取数据
        boost::system::error_code error;
        size_t bytes_read = 0;
        
        try {
            io_service.reset();
            io_service.poll();
            
            bytes_read = serial_port.read_some(boost::asio::buffer(temp_buffer), error);
            
            if (error) {
                if (error != boost::asio::error::would_block) {
                    // 只在首次错误或错误类型变化时输出
                    static boost::system::error_code last_error;
                    if (last_error != error) {
                        boost::lock_guard<boost::mutex> lock(cout_mutex);
                        std::cout << "串口读取错误: " << error.message() << std::endl;
                        last_error = error;
                    }
                    
                    // 遇到严重错误，认为端口已关闭
                    if (error == boost::asio::error::operation_aborted ||
                        error == boost::asio::error::connection_reset ||
                        error == boost::asio::error::eof ||
                        error == boost::asio::error::broken_pipe) {
                        port_open = false;
                        continue;
                    }
                }
            } else {
                // 成功读取，重置错误计数
                static boost::system::error_code last_error;
                last_error = boost::system::error_code();
            }
            
            if (bytes_read > 0) {
                // 将读取的数据添加到接收缓冲区
                size_t original_size = receive_buffer.size();
                receive_buffer.resize(original_size + bytes_read);
                std::copy(temp_buffer.begin(), temp_buffer.begin() + bytes_read, receive_buffer.begin() + original_size);
                
                // 处理缓冲区中的所有完整数据包
                while (receive_buffer.size() >= sizeof(helios::FrameHeader)) {
                    // 查找SOF
                    auto sof_it = std::find(receive_buffer.begin(), receive_buffer.end(), 0xA5);
                    if (sof_it == receive_buffer.end()) {
                        // 没有SOF，清空缓冲区
                        receive_buffer.clear();
                        break;
                    }
                    
                    // 如果SOF不在开头，删除SOF之前的数据
                    if (sof_it != receive_buffer.begin()) {
                        receive_buffer.erase(receive_buffer.begin(), sof_it);
                    }
                    
                    // 如果缓冲区不足以容纳一个完整的帧头，等待更多数据
                    if (receive_buffer.size() < sizeof(helios::FrameHeader)) {
                        break;
                    }
                    
                    // 帧头
                    helios::FrameHeader header;
                    memcpy(&header, receive_buffer.data(), sizeof(helios::FrameHeader));
                    
                    // 确保数据长度在合理
                    if (header.data_length > 1024 || header.data_length < sizeof(helios::MCUPacket)) {
                        // 非法长度，删除SOF，继续查找下一个SOF
                        receive_buffer.erase(receive_buffer.begin());
                        continue;
                    }
                    
                    // TODO: CRC校验
                    size_t total_packet_size = sizeof(helios::FrameHeader) + header.data_length;
                    if (receive_buffer.size() < total_packet_size) {
                        // 数据不完整，等待更多数据
                        break;
                    }
                    

                    if (header.cmd_id == helios::RECEIVE_AUTOAIM_RECEIVE_CMD_ID && 
                        header.data_length == sizeof(helios::MCUPacket)) {
                        auto now = std::chrono::high_resolution_clock::now();
                        
                        // Always extract the packet data
                        helios::MCUPacket packet;
                        memcpy(&packet, receive_buffer.data() + sizeof(helios::FrameHeader), sizeof(helios::MCUPacket));
                        
                        // Always update the latest gyro data for other threads
                        update_latest_gyro_data(packet);
                        
                        // Only add to sync queue if synchronization has started
                        if (sync_started.load(std::memory_order_acquire)) {
                            uint64_t current_sync_id = sync_gyro_id.fetch_add(1, std::memory_order_relaxed);
                            
                            // 获取当前系统时间作为接收时间戳
                            auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now).time_since_epoch().count();
                            
                            // Output enhanced information
                            std::cout << "[SERIAL] ID: " << std::setw(5) << current_sync_id 
                                      << " | 时间戳: " << std::setw(13) << now_ms 
                                      << " | Yaw: " << std::setw(8) << std::fixed << std::setprecision(3) << packet.yaw
                                      << " | Pitch: " << std::setw(8) << std::fixed << std::setprecision(3) << packet.pitch
                                      << " | 模式: " << static_cast<int>(packet.autoaim_mode) << std::endl;
                            
                            // Add to sync queue
                            {
                                boost::lock_guard<boost::mutex> lock(gyro_queue_mutex);
                                gyro_queue.push_back(TimestampedPacket(packet, now, current_sync_id));
                                new_gyro_available.store(true, std::memory_order_release);
                            }
                            
                            // Update coordinate transformations
                            auto timestamp_ns = std::chrono::time_point_cast<std::chrono::nanoseconds>(now).time_since_epoch().count();
                            transform_manager->updateTransform(timestamp_ns, packet.yaw, packet.pitch);
                        
                        rclcpp::Time ros_time(timestamp_ns);
                        // printf("Timestamp for tf: %.10ld\n", timestamp_ns);
                        geometry_msgs::msg::TransformStamped ts;
                        geometry_msgs::msg::TransformStamped cam_ts;
                        cam_ts.header.frame_id = "pitch_link";
                        cam_ts.child_frame_id = "camera_link";
                        cam_ts.header.stamp = ros_time;
                        cam_ts.transform.translation.x = 0.125;
                        cam_ts.transform.translation.y = 0.0;
                        cam_ts.transform.translation.z = -0.035;
                        tf2::Quaternion cam_q;
                        cam_q.setRPY(0, 0, 0);
                        cam_ts.transform.rotation = tf2::toMsg(cam_q);
                        tf_broadcaster_->sendTransform(cam_ts);
                    
                        // Static transform from camera_link to camera_optical_frame
                        geometry_msgs::msg::TransformStamped optical_ts;
                        optical_ts.header.frame_id = "camera_link";
                        optical_ts.child_frame_id = "camera_optical_frame";
                        optical_ts.header.stamp = ros_time;
                        tf2::Quaternion optical_q;
                        optical_q.setRPY(-M_PI/2, 0, -M_PI/2);
                        optical_ts.transform.rotation = tf2::toMsg(optical_q);
                        tf_broadcaster_->sendTransform(optical_ts);
                        
                        // odoom to yaw_link transform
                        ts.header.frame_id = "odoom";
                        ts.child_frame_id = "yaw_link";
                        ts.header.stamp = ros_time;
                        
                        tf2::Quaternion q;
                        q.setRPY(0, 0, -packet.yaw);  // Note the negative sign like in CtrlBridge
                        ts.transform.rotation = tf2::toMsg(q);
                        tf_broadcaster_->sendTransform(ts);
                        
                        // yaw_link to pitch_link transform
                        ts.header.frame_id = "yaw_link";
                        ts.child_frame_id = "pitch_link";
                        ts.header.stamp = ros_time;
                        q.setRPY(0, -packet.pitch, 0);  // Note the negative sign like in CtrlBridge
                        ts.transform.rotation = tf2::toMsg(q);
                        tf_broadcaster_->sendTransform(ts);
                    }
                }
                    
                    receive_buffer.erase(receive_buffer.begin(), receive_buffer.begin() + total_packet_size);
                }
            }
        } catch (const std::exception& e) {
            // 仅记录重要异常
            static std::string last_exception;
            if (last_exception != e.what()) {
                boost::lock_guard<boost::mutex> lock(cout_mutex);
                std::cerr << "串口处理异常: " << e.what() << std::endl;
                last_exception = e.what();
            }
            
            port_open = false;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        
    }
    
    // 关闭串口
    if (serial_port.is_open()) {
        try {
            boost::system::error_code close_ec;
            serial_port.close(close_ec);
        } catch (...) {
          
        }
    }
    
    {
        boost::lock_guard<boost::mutex> lock(cout_mutex);
        std::cout << "串口线程已结束" << std::endl;
    }
}

cv::Quatd ros2cv(const geometry_msgs::msg::Quaternion& q) {
    return cv::Quatd(q.w, q.x, q.y, q.z);
}
// 修改函数签名，接受匹配的陀螺仪数据
void process_matched_pair(const cv::Mat& frame, const helios::MCUPacket& matched_gyro, std::chrono::high_resolution_clock::time_point timestamp) {
    FrameData input_data(timestamp, frame);
    
    // 直接使用传入的匹配陀螺仪数据，而不是latest_gyro_data
    bool should_process = (matched_gyro.autoaim_mode == 0);
    
    if (should_process) {
        auto ts_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(timestamp).time_since_epoch().count();
        
        std::cout << "处理匹配的帧/陀螺仪对 - 帧时间戳: " << ts_ms 
                  << " 使用匹配的陀螺仪数据 (Yaw: " << matched_gyro.yaw 
                  << ", Pitch: " << matched_gyro.pitch << ")" << std::endl;
        
        {
            boost::lock_guard<boost::mutex> lock(detector_mutex);
            if (global_detector) {
                {
                    std::lock_guard<std::mutex> lock(latest_gyro_mutex);
                    latest_gyro_data = matched_gyro;
                    latest_gyro_timestamp = timestamp;
                    has_new_gyro_data = true;
                }
                
                FrameData result = global_detector->infer_sync(input_data);
                global_detector->visualize_detection_result(result);
                process_detection_result(result, timestamp);
            } else {
                std::cerr << "检测器未初始化。" << std::endl;
                return;
            }
        }
    } else {
        boost::lock_guard<boost::mutex> lock(cout_mutex);
        std::cout << "自瞄模式不是0，跳过检测。" << std::endl;
    }
}
void data_matching_thread_func() {
    // Wait for first serial data before attempting any matching
    {
        std::unique_lock<std::mutex> lock(initial_sync_mutex);
        if (!first_serial_received.load()) {
            std::cout << "Matching thread waiting for first serial data..." << std::endl;
            initial_sync_cv.wait(lock, []{ return first_serial_received.load(); });
            std::cout << "First serial data received, starting matching process." << std::endl;
        }
    }
    
    // Wait for sync to start
    while (running && !sync_started.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    if (!running) {
        return;
    }

    {
        boost::lock_guard<boost::mutex> lock(cout_mutex);
        std::cout << "开始基于同步ID的数据匹配线程" << std::endl;
    }

    int total_attempts = 0;
    int successful_matches = 0;

    while (running) {
        bool frame_available = new_frame_available.load(std::memory_order_acquire);
        bool gyro_available = new_gyro_available.load(std::memory_order_acquire);

        if (!frame_available || !gyro_available) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            continue;
        }

        total_attempts++;

        TimestampedFrame matched_frame;
        TimestampedPacket matched_gyro;
        bool found_match = false;
        
        {
            boost::lock_guard<boost::mutex> lock(frame_queue_mutex);
            boost::lock_guard<boost::mutex> lock2(gyro_queue_mutex);

            if (frame_queue.empty() || gyro_queue.empty()) {
                continue;
            }
            
            // 查找具有相同同步ID的帧和IMU数据
            for (const auto& frame : frame_queue) {
                for (const auto& gyro : gyro_queue) {
                    if (frame.sync_id == gyro.sync_id) {
                        matched_frame = frame;
                        matched_gyro = gyro;
                        found_match = true;
                        
                        std::cout << "匹配成功 - 同步ID: " << frame.sync_id 
                                  << ", 帧时间戳: " << frame.timestamp 
                                  << ", IMU时间戳: " << timePointToInt64(gyro.timestamp) << std::endl;
                        
                        break;
                    }
                }
                if (found_match) break;
            }
        }

        if (found_match) {
            successful_matches++;
            
            std::cout << "[MATCH] 同步ID: " << std::setw(5) << matched_frame.sync_id 
                      << " | 帧时间戳: " << std::setw(13) << matched_frame.timestamp 
                      << " | IMU时间戳: " << std::setw(13) << timePointToInt64(matched_gyro.timestamp) 
                      << " | 时差: " << std::setw(8) << (matched_frame.timestamp - timePointToInt64(matched_gyro.timestamp)) 
                      << "ms" << std::endl;
            
            // 传递帧和匹配的陀螺仪数据包
            process_matched_pair(matched_frame.frame, matched_gyro.packet, matched_gyro.timestamp);

            // 清理已处理的数据
            {
                boost::lock_guard<boost::mutex> lock(frame_queue_mutex);
                auto it = frame_queue.begin();
                while (it != frame_queue.end()) {
                    if (it->sync_id <= matched_frame.sync_id) {
                        it = frame_queue.erase(it);
                    } else {
                        ++it;
                    }
                }
                
                if (frame_queue.empty()) {
                    new_frame_available.store(false, std::memory_order_release);
                }
            }

            {
                boost::lock_guard<boost::mutex> lock(gyro_queue_mutex);
                auto it = gyro_queue.begin();
                while (it != gyro_queue.end()) {
                    if (it->sync_id <= matched_gyro.sync_id) {
                        it = gyro_queue.erase(it);
                    } else {
                        ++it;
                    }
                }
                
                if (gyro_queue.empty()) {
                    new_gyro_available.store(false, std::memory_order_release);
                }
            }
        } else {
            // 如果找不到匹配，清理过时的数据
            boost::lock_guard<boost::mutex> lock(frame_queue_mutex);
            boost::lock_guard<boost::mutex> lock2(gyro_queue_mutex);
            
            if (!frame_queue.empty() && !gyro_queue.empty()) {
                // 查找每个队列中最小的ID
                uint64_t min_frame_id = frame_queue.front().sync_id;
                uint64_t min_gyro_id = gyro_queue.front().sync_id;
                
                // 删除ID较小的项，因为它们可能永远不会匹配
                if (min_frame_id < min_gyro_id) {
                    frame_queue.pop_front();
                    std::cout << "丢弃过时的帧ID: " << min_frame_id << std::endl;
                } else if (min_gyro_id < min_frame_id) {
                    gyro_queue.pop_front();
                    std::cout << "丢弃过时的IMU数据ID: " << min_gyro_id << std::endl;
                }
            }
        }
    }

    {
        boost::lock_guard<boost::mutex> lock(cout_mutex);
        std::cout << "数据匹配线程已结束. 总尝试次数: " << total_attempts 
                  << ", 成功匹配次数: " << successful_matches << std::endl;
    }
}

void result_processing_thread_func() {
    // 添加切换机制的配置选项
    bool use_tf2_transform = false; // 默认使用tf2
    
    // 预先分配消息对象以减少内存分配
    auto armors_msg = std::make_unique<autoaim_interfaces::msg::Armors>();
    visualization_msgs::msg::MarkerArray marker_array;
    
    // 创建装甲板marker模板
    visualization_msgs::msg::Marker armor_marker;
    armor_marker.ns = "armors";
    armor_marker.action = visualization_msgs::msg::Marker::ADD;
    armor_marker.type = visualization_msgs::msg::Marker::CUBE;
    armor_marker.scale.x = 0.05;
    armor_marker.scale.z = 0.125;
    armor_marker.color.a = 1.0;
    armor_marker.color.g = 0.5;
    armor_marker.color.b = 1.0;
    armor_marker.lifetime = rclcpp::Duration::from_seconds(0.1);
    
    // 创建文本marker模板
    visualization_msgs::msg::Marker text_marker;
    text_marker.ns = "classification";
    text_marker.action = visualization_msgs::msg::Marker::ADD;
    text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text_marker.scale.z = 0.1;
    text_marker.color.a = 1.0;
    text_marker.color.r = 1.0;
    text_marker.color.g = 1.0;
    text_marker.color.b = 1.0;
    text_marker.lifetime = rclcpp::Duration::from_seconds(0.1);
    

    while (running) {
        std::tuple<std::chrono::high_resolution_clock::time_point, helios::MCUPacket, std::shared_future<FrameData>> armor_result_tuple;
        bool has_result = false;
        
        {
                boost::lock_guard<boost::mutex> lock(armor_pending_results_mutex);
                if (!armor_pending_results.empty()) {
                    armor_result_tuple = armor_pending_results.front();
                    armor_pending_results.pop_front();
                    has_result = true;
                    
                    auto ts_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(
                        std::get<0>(armor_result_tuple)).time_since_epoch().count();
                    std::cout << "Processing detection result - Timestamp: " << ts_ms 
                              << ", Queue size: " << armor_pending_results.size() << std::endl;}
        }
        
        
        
        if (!has_result) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        
        auto& timestamp = std::get<0>(armor_result_tuple);
        auto& gyro_data = std::get<1>(armor_result_tuple);
        auto& future = std::get<2>(armor_result_tuple);
        auto ts_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(timestamp).time_since_epoch().count();
        if(gyro_data.autoaim_mode != 0) {
            continue; // 跳过非自瞄模式的数据
        }
        
        std::future_status status = future.wait_for(std::chrono::milliseconds(5));
        if (status != std::future_status::ready) {
            std::cout << "Detection result not ready yet - Timestamp: " << ts_ms 
                      << ", pushing back to queue" << std::endl;
            boost::lock_guard<boost::mutex> lock(armor_pending_results_mutex);
            armor_pending_results.push_back(armor_result_tuple);
            continue;
        }
        std::cout << "Detection result is ready - Timestamp: " << ts_ms << std::endl;
        FrameData result = future.get();
        std::cout << "TRANSFORM DATA - Timestamp: " << ts_ms 
          << ", Yaw: " << gyro_data.yaw 
          << ", Pitch: " << gyro_data.pitch 
          << ", Mode: " << (int)gyro_data.autoaim_mode 
          << std::endl;
        // 转换时间戳为ROS时间
        auto timestamp_ns = std::chrono::time_point_cast<std::chrono::nanoseconds>(timestamp).time_since_epoch().count();
        rclcpp::Time ros_time(timestamp_ns);
        

        
        cv::Quatd odom2cam_rotation;
        cv::Vec3d odom2cam_translation;
        cv::Quatd cam2odom_rotation;
        cv::Vec3d cam2odom_translation;
        double yaw = 0.0;
        struct helios_cv::ArmorTransformInfo armor_transform_info(odom2cam_rotation, cam2odom_rotation);
        if (use_tf2_transform) {
            geometry_msgs::msg::TransformStamped ts_odom2cam, ts_cam2odom;
            try {
                ts_odom2cam = tf2_buffer->lookupTransform(
                    "camera_optical_frame",
                    "odoom",
                    ros_time,
                    rclcpp::Duration::from_seconds(0.05)
                );
                
                ts_cam2odom = tf2_buffer->lookupTransform(
                    "odoom",
                    "camera_optical_frame",
                    ros_time,
                    rclcpp::Duration::from_seconds(0.05)
                );
                
                auto odom2yawlink = tf2_buffer->lookupTransform(
                    "yaw_link",
                    "odoom",
                    ros_time,
                    rclcpp::Duration::from_seconds(0.05)
                );
            
                tf2::Quaternion q(
                    odom2yawlink.transform.rotation.x,
                    odom2yawlink.transform.rotation.y,
                    odom2yawlink.transform.rotation.z,
                    odom2yawlink.transform.rotation.w
                );
                tf2::Matrix3x3 m(q);
                double roll, pitch;
                m.getRPY(roll, pitch, yaw);
                
                // 转换TF2变换为OpenCV格式
                odom2cam_rotation = ros2cv(ts_odom2cam.transform.rotation);
                cam2odom_rotation = ros2cv(ts_cam2odom.transform.rotation);
                
            } catch (const tf2::TransformException& ex) {
                boost::lock_guard<boost::mutex> lock(cout_mutex);
                std::cerr << "Error while transforming with TF2: " << ex.what() << std::endl;
                std::cerr << "Falling back to TransformManager..." << std::endl;
                
                // transform_manager->getOdom2Cam(odom2cam_rotation, odom2cam_translation);
                // transform_manager->getCam2Odom(cam2odom_rotation, cam2odom_translation);
            }
        } else {
            // transform_manager->getOdom2Cam(odom2cam_rotation, odom2cam_translation);
            // transform_manager->getCam2Odom(cam2odom_rotation, cam2odom_translation);
            bool transform_success = transform_manager->getTransforms(
                timestamp_ns,
                cam2odom_rotation, cam2odom_translation,
                odom2cam_rotation, odom2cam_translation
            );
            
            if (transform_success) {
                helios_cv::ArmorTransformInfo armor_transform_info(odom2cam_rotation, cam2odom_rotation);
                armor_pnp_solver->update_transform_info(&armor_transform_info);
            //TODO :get yaw
            }
        }
        
        armors_msg->armors.clear();
        armors_msg->header.stamp = ros_time;
        armors_msg->header.frame_id = "camera_optical_frame";
        
        if (result.has_valid_results()) {
            try {
                boost::lock_guard<boost::mutex> lock(pnp_mutex);
                if (armor_pnp_solver) {
                    std::cout << "Processing " << result.armors.size() << " armors - Timestamp: " << ts_ms << std::endl;
                    std::cout << "TRANSFORM DATA in has result  - Timestamp: " << ts_ms 
                        << ", Yaw: " << gyro_data.yaw 
                        << ", Pitch: " << gyro_data.pitch 
                        << ", Mode: " << (int)gyro_data.autoaim_mode 
                        
                        << " \n "<<std::endl;
                    
                    for (const auto& armor : result.armors) {
                        cv::Mat rvec, tvec;
                        bool pose_solved = armor_pnp_solver->solve_pose(armor, rvec, tvec);
                        
                        if (pose_solved) {
                            autoaim_interfaces::msg::Armor armor_msg;
                            
                            armor_msg.type = static_cast<int>(armor.type);
                            armor_msg.number = armor.number;

                            armor_msg.pose.position.x = tvec.at<double>(0);
                            armor_msg.pose.position.y = tvec.at<double>(1);
                            armor_msg.pose.position.z = tvec.at<double>(2);

                            cv::Mat rotation_matrix;
                            if (armor_pnp_solver->use_projection()) {
                                rotation_matrix = rvec;
                            } else {
                                cv::Rodrigues(rvec, rotation_matrix);
                            }
                            tf2::Matrix3x3 tf2_rotation_matrix(
                                rotation_matrix.at<double>(0, 0),
                                rotation_matrix.at<double>(0, 1),
                                rotation_matrix.at<double>(0, 2),
                                rotation_matrix.at<double>(1, 0),
                                rotation_matrix.at<double>(1, 1),
                                rotation_matrix.at<double>(1, 2),
                                rotation_matrix.at<double>(2, 0),
                                rotation_matrix.at<double>(2, 1),
                                rotation_matrix.at<double>(2, 2)
                            );
                            tf2::Quaternion tf2_quaternion;
                            tf2_rotation_matrix.getRotation(tf2_quaternion);
                            armor_msg.pose.orientation = tf2::toMsg(tf2_quaternion);
                            armor_msg.distance_to_image_center =
                                armor_pnp_solver->calculateDistanceToCenter(armor.center);

                            armors_msg->armors.push_back(armor_msg);
                        }
                    }
                    
                    if (!armors_msg->armors.empty()) {
                        // 添加调试可视化
                        armor_pnp_solver->draw_projection_points(result.frame);
                        cv::imshow("Armor Projection", result.frame);
                        cv::waitKey(1);
                        
                        // 发布装甲板markers
                        marker_array.markers.clear();
                        int marker_id = 0;
                        for (const auto& armor : armors_msg->armors) {
                        
                            armor_marker.id = marker_id++;
                            
                            armor_marker.header.stamp = ros_time;
                            armor_marker.header.frame_id = "camera_optical_frame"; 
                            armor_marker.scale.y = (armor.type == 0) ? 0.135 : 0.23; 
                            armor_marker.pose = armor.pose;
                            marker_array.markers.push_back(armor_marker);
                        
                            // 编号文本marker
                            text_marker.id = marker_id++;
                            // 修复：正确设置 header
                            text_marker.header.stamp = ros_time;
                            text_marker.header.frame_id = "camera_optical_frame"; 
                            text_marker.pose.position = armor.pose.position;
                            text_marker.pose.position.y -= 0.1; 
                            text_marker.text = armor.number;
                            marker_array.markers.push_back(text_marker);
                        }
                    } else {
                        // 如果没有检测到装甲板，发送一个DELETE action的marker清除之前的markers
                        marker_array.markers.clear();
                        armor_marker.action = visualization_msgs::msg::Marker::DELETE;
                        armor_marker.id = 0;
                        armor_marker.header = armors_msg->header;
                        marker_array.markers.push_back(armor_marker);
                        
                        // 重置回ADD action以备下次使用
                        armor_marker.action = visualization_msgs::msg::Marker::ADD;
                    }
                    
                    // 发布markers
                    marker_pub_->publish(marker_array);
                }
            } catch (const cv::Exception& e) {
                boost::lock_guard<boost::mutex> lock(cout_mutex);
                std::cerr << "OpenCV 异常: " << e.what() << std::endl;

            } catch (const std::exception& e) {
                boost::lock_guard<boost::mutex> lock(cout_mutex);
                std::cerr << "标准异常: " << e.what() << std::endl;
            }
        }
        
        // 发布装甲板消息
        armors_pub_->publish(*armors_msg);
    }
}
bool load_camera_parameters_yaml(const std::string& yaml_file, 
                               std::array<double, 9>& camera_matrix,
                               std::vector<double>& dist_coeffs) {
    try {
        // 检查文件是否存在
        std::ifstream file(yaml_file);
        if (!file.good()) {
            std::cerr << "Failed to open camera calibration file: " << yaml_file << std::endl;
            return false;
        }
        file.close();

        YAML::Node config = YAML::LoadFile(yaml_file);

        if (!config["camera_matrix"] || !config["camera_matrix"]["data"] || 
            !config["distortion_coefficients"] || !config["distortion_coefficients"]["data"]) {
            std::cerr << "Missing required nodes in camera calibration file" << std::endl;
            return false;
        }

        auto cm_node = config["camera_matrix"]["data"];
        if (!cm_node.IsSequence() || cm_node.size() != 9) {
            std::cerr << "Invalid camera_matrix data format or size" << std::endl;
            return false;
        }

        for (int i = 0; i < 9; i++) {
            camera_matrix[i] = cm_node[i].as<double>();
        }

        auto dist_node = config["distortion_coefficients"]["data"];
        if (!dist_node.IsSequence() || dist_node.size() == 0) {
            std::cerr << "Invalid distortion_coefficients data format or empty" << std::endl;
            return false;
        }

        dist_coeffs.resize(dist_node.size());
        for (size_t i = 0; i < dist_node.size(); i++) {
            dist_coeffs[i] = dist_node[i].as<double>();
        }

        return true;

    } catch (const YAML::Exception& e) {
        std::cerr << "YAML parsing error when loading camera parameters: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error when loading camera parameters: " << e.what() << std::endl;
        return false;
    }
}



std::atomic<bool> received_sigint(false);


void sigint_handler(int signum) {
    (void)signum;  
    received_sigint.store(true);
    running.store(false);
    std::cout << "Received shutdown signal, terminating gracefully..." << std::endl;
}

int main(int argc, char* argv[]) {
    std::string port_name = "/dev/ttyACM0";  
    int baud_rate = 921600;                 
    
    // 模型和标定文件路径
    std::string model_path = "/home/zyi/Downloads/0405_9744.onnx"; 
    // std::string model_path = "/home/helios/Desktop/0405_9744.onnx"; 
    std::string device_name = "GPU";
    std::string calibration_file = "/home/zyi/cs016_8mm.yaml";
    // std::string calibration_file = "/home/helios/Desktop/cs016_8mm.yaml";
    
    // Initialize ROS 2 with disabled signal handlers
    rclcpp::InitOptions init_options;
    init_options.shutdown_on_signal = false;  // Corrected from shutdown_on_sigint
    rclcpp::init(argc, argv, init_options);
    
    // Register our own signal handler
    signal(SIGINT, sigint_handler);
    
    auto node = std::make_shared<rclcpp::Node>("helios_vision_node");
    
    // Get parameters from ROS2 if specified, otherwise use defaults
    // node->declare_parameter("port_name", port_name);
    // node->declare_parameter("baud_rate", baud_rate);
    // node->declare_parameter("model_path", model_path);
    // node->declare_parameter("device_name", device_name);
    // node->declare_parameter("calibration_file", calibration_file);
    
    // // Only override values if they're explicitly set in the ROS 2 parameter system
    // if (node->has_parameter("port_name")) 
    //     port_name = node->get_parameter("port_name").as_string();
    // if (node->has_parameter("baud_rate")) 
    //     baud_rate = node->get_parameter("baud_rate").as_int();
    // if (node->has_parameter("model_path")) 
    //     model_path = node->get_parameter("model_path").as_string();
    // if (node->has_parameter("device_name")) 
    //     device_name = node->get_parameter("device_name").as_string();
    // if (node->has_parameter("calibration_file")) 
    //     calibration_file = node->get_parameter("calibration_file").as_string();
    
    // Continue to support command line arguments which override everything
    // if (argc > 1) {
    //     port_name = argv[1];
    // }
    
    // if (argc > 2) {
    //     baud_rate = std::stoi(argv[2]);
    // }
    
    // if (argc > 3) {
    //     model_path = argv[3];
    // }
    
    // if (argc > 4) {
    //     calibration_file = argv[4];
    // }
    armors_pub_ = node->create_publisher<autoaim_interfaces::msg::Armors>(
        "detector/armors", 
        rclcpp::QoS(30).best_effort()  // Increased queue size, best effort delivery
    );
    // 初始化TF2cx
    tf2_buffer = std::make_shared<tf2_ros::Buffer>(node->get_clock());

    transform_manager = std::make_unique<TimestampedTransformManager>();



    tf2_buffer = std::make_shared<tf2_ros::Buffer>(node->get_clock());
    tf2_listener = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer);
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(*node);
    marker_pub_ = node->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/detector/marker", 10);
    // Print the settings that will be used
    std::cout << "Using settings:" << std::endl;
    std::cout << "  Port: " << port_name << std::endl;
    std::cout << "  Baud rate: " << baud_rate << std::endl;
    std::cout << "  Model path: " << model_path << std::endl;
    std::cout << "  Calibration file: " << calibration_file << std::endl;
    
    // 加载相机参数
    std::array<double, 9> camera_matrix;
    std::vector<double> dist_coeffs;
    
    if (!load_camera_parameters_yaml(calibration_file, camera_matrix, dist_coeffs)) {
        std::cerr << "Failed to load camera parameters. Exiting." << std::endl;
        rclcpp::shutdown();
        return EXIT_FAILURE;
    }
    // 初始化检测器
    if (!initialize_detector(model_path)) {
        std::cerr << "Failed to initialize the detector. Exiting." << std::endl;
        rclcpp::shutdown();
        return EXIT_FAILURE;
    }
    
    // 初始化PnP解算器
    if (!initialize_pnp_solvers(camera_matrix, dist_coeffs)) {
        std::cerr << "Failed to initialize PnP solvers. Exiting." << std::endl;
        rclcpp::shutdown();
        return EXIT_FAILURE;
    }
    

    
    program_start_time = std::chrono::high_resolution_clock::now();
    
    // 创建线程
    boost::thread sync_thread(start_sync); 
    boost::thread cam_thread(camera_thread_func);
    boost::thread serial_thread(serial_thread_func, port_name, baud_rate, node);
    boost::thread matching_thread(data_matching_thread_func);
    boost::thread result_thread(result_processing_thread_func);  
    
    // 设置线程亲和性以提高性能
    if (cam_thread.native_handle()) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(0, &cpuset);
        CPU_SET(1, &cpuset);
        CPU_SET(2, &cpuset);
        pthread_setaffinity_np(cam_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    }

    if (serial_thread.native_handle()) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(3, &cpuset);
        CPU_SET(4, &cpuset);
        pthread_setaffinity_np(serial_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    }

    if (matching_thread.native_handle()) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(5, &cpuset);
        CPU_SET(6, &cpuset);
        pthread_setaffinity_np(matching_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    }

    if (result_thread.native_handle()) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(7, &cpuset);  // Use a less busy core
        CPU_SET(8, &cpuset);  // Add another core
        pthread_setaffinity_np(result_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
        
        // Set higher priority
        struct sched_param params;
        params.sched_priority = sched_get_priority_max(SCHED_FIFO) - 1;
        pthread_setschedparam(result_thread.native_handle(), SCHED_FIFO, &params);
    }
    
    // Main loop to check for shutdown
    std::cout << "All threads started. Press Ctrl+C to exit." << std::endl;
    
    while (rclcpp::ok() && !received_sigint.load()) {
        rclcpp::spin_some(node);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Set running to false to signal all threads to terminate
    std::cout << "Shutting down all threads..." << std::endl;
    running.store(false);
    
    // Join threads with timeout - fixed to remove the error_code parameter
    auto join_with_timeout = [](boost::thread& t, const std::string& name) {
        if (t.joinable()) {
            std::cout << "Waiting for " << name << " thread to terminate..." << std::endl;
            
            // try_join_for returns a boolean success indicator
            bool joined = t.try_join_for(boost::chrono::seconds(3));
            
            if (!joined) {
                std::cerr << "Thread " << name << " did not terminate gracefully, interrupting..." << std::endl;
                t.interrupt();
                t.try_join_for(boost::chrono::seconds(1));
            } else {
                std::cout << name << " thread terminated successfully." << std::endl;
            }
        }
    };
    join_with_timeout(sync_thread, "sync");
    join_with_timeout(cam_thread, "camera");
    join_with_timeout(serial_thread, "serial");
    join_with_timeout(matching_thread, "matching");
    join_with_timeout(result_thread, "result");
    
    // 清理资源
    {
        boost::lock_guard<boost::mutex> lock(detector_mutex);
        global_detector.reset();
    }
    
    {
        boost::lock_guard<boost::mutex> lock(pnp_mutex);
        armor_pnp_solver.reset();
        energy_pnp_solver.reset();
    }
    
    rclcpp::shutdown();
    std::cout << "程序已正常退出" << std::endl;
    return 0;
}
