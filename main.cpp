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


constexpr int MAX_QUEUE_SIZE = 10;         // 减少队列大小以降低内存占用
constexpr float TIME_MATCH_MIN_MS = 7;     // 时间戳匹配最小值 (ms)
constexpr float TIME_MATCH_MAX_MS = 29;     // 时间戳匹配最大值 (ms)
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
struct alignas(64) TimestampedPacket {
    helios::MCUPacket packet;
    std::chrono::high_resolution_clock::time_point timestamp;

    TimestampedPacket() 
        : packet(), timestamp(std::chrono::high_resolution_clock::time_point::min()) {}

    TimestampedPacket(const helios::MCUPacket& p, std::chrono::high_resolution_clock::time_point ts_ms) 
        : packet(p), timestamp(ts_ms) {}
};

struct alignas(64) TimestampedFrame {    
    cv::Mat frame;
    int64_t timestamp;

    TimestampedFrame() 
        : frame(), timestamp(-1) {}

    TimestampedFrame(const cv::Mat& f, int64_t ts) : frame(f), timestamp(ts) {}
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

std::deque<std::tuple<std::chrono::high_resolution_clock::time_point, helios::MCUPacket, std::shared_future<FrameData>>> armor_pending_results;

// 用全局异步检测函数替代原有的process_matched_pair函数
void process_matched_pair(const cv::Mat& frame, const helios::MCUPacket& gyro_data, std::chrono::high_resolution_clock::time_point timestamp) {
    FrameData input_data(timestamp, frame);
    if(gyro_data.autoaim_mode == 0){
        auto ts_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(timestamp).time_since_epoch().count();
    std::cout << "Processing matched pair - Timestamp: " << ts_ms 
              << ", Autoaim mode: " << (int)gyro_data.autoaim_mode 
              << ", Yaw: " << gyro_data.yaw 
              << ", Pitch: " << gyro_data.pitch << std::endl;
        std::shared_future<FrameData> shared_future;
        {
            boost::lock_guard<boost::mutex> lock(detector_mutex);
            if (global_detector) {
                // 将std::future转换为std::shared_future
                std::cout << "Submitting frame to detector - Timestamp: " << ts_ms << std::endl;
                
                // 将std::future转换为std::shared_future
                std::future<FrameData> future = global_detector->submit_frame_async(input_data);
                shared_future = future.share();
                
                std::cout << "Frame submitted successfully - Timestamp: " << ts_ms << std::endl;
                
            } else {
                std::cerr << "Detector not initialized." << std::endl;
                return;
            }
        }
        
        {
            boost::lock_guard<boost::mutex> lock(armor_pending_results_mutex);
            armor_pending_results.emplace_back(timestamp, gyro_data, shared_future);
            std::cout << "Added result to pending queue - Size: " << armor_pending_results.size() 
                      << ", Timestamp: " << ts_ms << std::endl;
            // 限制队列大小
            if (armor_pending_results.size() > MAX_PENDING_RESULTS) {
                auto dropped_ts = std::chrono::time_point_cast<std::chrono::milliseconds>(
                    std::get<0>(armor_pending_results.front())).time_since_epoch().count();
                armor_pending_results.pop_front();
                std::cout << "WARNING: Queue size exceeded limit, dropped result with timestamp: " 
                          << dropped_ts << std::endl;
            }
        }
    }
    else{
        boost::lock_guard<boost::mutex> lock(cout_mutex);
        std::cout << "Autoaim mode is 1, skipping detection." << std::endl;
        {
            // boost::lock_guard<boost::mutex> lock(pending_results_mutex);
            // //pending_results.emplace_back(armor)
            // if (pending_results.size() > MAX_PENDING_RESULTS) {
            //     pending_results.pop_front();
            // }
        }
        //打符

        
    }
}

// 相机线程函数 - 优化版
void camera_thread_func() {
    pthread_t this_thread = pthread_self();
    struct sched_param params;
    params.sched_priority = sched_get_priority_max(SCHED_FIFO);
    pthread_setschedparam(this_thread, SCHED_FIFO, &params);
    
    camera::HikCamera hikCam;
    auto time_start = std::chrono::high_resolution_clock::now();
    std::string camera_config_path = HIK_CONFIG_FILE_PATH"/camera_config.yaml";
    // std::string intrinsic_para_path = HIK_CALI_FILE_PATH"/caliResults/calibCameraData.yml";
    
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
    
    // 主处理循环
    cv::Mat frame;
    uint64_t device_ts;
    int64_t host_ts;
    
    while (running) {
        bool hasNew = hikCam.ReadImg(frame, &device_ts, &host_ts);
        
        if (hasNew && !frame.empty()) {
            #ifdef DEBUG
            std::cout << " Host: " << host_ts << std::endl;
            #endif
            if (frame_queue.empty() && !new_frame_available.load(std::memory_order_relaxed)) {
                boost::lock_guard<boost::mutex> lock(frame_queue_mutex);
                frame_queue.push_back(TimestampedFrame(frame, host_ts));
                new_frame_available.store(true, std::memory_order_release);
            } else {
                boost::lock_guard<boost::mutex> lock(frame_queue_mutex);
                // boost::circular_buffer会自动处理大小限制
                frame_queue.push_back(TimestampedFrame(frame, host_ts));
                new_frame_available.store(true, std::memory_order_release);
            }
            
            // cv::imshow("Raw Camera", frame);
            // cv::waitKey(1);
            
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
    
    // 等待相机初始化完成
    while (running && !camera_initialized.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
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
                        
                        helios::MCUPacket packet;
                        memcpy(&packet, receive_buffer.data() + sizeof(helios::FrameHeader), sizeof(helios::MCUPacket));
                        
                        // 获取当前系统时间作为接收时间戳
                        auto now = std::chrono::high_resolution_clock::now();
                        #ifdef DEBUG
                        auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now).time_since_epoch().count();
                        std::cout << "Serial data timestamp (milliseconds): " << now_ms << std::endl;
                        #endif
                        // 自身颜色 0 蓝 1 红
                        static bool last_is_blue = true; // 默认蓝队
                        bool current_is_blue = (packet.self_color == 1);
                        
                        // 只有当颜色发生变化时才更新检测器的颜色设置
                        if (current_is_blue != last_is_blue) {
                            boost::lock_guard<boost::mutex> lock(detector_mutex);
                            if (global_detector) {
                                global_detector->set_is_blue(current_is_blue);
                            }
                            last_is_blue = current_is_blue;
                        }
                        

                        {
                            boost::lock_guard<boost::mutex> lock(gyro_queue_mutex);
                            gyro_queue.push_back(TimestampedPacket(packet, now));
                            new_gyro_available.store(true, std::memory_order_release);
                        }
                        
                        
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
void data_matching_thread_func() {
    // 等待相机初始化完成
    while (running && !camera_initialized.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    if (!running) {
        return;
    }

    {
        boost::lock_guard<boost::mutex> lock(cout_mutex);
        std::cout << "开始数据匹配线程" << std::endl;
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

        TimestampedFrame matched_frame;  // 初始化为默认值
        TimestampedPacket matched_gyro;  // 初始化为默认值
        bool found_match = false;
        int64_t best_time_diff = INT64_MAX;

        {
            boost::lock_guard<boost::mutex> lock(frame_queue_mutex);
            boost::lock_guard<boost::mutex> lock2(gyro_queue_mutex);

            if (!frame_queue.empty() && !gyro_queue.empty()) {
                
                for (const auto& frame : frame_queue) {
                    for (const auto& gyro : gyro_queue) {
                        int64_t time_diff = frame.timestamp - timePointToInt64(gyro.timestamp);
                        
                        if (time_diff >= TIME_MATCH_MIN_MS && time_diff <= TIME_MATCH_MAX_MS) {
                            if (std::abs(time_diff) < std::abs(best_time_diff)) {
                                matched_frame = TimestampedFrame(frame.frame, frame.timestamp);  
                                matched_gyro = TimestampedPacket(gyro.packet, gyro.timestamp);
                                best_time_diff = time_diff;
                                found_match = true;
                                std::cout << "Matched gyro data - Yaw: " << matched_gyro.packet.yaw 
                                << ", Pitch: " << matched_gyro.packet.pitch 
                                << ", Mode: " << (int)matched_gyro.packet.autoaim_mode 
                                << "time_stamp: " << timePointToInt64(matched_gyro.timestamp) << std::endl;
                                #ifdef DEBUG
                                std::cout << "Matched pair - Frame timestamp: " << matched_frame.timestamp 
                                << ", Gyro timestamp: " << timePointToInt64(matched_gyro.timestamp) 
                                << ", Time difference: " << best_time_diff << "ms" << std::endl;
                                #endif
                            }
                        }
                    }
                }
            }
        }

        if (found_match) {
            successful_matches++;

            // {
            //     boost::lock_guard<boost::mutex> lock(cout_mutex);
            //     std::cout << "找到匹配: 时间差 = " << best_time_diff << "ms" << std::endl;
            // }
            
            process_matched_pair(matched_frame.frame, matched_gyro.packet, matched_gyro.timestamp);

            // 清理处理过的数据
            {
                boost::lock_guard<boost::mutex> lock(frame_queue_mutex);
                while (!frame_queue.empty() && frame_queue.front().timestamp <= matched_frame.timestamp) {
                    frame_queue.pop_front();
                }
                if (frame_queue.empty()) {
                    new_frame_available.store(false, std::memory_order_release);
                }
            }

            {
                boost::lock_guard<boost::mutex> lock(gyro_queue_mutex);
                while (!gyro_queue.empty() && gyro_queue.front().timestamp <= matched_gyro.timestamp) {
                    gyro_queue.pop_front();
                }
                if (gyro_queue.empty()) {
                    new_gyro_available.store(false, std::memory_order_release);
                }
            }
        } else {
            // 清理过时数据
            {
                boost::lock_guard<boost::mutex> lock(frame_queue_mutex);
                boost::lock_guard<boost::mutex> lock2(gyro_queue_mutex);

                if (!frame_queue.empty() && !gyro_queue.empty()) {
                    const auto& oldest_frame = frame_queue.front();
                    const auto& oldest_gyro = gyro_queue.front();

                    int64_t oldest_gyro_timestamp = timePointToInt64(oldest_gyro.timestamp);

                    // 如果 frame 的时间戳比 gyro 的时间戳早太多，移除 frame
                    if (oldest_frame.timestamp + static_cast<int64_t>(TIME_MATCH_MAX_MS) < oldest_gyro_timestamp) {
                        frame_queue.pop_front();
                    }

                    // 如果 gyro 的时间戳比 frame 的时间戳早太多，移除 gyro
                    if (oldest_gyro_timestamp + static_cast<int64_t>(TIME_MATCH_MAX_MS) < oldest_frame.timestamp) {
                        gyro_queue.pop_front();
                    }
                }
            }
        }
    }

    {
        boost::lock_guard<boost::mutex> lock(cout_mutex);
        std::cout << "数据匹配线程已结束" << std::endl;
    }
}

void result_processing_thread_func() {
    // 添加切换机制的配置选项
    bool use_tf2_transform = true; // 默认使用tf2
    
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
                        // armor_pnp_solver->draw_projection_points(result.frame);
                        // cv::imshow("Armor Projection", result.frame);
                        // cv::waitKey(1);
                        
                        // 发布装甲板markers
                        marker_array.markers.clear();
                        int marker_id = 0;
                        for (const auto& armor : armors_msg->armors) {
                            auto time_now = std::chrono::high_resolution_clock::now();
                            auto marker_ns = std::chrono::time_point_cast<std::chrono::nanoseconds>(time_now).time_since_epoch().count();
                            rclcpp::Time ros_time(marker_ns);
                        
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
    std::string device_name = "GPU";
    std::string calibration_file = "/home/zyi/cs016_8mm.yaml";
    
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
    

    
    // 创建必要的线程
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
