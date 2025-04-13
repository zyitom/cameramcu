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

// 性能优化的关键参数
constexpr int MAX_QUEUE_SIZE = 10;         // 减少队列大小以降低内存占用
constexpr float TIME_MATCH_MIN_MS = 7;     // 时间戳匹配最小值 (ms)
constexpr float TIME_MATCH_MAX_MS = 9;     // 时间戳匹配最大值 (ms)
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

std::unique_ptr<YOLOXDetector> global_detector;
boost::mutex detector_mutex;

bool initialize_detector(const std::string& model_path, const std::string& device_name = "GPU") {
    try {
        boost::lock_guard<boost::mutex> lock(detector_mutex);
        global_detector = std::make_unique<YOLOXDetector>(model_path, device_name);
        return global_detector->initialize();
    } catch (const std::exception& e) {
        boost::lock_guard<boost::mutex> lock(cout_mutex);
        std::cerr << "Failed to initialize detector: " << e.what() << std::endl;
        return false;
    }
}

constexpr int MAX_PENDING_RESULTS = 5; // 限制挂起的结果数量
boost::mutex pending_results_mutex;

std::deque<std::tuple<std::chrono::high_resolution_clock::time_point, helios::MCUPacket, std::shared_future<FrameData>>> pending_results;

// 用全局异步检测函数替代原有的process_matched_pair函数
void process_matched_pair(const cv::Mat& frame, const helios::MCUPacket& gyro_data, std::chrono::high_resolution_clock::time_point timestamp) {
    FrameData input_data(timestamp, frame);
    
    // 异步提交检测任务（不阻塞主线程）
    std::shared_future<FrameData> shared_future;
    {
        boost::lock_guard<boost::mutex> lock(detector_mutex);
        if (global_detector) {
            // 将std::future转换为std::shared_future
            std::future<FrameData> future = global_detector->submit_frame_async(input_data);
            shared_future = future.share();
        } else {
            std::cerr << "Detector not initialized." << std::endl;
            return;
        }
    }
    
    // 添加到结果处理队列
    {
        boost::lock_guard<boost::mutex> lock(pending_results_mutex);
        pending_results.emplace_back(timestamp, gyro_data, shared_future);
        
        // 限制队列大小
        if (pending_results.size() > MAX_PENDING_RESULTS) {
            pending_results.pop_front();
        }
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
    std::string intrinsic_para_path = HIK_CALI_FILE_PATH"/caliResults/calibCameraData.yml";
    
    try {
        hikCam.Init(true, camera_config_path, intrinsic_para_path, false);
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
            // 快速路径：如果队列为空，直接添加而不需要锁
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
            
            cv::imshow("Raw Camera", frame);
            int key = cv::waitKey(1);
            if (key == 27) { 
                running = false;
                break;
            }
        }
    }
    
    {
        boost::lock_guard<boost::mutex> lock(cout_mutex);
        std::cout << "相机线程已结束" << std::endl;
    }
}





void serial_thread_func(const std::string& port_name, int baud_rate) {
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
        
        // 配置串口参数
        serial_port.set_option(boost::asio::serial_port_base::baud_rate(baud_rate), ec);
        serial_port.set_option(boost::asio::serial_port_base::character_size(8), ec);
        serial_port.set_option(boost::asio::serial_port_base::parity(boost::asio::serial_port_base::parity::none), ec);
        serial_port.set_option(boost::asio::serial_port_base::stop_bits(boost::asio::serial_port_base::stop_bits::one), ec);
        serial_port.set_option(boost::asio::serial_port_base::flow_control(boost::asio::serial_port_base::flow_control::none), ec);
        
        // 使用POSIX API设置非阻塞模式
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
                    
                    // 解析帧头
                    helios::FrameHeader header;
                    memcpy(&header, receive_buffer.data(), sizeof(helios::FrameHeader));
                    
                    // 简单验证，确保数据长度在合理范围内
                    if (header.data_length > 1024 || header.data_length < sizeof(helios::MCUPacket)) {
                        // 非法长度，删除SOF，继续查找下一个SOF
                        receive_buffer.erase(receive_buffer.begin());
                        continue;
                    }
                    
                    // 检查是否有完整的数据包(仅包含帧头和数据，忽略CRC16)
                    size_t total_packet_size = sizeof(helios::FrameHeader) + header.data_length;
                    if (receive_buffer.size() < total_packet_size) {
                        // 数据不完整，等待更多数据
                        break;
                    }
                    

                    auto now = std::chrono::high_resolution_clock::now();
                    
                    if (header.cmd_id == helios::RECEIVE_AUTOAIM_RECEIVE_CMD_ID && 
                        header.data_length == sizeof(helios::MCUPacket)) {
                        
                        helios::MCUPacket packet;
                        memcpy(&packet, receive_buffer.data() + sizeof(helios::FrameHeader), sizeof(helios::MCUPacket));
                        {
                            boost::lock_guard<boost::mutex> lock(gyro_queue_mutex);
                            gyro_queue.push_back(TimestampedPacket(packet, now));
                            new_gyro_available.store(true, std::memory_order_release);
                        }
                    }
                    
                    // 删除已处理的数据包
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
        
        // 短暂休眠以减少CPU使用
        if (bytes_read == 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
    
    // 关闭串口
    if (serial_port.is_open()) {
        try {
            boost::system::error_code close_ec;
            serial_port.close(close_ec);
        } catch (...) {
            // 忽略关闭错误
        }
    }
    
    {
        boost::lock_guard<boost::mutex> lock(cout_mutex);
        std::cout << "串口线程已结束" << std::endl;
    }
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
// 结果处理线程 - 新增线程处理异步结果
void result_processing_thread_func() {
    while (running) {
        // 处理队列中的一个结果
        std::tuple<std::chrono::high_resolution_clock::time_point, helios::MCUPacket, std::shared_future<FrameData>> result_tuple;
        bool has_result = false;
        
        {
            boost::lock_guard<boost::mutex> lock(pending_results_mutex);
            if (!pending_results.empty()) {
                result_tuple = pending_results.front();  // 复制是安全的，因为使用shared_future
                pending_results.pop_front();
                has_result = true;
            }
        }
        
        if (has_result) {
            auto& future = std::get<2>(result_tuple);
            
            try {
                // 等待结果完成，但设置超时以避免阻塞
                std::future_status status = future.wait_for(std::chrono::milliseconds(50));
                
                if (status == std::future_status::ready) {
                    // 处理完成的结果，例如显示或后处理
                    FrameData result = future.get();
                    cv::imshow("Raw Casdfsdfsdfmera", result.frame);
                    if (result.has_valid_results()) {
                        cv::Mat visual_result = global_detector->visualize_detection_result(result);
                        cv::imshow("Detection Result", visual_result);
                        cv::waitKey(1);
                    }
                } else {
                    // 如果还没准备好，放回队列末尾
                    boost::lock_guard<boost::mutex> lock(pending_results_mutex);
                    pending_results.push_back(result_tuple); 
                }
            } catch (const std::exception& e) {
                boost::lock_guard<boost::mutex> lock(cout_mutex);
                std::cerr << "Error processing result: " << e.what() << std::endl;
            }
        } else {
            // 如果队列为空，短暂休眠
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
}

int main(int argc, char* argv[]) {
    std::string port_name = "/dev/ttyACM0";  
    int baud_rate = 921600;                 
    
    // Add paths for your detector model
    std::string model_path = "/home/zyi/Downloads/fast-big1.onnx"; // Replace with your actual model path
    std::string device_name = "GPU";
    
    if (argc > 1) {
        port_name = argv[1];
    }
    
    if (argc > 2) {
        baud_rate = std::stoi(argv[2]);
    }
    
    if (argc > 3) {
        model_path = argv[3];
    }
    

    if (!initialize_detector(model_path, device_name)) {
        std::cerr << "Failed to initialize the detector. Exiting." << std::endl;
        return EXIT_FAILURE;
    }
    
    // 创建必要的线程
    boost::thread cam_thread(camera_thread_func);
    boost::thread serial_thread(serial_thread_func, port_name, baud_rate);
    boost::thread matching_thread(data_matching_thread_func);
    boost::thread result_thread(result_processing_thread_func);  // 新增结果处理线程
    
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
        pthread_setaffinity_np(serial_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    }

    if (matching_thread.native_handle()) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(0, &cpuset);
        CPU_SET(1, &cpuset);
        pthread_setaffinity_np(matching_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    }

    if (result_thread.native_handle()) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(2, &cpuset);
        pthread_setaffinity_np(result_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    }
    
    cam_thread.join();
    serial_thread.join();
    matching_thread.join();
    result_thread.join();
    
    {
        boost::lock_guard<boost::mutex> lock(detector_mutex);
        global_detector.reset();
    }
    
    std::cout << "程序已正常退出" << std::endl;
    return 0;
}