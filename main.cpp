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

#include <boost/thread/mutex.hpp>
#include <boost/thread/lock_guard.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/lockfree/queue.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/thread/thread.hpp>
#include <boost/bind.hpp>
#include <future> 
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
    
    serial::Serial serial_port;
    try {
        serial_port.setPort(port_name);
        serial_port.setBaudrate(baud_rate);
        serial::Timeout timeout = serial::Timeout::simpleTimeout(50); 
        serial_port.setTimeout(timeout);
        serial_port.open();
        
        if (!serial_port.isOpen()) {
            boost::lock_guard<boost::mutex> lock(cout_mutex);
            std::cerr << "Failed to open serial port." << std::endl;
            running = false;
            return;
        }
    } catch (const std::exception& e) {
        {
            boost::lock_guard<boost::mutex> lock(cout_mutex);
            std::cerr << "Failed to open serial port: " << e.what() << std::endl;
        }
        running = false;
        return;
    }

    serial_port.flush();
    
    std::vector<uint8_t> buffer(BUFFER_SIZE, 0); // 预分配且初始化为0
    
    // 直接进入主处理循环
    {
        boost::lock_guard<boost::mutex> lock(cout_mutex);
        std::cout << "开始读取串口数据" << std::endl;
    }
    
    while (running) {
        if (serial_port.available()) {
            try {
                
                size_t available_bytes = static_cast<size_t>(serial_port.available());
                size_t bytes_to_read = std::min(available_bytes, BUFFER_SIZE);
                
                size_t bytes_read = serial_port.read(buffer.data(), bytes_to_read);
                
                if (bytes_read >= sizeof(helios::MCUPacket)) {
                    auto now = std::chrono::high_resolution_clock::now();
                    // int64_t timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    //     now.time_since_epoch()).count();
                    // int64_t unix_timestamp_ms = timestamp_ns / 1000000; // 转换为毫秒
                    
                    helios::MCUPacket packet;
                    memcpy(&packet, buffer.data(), sizeof(helios::MCUPacket));
                    
                    // 快速路径：如果队列为空，直接添加
                    if (gyro_queue.empty() && !new_gyro_available.load(std::memory_order_relaxed)) {
                        boost::lock_guard<boost::mutex> lock(gyro_queue_mutex);
                        gyro_queue.push_back(TimestampedPacket(packet, now));   
                        new_gyro_available.store(true, std::memory_order_release);
                    } else {
                        boost::lock_guard<boost::mutex> lock(gyro_queue_mutex);
                        gyro_queue.push_back(TimestampedPacket(packet, now));
                        new_gyro_available.store(true, std::memory_order_release);
                    }
                }
            } catch (const std::exception& e) {
                boost::lock_guard<boost::mutex> lock(cout_mutex);
                std::cerr << "Error reading from serial port: " << e.what() << std::endl;
            }
        } 
    }

    serial_port.close();
    
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
    int baud_rate = 92160000;                 
    
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