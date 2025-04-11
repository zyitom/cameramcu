#include "Serial.hpp"
#include <serial/serial.h>
#include "Protocol.hpp"
#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <memory>
#include <atomic>
#include <mutex>
#include <deque>
#include <cmath>
#include "hikvision_camera.h"
#include "ovdetector.hpp"
#include <optional>


constexpr int MAX_QUEUE_SIZE = 5;         // 减少队列大小以降低内存占用
constexpr float TIME_MATCH_MIN_MS = 7;    // 时间戳匹配最小值 (ms)
constexpr float TIME_MATCH_MAX_MS = 9;    // 时间戳匹配最大值 (ms)
constexpr size_t BUFFER_SIZE = 64;        // 串口缓冲区大小，提高为2的幂次方以优化内存对齐与缓存命中


std::atomic<bool> running(true);
std::atomic<bool> camera_initialized(false);


std::mutex cout_mutex;
std::mutex frame_deque_mutex;
std::mutex gyro_deque_mutex;
std::mutex detector_mutex;


std::atomic<bool> new_frame_available(false);
std::atomic<bool> new_gyro_available(false);


struct alignas(64) TimestampedPacket {    
    helios::MCUPacket packet;
    int64_t timestamp;
    
    TimestampedPacket() : timestamp(0) {}
    TimestampedPacket(const helios::MCUPacket& p, int64_t ts) : packet(p), timestamp(ts) {}
};

struct alignas(64) TimestampedFrame {    
    cv::Mat frame;
    int64_t timestamp;
    
    TimestampedFrame() : timestamp(0) {}
    TimestampedFrame(const cv::Mat& f, int64_t ts) : frame(f), timestamp(ts) {}
};

std::deque<TimestampedFrame> frame_deque;
std::deque<TimestampedPacket> gyro_deque;


constexpr int MAX_PENDING_RESULTS = 3; 
std::mutex pending_results_mutex;
std::deque<std::tuple<int64_t, helios::MCUPacket, std::future<FrameData>>> pending_results;

std::unique_ptr<YOLOXDetector> global_detector;

bool initialize_detector(const std::string& model_path, const std::string& device_name = "GPU") {
    try {
        std::lock_guard<std::mutex> lock(detector_mutex);
        global_detector = std::make_unique<YOLOXDetector>(model_path, device_name);
        return global_detector->initialize();
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize detector: " << e.what() << std::endl;
        return false;
    }
}

void process_matched_pair(const cv::Mat& frame, const helios::MCUPacket& gyro_data, int64_t timestamp) {
    auto input_timestamp = std::chrono::high_resolution_clock::now();
    FrameData input_data(input_timestamp, frame);

  
    std::future<FrameData> future;
    {
        std::lock_guard<std::mutex> lock(detector_mutex);
        if (global_detector) {
            future = global_detector->submit_frame_async(input_data);
        } else {
            std::cerr << "Detector not initialized." << std::endl;
            return;
        }
    }
    
   
    {
        std::lock_guard<std::mutex> lock(pending_results_mutex);
        pending_results.push_back({timestamp, gyro_data, std::move(future)});
        
 
        while (pending_results.size() > MAX_PENDING_RESULTS) {
            pending_results.pop_front();
        }
    }
}

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
        std::cerr << "相机初始化失败: " << e.what() << std::endl;
        running = false;
        return;
    }
    
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
            if (frame_deque.empty() && !new_frame_available.load(std::memory_order_relaxed)) {
                std::lock_guard<std::mutex> lock(frame_deque_mutex);
                frame_deque.emplace_back(frame, host_ts);
                new_frame_available.store(true, std::memory_order_release);
            } else {
                std::lock_guard<std::mutex> lock(frame_deque_mutex);
                // 限制队列大小
                while (frame_deque.size() >= static_cast<size_t>(MAX_QUEUE_SIZE)) {
                    frame_deque.pop_front();  // 移除最老的帧
                }
                frame_deque.emplace_back(frame, host_ts);
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
}


void serial_thread_func(const std::string& port_name, int baud_rate) {

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
            std::cerr << "Failed to open serial port." << std::endl;
            running = false;
            return;
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to open serial port: " << e.what() << std::endl;
        running = false;
        return;
    }

    serial_port.flush();
    
    std::vector<uint8_t> buffer(BUFFER_SIZE, 0);
    
    while (running) {
        if (serial_port.available()) {
            try {
                
                size_t available_bytes = static_cast<size_t>(serial_port.available());
                size_t bytes_to_read = std::min(available_bytes, BUFFER_SIZE);
                
                size_t bytes_read = serial_port.read(buffer.data(), bytes_to_read);
                
                if (bytes_read >= sizeof(helios::MCUPacket)) {
                    auto now = std::chrono::high_resolution_clock::now();
                    int64_t timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        now.time_since_epoch()).count();
                    int64_t unix_timestamp_ms = timestamp_ns / 1000000; // 转换为毫秒
                    
                    helios::MCUPacket packet;
                    memcpy(&packet, buffer.data(), sizeof(helios::MCUPacket));
                    
                    // 快速路径：如果队列为空，直接添加
                    if (gyro_deque.empty() && !new_gyro_available.load(std::memory_order_relaxed)) {
                        std::lock_guard<std::mutex> lock(gyro_deque_mutex);
                        gyro_deque.emplace_back(packet, unix_timestamp_ms);
                        new_gyro_available.store(true, std::memory_order_release);
                    } else {
                        std::lock_guard<std::mutex> lock(gyro_deque_mutex);
                        // 限制队列大小
                        while (gyro_deque.size() >= static_cast<size_t>(MAX_QUEUE_SIZE)) {
                            gyro_deque.pop_front();  // 移除最老的数据
                        }
                        gyro_deque.emplace_back(packet, unix_timestamp_ms);
                        new_gyro_available.store(true, std::memory_order_release);
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "Error reading from serial port: " << e.what() << std::endl;
            }
        } 
    }

    serial_port.close();
}

void data_matching_thread_func() {
    while (running && !camera_initialized.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    
    if (!running) {
        return;
    }
    
    while (running) {
        bool frame_available = new_frame_available.load(std::memory_order_acquire);
        bool gyro_available = new_gyro_available.load(std::memory_order_acquire);
        
        if (!frame_available || !gyro_available) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            continue;
        }
        
        std::vector<TimestampedFrame> local_frames;
        std::vector<TimestampedPacket> local_gyros;
        
        {
            std::lock_guard<std::mutex> lock(frame_deque_mutex);
            if (!frame_deque.empty()) {
                local_frames.assign(frame_deque.begin(), frame_deque.end());
            }
        }
        
        {
            std::lock_guard<std::mutex> lock(gyro_deque_mutex);
            if (!gyro_deque.empty()) {
                local_gyros.assign(gyro_deque.begin(), gyro_deque.end());
            }
        }
        
        if (local_frames.empty() || local_gyros.empty()) {
            continue;
        }

        size_t frame_idx = 0;
        size_t gyro_idx = 0;
        bool found_match = false;
        TimestampedFrame matched_frame;
        TimestampedPacket matched_gyro;
        
        while (frame_idx < local_frames.size() && gyro_idx < local_gyros.size()) {
            const auto& current_frame = local_frames[frame_idx];
            const auto& current_gyro = local_gyros[gyro_idx];
            
            // 计算时间差：帧时间戳 - 陀螺仪时间戳
            int64_t time_diff = current_frame.timestamp - current_gyro.timestamp;
            
            // 检查时间差是否在指定范围内
            if (time_diff >= TIME_MATCH_MIN_MS && time_diff <= TIME_MATCH_MAX_MS) {
                // 找到符合要求的匹配
                matched_frame = current_frame;
                matched_gyro = current_gyro;
                found_match = true;
                break; // 找到第一个匹配就退出
            } 
            // 移动指针以继续寻找可能的匹配
            else if (time_diff < TIME_MATCH_MIN_MS) {
                frame_idx++;
            } else {
                // 陀螺仪时间戳较早，移动到下一个陀螺仪数据
                gyro_idx++;
            }
        }
        
        if (found_match) {

            process_matched_pair(matched_frame.frame, matched_gyro.packet, matched_gyro.timestamp);

            {
                std::lock_guard<std::mutex> lock(frame_deque_mutex);
                auto it = frame_deque.begin();
                while (it != frame_deque.end() && it->timestamp <= matched_frame.timestamp) {
                    it = frame_deque.erase(it);
                }
                
                if (frame_deque.empty()) {
                    new_frame_available.store(false, std::memory_order_release);
                }
            }
            
            {
                std::lock_guard<std::mutex> lock(gyro_deque_mutex);
                auto it = gyro_deque.begin();
                while (it != gyro_deque.end() && it->timestamp <= matched_gyro.timestamp) {
                    it = gyro_deque.erase(it);
                }
                
                if (gyro_deque.empty()) {
                    new_gyro_available.store(false, std::memory_order_release);
                }
            }
        } else {
         
            if (!local_frames.empty() && !local_gyros.empty()) {
                const auto& oldest_frame = local_frames.front();
                const auto& oldest_gyro = local_gyros.front();
                
                // 如果最老的帧时间戳比最老的陀螺仪数据时间戳早超过10ms，则无法匹配
                if (oldest_frame.timestamp + 10 < oldest_gyro.timestamp) {
                    // 最老的帧已经太老，无法与任何陀螺仪数据匹配
                    std::lock_guard<std::mutex> lock(frame_deque_mutex);
                    if (!frame_deque.empty() && frame_deque.front().timestamp == oldest_frame.timestamp) {
                        frame_deque.pop_front();
                    }
                }
                // 如果最老的陀螺仪数据比最老的帧时间戳早超过10ms，则无法匹配
                else if (oldest_gyro.timestamp + 10 < oldest_frame.timestamp - TIME_MATCH_MAX_MS) {
                    // 最老的陀螺仪数据已经太老，无法与任何帧匹配
                    std::lock_guard<std::mutex> lock(gyro_deque_mutex);
                    if (!gyro_deque.empty() && gyro_deque.front().timestamp == oldest_gyro.timestamp) {
                        gyro_deque.pop_front();
                    }
                }
            }
        }
    }
}

void result_processing_thread_func() {
    // 设置线程优先级
    pthread_t this_thread = pthread_self();
    struct sched_param params;
    params.sched_priority = sched_get_priority_max(SCHED_FIFO); // 比相机线程一样
    pthread_setschedparam(this_thread, SCHED_FIFO, &params);
    
    // 等待相机初始化完成
    while (running && !camera_initialized.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    
    if (!running) {
        return;
    }
    
    while (running) {
        std::vector<std::tuple<int64_t, helios::MCUPacket, std::future<FrameData>>> completed_futures;
        
        // 获取需要处理的结果
        {
            std::lock_guard<std::mutex> lock(pending_results_mutex);
            
            // 检查已完成的future，不等待
            auto it = pending_results.begin();
            while (it != pending_results.end()) {
                auto& [ts, gyro, future] = *it;
                
                if (future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                    completed_futures.push_back(std::move(*it));
                    it = pending_results.erase(it);
                } else {
                    ++it;
                }
            }
        }
        
        // 处理已完成的结果
        for (auto& [timestamp, gyro_data, future] : completed_futures) {
            try {
                FrameData result = future.get();
                
                if (result.processed && !result.objects.empty()) {
                    cv::Mat processed_frame = visualize_detection(result.frame, result.objects);
                    cv::imshow("Detection Results", processed_frame);
                    cv::waitKey(1);
                }
            } catch (const std::exception& e) {
                std::cerr << "处理检测结果异常: " << e.what() << std::endl;
            }
        }
        
        // 如果没有任务要处理，短暂休眠
        if (completed_futures.empty()) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
}

int main(int argc, char* argv[]) {
    std::string port_name = "/dev/ttyACM0";  
    int baud_rate = 92160000;                 
    
    std::string model_path = "/home/zyi/Downloads/409_9704.onnx";
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
    

    std::thread cam_thread(camera_thread_func);
    std::thread serial_thread(serial_thread_func, port_name, baud_rate);
    std::thread matching_thread(data_matching_thread_func);
    std::thread result_thread(result_processing_thread_func); 
    
    // 设置线程亲和性以提高性能
    if (cam_thread.native_handle()) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(0, &cpuset);
        pthread_setaffinity_np(cam_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    }

    if (serial_thread.native_handle()) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(1, &cpuset); // 串口线程单独分配一个核心
        pthread_setaffinity_np(serial_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    }

    if (matching_thread.native_handle()) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        // 允许匹配线程使用单个核心，避免频繁切换
        CPU_SET(2, &cpuset);
        pthread_setaffinity_np(matching_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    }
    
    if (result_thread.native_handle()) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(3, &cpuset); // 结果处理线程使用第4个核心
        pthread_setaffinity_np(result_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    }
    
    cam_thread.join();
    serial_thread.join();
    matching_thread.join();
    result_thread.join();
    
    
    {
        std::lock_guard<std::mutex> lock(detector_mutex);
        global_detector.reset();
    }
    
    std::cout << "程序已正常退出" << std::endl;
    return 0;
}