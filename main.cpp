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

// 性能优化的关键参数
constexpr int MAX_QUEUE_SIZE = 10;         // 减少队列大小以降低内存占用
constexpr int TIME_MATCH_THRESHOLD_MS = 8; // 时间戳匹配阈值
constexpr size_t BUFFER_SIZE = 64;         // 串口缓冲区大小

// 核心状态控制
std::atomic<bool> running(true);
std::atomic<bool> camera_initialized(false);

// 使用单独的线程锁，减少全局锁争用
std::mutex cout_mutex;
std::mutex frame_deque_mutex;
std::mutex gyro_deque_mutex;

// 无锁数据处理的原子标志
std::atomic<bool> new_frame_available(false);
std::atomic<bool> new_gyro_available(false);

// 数据结构优化：使用内存对齐且尽量避免虚拟函数以减少缓存未命中
// 使用struct而非class，避免不必要的封装开销
struct alignas(64) TimestampedPacket {    // 64字节对齐以匹配缓存行大小
    helios::MCUPacket packet;
    int64_t timestamp;
    
    TimestampedPacket() : timestamp(0) {}
    TimestampedPacket(const helios::MCUPacket& p, int64_t ts) : packet(p), timestamp(ts) {}
};

struct alignas(64) TimestampedFrame {     // 64字节对齐以匹配缓存行大小
    cv::Mat frame;
    int64_t timestamp;
    
    TimestampedFrame() : timestamp(0) {}
    TimestampedFrame(const cv::Mat& f, int64_t ts) : frame(f.clone()), timestamp(ts) {}
};

// 使用表示匹配数据对的结构体
struct alignas(64) MatchedData {          // 64字节对齐以匹配缓存行大小
    cv::Mat frame;
    helios::MCUPacket gyro_data;
    int64_t timestamp;
    
    MatchedData(const cv::Mat& f, const helios::MCUPacket& g, int64_t ts) 
        : frame(f.clone()), gyro_data(g), timestamp(ts) {}
};

// 优化数据存储：预分配内存的环形缓冲区
// 使用双端队列提高插入和删除效率
std::deque<TimestampedFrame> frame_deque;
std::deque<TimestampedPacket> gyro_deque;

// 数据处理函数：用于处理匹配的数据对
void process_matched_pair(const cv::Mat& frame, const helios::MCUPacket& gyro_data, int64_t timestamp) {
    // 使用参数避免警告
    (void)timestamp;
    
    // 在这里添加你的处理逻辑
    // 使用 frame 和 gyro_data 进行处理


    // 显示匹配的图像作为示例
    cv::imshow("Matched Frame", frame);
    cv::waitKey(1);
    
    // 可以在这里添加进一步的处理...
}


void camera_thread_func() {
    // 设置线程优先级
    pthread_t this_thread = pthread_self();
    struct sched_param params;
    params.sched_priority = sched_get_priority_max(SCHED_FIFO);
    pthread_setschedparam(this_thread, SCHED_FIFO, &params);
    
    // 初始化相机
    camera::HikCamera hikCam;
    auto time_start = std::chrono::steady_clock::now();
    std::string camera_config_path = HIK_CONFIG_FILE_PATH"/camera_config.yaml";
    std::string intrinsic_para_path = HIK_CALI_FILE_PATH"/caliResults/calibCameraData.yml";
    
    try {
        hikCam.Init(true, camera_config_path, intrinsic_para_path, time_start);
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cerr << "相机初始化失败: " << e.what() << std::endl;
        running = false;
        return;
    }
    
    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "海康相机初始化成功" << std::endl;
    }
    
    // 清空缓存的图像
    cv::Mat temp_frame;
    uint64_t temp_device_ts;
    int64_t temp_host_ts;
    while (hikCam.ReadImg(temp_frame, &temp_device_ts, &temp_host_ts)) {
        // 持续读取直到没有新图像
    }
    
    camera_initialized.store(true);
    
 

    
    // 主处理循环
    cv::Mat frame;
    uint64_t device_ts;
    int64_t host_ts;
    
    while (running) {
        bool hasNew = hikCam.ReadImg(frame, &device_ts, &host_ts);
        
        if (hasNew && !frame.empty()) {
            // 快速路径：如果队列为空，直接添加而不需要锁
            if (frame_deque.empty() && !new_frame_available.load(std::memory_order_relaxed)) {
                std::lock_guard<std::mutex> lock(frame_deque_mutex);
                frame_deque.emplace_back(frame, host_ts);
                new_frame_available.store(true, std::memory_order_release);
            } else {
                std::lock_guard<std::mutex> lock(frame_deque_mutex);
                // 限制队列大小
                if (frame_deque.size() >= static_cast<size_t>(MAX_QUEUE_SIZE)) {
                    frame_deque.pop_front();  // 移除最老的帧
                }
                frame_deque.emplace_back(frame, host_ts);
                new_frame_available.store(true, std::memory_order_release);
            }
            
            // 显示原始图像以进行调试
            cv::imshow("Raw Camera", frame);
            int key = cv::waitKey(1);
            if (key == 27) { // ESC键退出
                running = false;
                break;
            }
        }
    }
    
    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "相机线程已结束" << std::endl;
    }
}

// 串口线程函数 - 优化版
void serial_thread_func(const std::string& port_name, int baud_rate) {
    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "Opening serial port: " << port_name << " at " << baud_rate << " baud" << std::endl;
    }
    
    // 等待相机初始化完成
    while (running && !camera_initialized.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    if (!running) {
        return;
    }
    
    serial::Serial serial_port;
    try {
        serial_port.setPort(port_name);
        serial_port.setBaudrate(baud_rate);
        serial::Timeout timeout = serial::Timeout::simpleTimeout(100); 
        serial_port.setTimeout(timeout);
        serial_port.open();
        
        if (!serial_port.isOpen()) {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cerr << "Failed to open serial port." << std::endl;
            running = false;
            return;
        }
    } catch (const std::exception& e) {
        {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cerr << "Failed to open serial port: " << e.what() << std::endl;
        }
        running = false;
        return;
    }
    
    // 清空缓冲区
    serial_port.flush();
    
    // deque没有reserve方法，删除此行
    // gyro_deque.reserve(MAX_QUEUE_SIZE);
    
    std::vector<uint8_t> buffer(BUFFER_SIZE, 0); // 预分配且初始化为0
    
    // 直接进入主处理循环
    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "开始读取串口数据" << std::endl;
    }
    
    while (running) {
        // 非阻塞检查串口数据
        if (serial_port.available()) {
            try {
                // 修复std::min类型不匹配问题
                size_t available_bytes = static_cast<size_t>(serial_port.available());
                size_t bytes_to_read = std::min(available_bytes, BUFFER_SIZE);
                
                // 读取可用数据
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
                        if (gyro_deque.size() >= static_cast<size_t>(MAX_QUEUE_SIZE)) {
                            gyro_deque.pop_front();  // 移除最老的数据
                        }
                        gyro_deque.emplace_back(packet, unix_timestamp_ms);
                        new_gyro_available.store(true, std::memory_order_release);
                    }
                }
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(cout_mutex);
                std::cerr << "Error reading from serial port: " << e.what() << std::endl;
            }
        } else {
            // 短暂休眠，避免CPU空转，使用极短的休眠时间
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }
    
    // 清理资源
    serial_port.close();
    
    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "串口线程已结束" << std::endl;
    }
}

// 数据匹配线程 - 优化版
void data_matching_thread_func() {
    // 等待相机初始化完成
    while (running && !camera_initialized.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    if (!running) {
        return;
    }
    
    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "开始数据匹配线程" << std::endl;
    }
    
    while (running) {
        // 使用无锁检查是否有新数据
        bool frame_available = new_frame_available.load(std::memory_order_acquire);
        bool gyro_available = new_gyro_available.load(std::memory_order_acquire);
        
        // 如果没有新数据，短暂休眠并继续
        if (!frame_available || !gyro_available) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            continue;
        }
        
        // 创建局部数据副本以减少锁持有时间
        std::vector<TimestampedFrame> local_frames;
        std::vector<TimestampedPacket> local_gyros;
        
        // 获取相机帧
        {
            std::lock_guard<std::mutex> lock(frame_deque_mutex);
            if (!frame_deque.empty()) {
                local_frames.assign(frame_deque.begin(), frame_deque.end());
            }
        }
        
        // 获取陀螺仪数据
        {
            std::lock_guard<std::mutex> lock(gyro_deque_mutex);
            if (!gyro_deque.empty()) {
                local_gyros.assign(gyro_deque.begin(), gyro_deque.end());
            }
        }
        
        // 如果没有足够的数据，继续等待
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
            
            int64_t time_diff = std::abs(current_frame.timestamp - current_gyro.timestamp);
            
            if (time_diff <= TIME_MATCH_THRESHOLD_MS) {
                // 找到匹配
                matched_frame = current_frame;
                matched_gyro = current_gyro;
                found_match = true;
                break;
            } else if (current_frame.timestamp < current_gyro.timestamp) {
                // 帧时间戳较早，移动到下一帧
                frame_idx++;
            } else {
                // 陀螺仪时间戳较早，移动到下一个陀螺仪数据
                gyro_idx++;
            }
        }
        
        if (found_match) {
            std::cout << "匹配成功: Frame TS: " << matched_frame.timestamp 
                      << ", Gyro TS: " << matched_gyro.timestamp << std::endl;
            // 处理匹配的数据对
            process_matched_pair(matched_frame.frame, matched_gyro.packet, matched_gyro.timestamp);
            
            // 移除已匹配及更早的数据
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
            // 清理过时的数据
            if (!local_frames.empty() && !local_gyros.empty()) {
                const auto& oldest_frame = local_frames.front();
                const auto& oldest_gyro = local_gyros.front();
                
                if (oldest_frame.timestamp + TIME_MATCH_THRESHOLD_MS < oldest_gyro.timestamp) {
                    // 最老的帧已经太老，无法与任何陀螺仪数据匹配
                    std::lock_guard<std::mutex> lock(frame_deque_mutex);
                    if (!frame_deque.empty() && frame_deque.front().timestamp == oldest_frame.timestamp) {
                        frame_deque.pop_front();
                    }
                } else if (oldest_gyro.timestamp + TIME_MATCH_THRESHOLD_MS < oldest_frame.timestamp) {
                    // 最老的陀螺仪数据已经太老，无法与任何帧匹配
                    std::lock_guard<std::mutex> lock(gyro_deque_mutex);
                    if (!gyro_deque.empty() && gyro_deque.front().timestamp == oldest_gyro.timestamp) {
                        gyro_deque.pop_front();
                    }
                }
            }
        }
    }
    
    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "数据匹配线程已结束" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::string port_name = "/dev/ttyACM0";  
    int baud_rate = 92160000;                 
    
    if (argc > 1) {
        port_name = argv[1];
    }
    
    if (argc > 2) {
        baud_rate = std::stoi(argv[2]);
    }
    
    // 创建必要的线程
    std::thread cam_thread(camera_thread_func);
    std::thread serial_thread(serial_thread_func, port_name, baud_rate);
    std::thread matching_thread(data_matching_thread_func);
    

    // 这对于多核系统特别有用
    if (cam_thread.native_handle()) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(0, &cpuset);  // 分配给CPU 0
        pthread_setaffinity_np(cam_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    }
    
    if (serial_thread.native_handle()) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(1, &cpuset);  // 分配给CPU 1
        pthread_setaffinity_np(serial_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    }
    
    if (matching_thread.native_handle()) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(2, &cpuset);  // 分配给CPU 2
        pthread_setaffinity_np(matching_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    }
    
    // 等待线程结束
    cam_thread.join();
    serial_thread.join();
    matching_thread.join();
    
    std::cout << "程序已正常退出" << std::endl;
    return 0;
}