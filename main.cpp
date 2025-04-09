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
#include <iomanip>  // 用于格式化输出
#include "hikvision_camera.h"
#include "ovdetector.hpp"
// 性能优化的关键参数
constexpr int MAX_QUEUE_SIZE = 10;         // 减少队列大小以降低内存占用
constexpr float TIME_MATCH_MIN_MS = 7;       // 时间戳匹配最小值 (ms)
constexpr float TIME_MATCH_MAX_MS = 9;  
constexpr size_t BUFFER_SIZE = 32;         // 串口缓冲区大小，提高为2的幂次方以优化内存对齐

// 核心状态控制
std::atomic<bool> running(true);
std::atomic<bool> camera_initialized(false);

// 使用单独的线程锁，减少全局锁争用
std::mutex cout_mutex;
std::mutex frame_deque_mutex;
std::mutex gyro_deque_mutex;
std::mutex stats_mutex;  // 用于保护统计数据

// 无锁数据处理的原子标志
std::atomic<bool> new_frame_available(false);
std::atomic<bool> new_gyro_available(false);

// 添加用于跟踪匹配帧率的结构体
struct MatchStats {
    int matched_count = 0;
    int total_frames = 0;
    int total_gyro = 0;
    std::chrono::time_point<std::chrono::steady_clock> last_report_time;
    std::vector<int> time_diffs;  // 存储时间戳差异统计

    MatchStats() {
        last_report_time = std::chrono::steady_clock::now();
    }
    
    void add_match(int64_t time_diff) {
        matched_count++;
        time_diffs.push_back(std::abs(static_cast<int>(time_diff)));
    }
    
    void add_frame() {
        total_frames++;
    }
    
    void add_gyro() {
        total_gyro++;
    }
    
    // 打印统计信息并重置计数器
    void report_and_reset() {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_report_time).count();
        
        if (elapsed >= 1000) {  // 每秒打印一次
            float match_rate = (matched_count * 1000.0f) / elapsed;
            float frame_rate = (total_frames * 1000.0f) / elapsed;
            float gyro_rate = (total_gyro * 1000.0f) / elapsed;
            
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "\n--- 匹配统计信息 ---" << std::endl;
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "相机帧率: " << frame_rate << " FPS" << std::endl;
            std::cout << "陀螺仪数据率: " << gyro_rate << " Hz" << std::endl;
            std::cout << "匹配帧率: " << match_rate << " FPS" << std::endl;
            std::cout << "匹配率: " << (matched_count * 100.0f / (total_frames > 0 ? total_frames : 1)) << "%" << std::endl;
            
            // 计算时间差异统计
            if (!time_diffs.empty()) {
                int min_diff = *std::min_element(time_diffs.begin(), time_diffs.end());
                int max_diff = *std::max_element(time_diffs.begin(), time_diffs.end());
                float avg_diff = 0;
                for (int diff : time_diffs) {
                    avg_diff += diff;
                }
                avg_diff /= time_diffs.size();
                
                std::cout << "时间戳差异: 最小=" << min_diff << "ms, 平均=" << avg_diff 
                          << "ms, 最大=" << max_diff << "ms" << std::endl;
                std::cout << "--------------------" << std::endl;
            }
            
            // 重置计数器和时间
            matched_count = 0;
            total_frames = 0;
            total_gyro = 0;
            time_diffs.clear();
            last_report_time = now;
        }
    }
};

MatchStats match_stats;  // 全局统计对象

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
int last_timestamp;

std::unique_ptr<YOLOXDetector> global_detector;
std::mutex detector_mutex;

// Initialize the detector during startup
bool initialize_detector(const std::string& model_path, const std::string& device_name = "GPU") {
    try {
        std::lock_guard<std::mutex> lock(detector_mutex);
        global_detector = std::make_unique<YOLOXDetector>(model_path, device_name);
        return global_detector->initialize();
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cerr << "Failed to initialize detector: " << e.what() << std::endl;
        return false;
    }
}
// 数据处理函数：用于处理匹配的数据对
void process_matched_pair(const cv::Mat& frame, const helios::MCUPacket& gyro_data, int64_t timestamp) {
    // Create a local copy to avoid modifying the original frame
    cv::Mat processed_frame = frame.clone();
    
    // Use the detector to find objects in the frame
    std::vector<Object> detected_objects;
    
    {
        // Use the detector with proper locking to ensure thread safety
        std::lock_guard<std::mutex> lock(detector_mutex);
        if (global_detector) {
            // Get frame dimensions
            int original_width = frame.cols;
            int original_height = frame.rows;
            
            // Get detector input size and calculate scaling factors
            auto [model_width, model_height] = global_detector->get_input_size();
            float scale_x = static_cast<float>(model_width) / original_width;
            float scale_y = static_cast<float>(model_height) / original_height;
            
            // Set scaling for proper detection
            global_detector->set_scale(scale_x, scale_y);
            
            // Perform detection
            detected_objects = global_detector->detect(frame);
        }
    }
    
    // Draw the detected objects on the frame
    if (!detected_objects.empty()) {
        // processed_frame = visualize_detection(frame, detected_objects);
        
        // You can also use gyro data here if needed
        // For example, you could use gyro data to compensate for camera movement
        
        // Log gyro data for debugging or analysis
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "Gyro data - Pitch: " << gyro_data.pitch 
                  << ", Yaw: " << gyro_data.yaw 
                  << ", Roll: " << gyro_data.roll 
                  << " at timestamp: " << timestamp << std::endl;
    }
    
    // Display the processed frame
    cv::imshow("Detection Results", processed_frame);
    cv::waitKey(1);
}
// 相机线程函数 - 优化版
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
            // 更新统计
            {
                std::lock_guard<std::mutex> lock(stats_mutex);
                match_stats.add_frame();
            }
            
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
        serial::Timeout timeout = serial::Timeout::simpleTimeout(100); // 减少超时时间以提高响应性
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
                    // 获取高精度时间戳
                    auto now = std::chrono::high_resolution_clock::now();
                    int64_t timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        now.time_since_epoch()).count();
                    int64_t unix_timestamp_ms = timestamp_ns / 1000000; // 转换为毫秒
                    
                    helios::MCUPacket packet;
                    memcpy(&packet, buffer.data(), sizeof(helios::MCUPacket));
                    
                    // 更新统计
                    {
                        std::lock_guard<std::mutex> lock(stats_mutex);
                        match_stats.add_gyro();
                    }
                    
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

// 统计线程 - 用于定期打印统计信息
void stats_thread_func() {
    // 等待相机初始化完成
    while (running && !camera_initialized.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    if (!running) {
        return;
    }
    
    while (running) {
        {
            std::lock_guard<std::mutex> lock(stats_mutex);
            match_stats.report_and_reset();
        }
        
        // 每100毫秒检查一次，避免频繁锁争用
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
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
        
        // 实现高效的匹配算法，寻找符合7-8ms时间差的匹配
        
        size_t frame_idx = 0;
        size_t gyro_idx = 0;
        bool found_match = false;
        TimestampedFrame matched_frame;
        TimestampedPacket matched_gyro;
        
        // 假设陀螺仪数据比帧时间戳早7-8ms，即 frame_ts - gyro_ts 应在 7-8ms 范围内
        // 即：帧时间戳应该比陀螺仪时间戳大7-8ms
        while (frame_idx < local_frames.size() && gyro_idx < local_gyros.size()) {
            const auto& current_frame = local_frames[frame_idx];
            const auto& current_gyro = local_gyros[gyro_idx];
            
            // 计算时间差：帧时间戳 - 陀螺仪时间戳
            int64_t time_diff = current_frame.timestamp - current_gyro.timestamp;
            
            // 检查时间差是否在 7-8ms 范围内
            if (time_diff >= TIME_MATCH_MIN_MS && time_diff <= TIME_MATCH_MAX_MS) {
                // 找到符合要求的匹配
                matched_frame = current_frame;
                matched_gyro = current_gyro;
                found_match = true;
                
                // 记录匹配信息用于调试
                // {
                //     std::lock_guard<std::mutex> lock(cout_mutex);
                //     std::cout << "找到匹配: 帧时间戳=" << matched_frame.timestamp 
                //               << ", 陀螺仪时间戳=" << matched_gyro.timestamp 
                //               << ", 时间差=" << time_diff << "ms" << std::endl;
                // }
                
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
            // 更新匹配统计
            {
                std::lock_guard<std::mutex> lock(stats_mutex);
                match_stats.add_match(matched_frame.timestamp - matched_gyro.timestamp);
            }
            // std::cout << "匹配成功: 帧时间戳=" << matched_frame.timestamp 
            //           << ", 陀螺仪时间戳=" << matched_gyro.timestamp 
            //           << ", 时间差=" << (matched_frame.timestamp - matched_gyro.timestamp) << "ms" << std::endl;
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
                
                // 如果最老的帧时间戳比最老的陀螺仪数据时间戳早超过10ms，则无法匹配
                if (oldest_frame.timestamp + 10 < oldest_gyro.timestamp) {
                    // 最老的帧已经太老，无法与任何陀螺仪数据匹配
                    std::lock_guard<std::mutex> lock(frame_deque_mutex);
                    if (!frame_deque.empty() && frame_deque.front().timestamp == oldest_frame.timestamp) {
                        frame_deque.pop_front();
                        
                        // 输出调试信息
                        std::lock_guard<std::mutex> cout_lock(cout_mutex);
                        std::cout << "丢弃过时帧: 时间戳=" << oldest_frame.timestamp << std::endl;
                    }
                }
                // 如果最老的陀螺仪数据比最老的帧时间戳早超过10ms，则无法匹配
                else if (oldest_gyro.timestamp + 10 < oldest_frame.timestamp - TIME_MATCH_MAX_MS) {
                    // 最老的陀螺仪数据已经太老，无法与任何帧匹配
                    std::lock_guard<std::mutex> lock(gyro_deque_mutex);
                    if (!gyro_deque.empty() && gyro_deque.front().timestamp == oldest_gyro.timestamp) {
                        gyro_deque.pop_front();
                        
                        // 输出调试信息
                        std::lock_guard<std::mutex> cout_lock(cout_mutex);
                        std::cout << "丢弃过时陀螺仪数据: 时间戳=" << oldest_gyro.timestamp << std::endl;
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
    
    // Add paths for your detector model
    std::string model_path = "/home/zyi/Downloads/0405_9744.onnx"; // Replace with your actual model path
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
    
    // Initialize the detector before starting threads
    if (!initialize_detector(model_path, device_name)) {
        std::cerr << "Failed to initialize the detector. Exiting." << std::endl;
        return EXIT_FAILURE;
    }
    // 创建必要的线程
    std::thread cam_thread(camera_thread_func);
    std::thread serial_thread(serial_thread_func, port_name, baud_rate);
    std::thread matching_thread(data_matching_thread_func);
    std::thread stats_thread(stats_thread_func);  // 添加统计线程
    
    // 设置线程亲和性以提高性能（如果系统支持）
    // 这对于多核系统特别有用
    // 设置线程亲和性以提高性能
    if (cam_thread.native_handle()) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(0, &cpuset);
        CPU_SET(1, &cpuset); // 添加第二个核心
        CPU_SET(2, &cpuset); // 添加第三个核心
        pthread_setaffinity_np(cam_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    }

    if (serial_thread.native_handle()) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(3, &cpuset); // 串口线程单独分配一个核心
        pthread_setaffinity_np(serial_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    }

    if (matching_thread.native_handle()) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        // 允许匹配线程也使用多个核心
        CPU_SET(0, &cpuset);
        CPU_SET(1, &cpuset);
        pthread_setaffinity_np(matching_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    }

    if (stats_thread.native_handle()) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(3, &cpuset); // 统计线程可以和串口线程共享核心
        pthread_setaffinity_np(stats_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
}
    
    // 等待线程结束
    cam_thread.join();
    serial_thread.join();
    matching_thread.join();
    stats_thread.join();
    {
        std::lock_guard<std::mutex> lock(detector_mutex);
        global_detector.reset();
    }
    std::cout << "程序已正常退出" << std::endl;
    return 0;
}