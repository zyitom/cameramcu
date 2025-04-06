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
#include <queue>
#include <iomanip>
#include "hikvision_camera.h"

std::atomic<bool> running(true);
std::atomic<bool> camera_initialized(false);
std::atomic<bool> start_processing(false);
std::condition_variable start_cv;
std::mutex start_mutex;

// 添加时间点记录
std::chrono::time_point<std::chrono::steady_clock> sync_start_time;

std::mutex cout_mutex; 
std::mutex frame_rate_mutex; 
std::atomic<int64_t> last_received_timestamp(0);

struct FrameRateStats {
    std::chrono::time_point<std::chrono::steady_clock> lastUpdateTime;
    int frameCount;
    float currentFPS;
    
    FrameRateStats() : frameCount(0), currentFPS(0.0f) {
        lastUpdateTime = std::chrono::steady_clock::now();
    }
    
    void update() {
        frameCount++;
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastUpdateTime).count();
        
        if (elapsed > 1000) {
            currentFPS = frameCount * 1000.0f / elapsed;
            frameCount = 0;
            lastUpdateTime = now;
        }
    }
};

// 创建结构体用于存储带时间戳的数据包
struct TimestampedPacket {
    helios::MCUPacket packet;
    int64_t timestamp;
    TimestampedPacket() : packet(), timestamp(0) {}

    TimestampedPacket(const helios::MCUPacket& p, int64_t ts) : packet(p), timestamp(ts) {}
};

FrameRateStats cameraFPS;
FrameRateStats serialRxFPS;
FrameRateStats serialTxFPS;

void synchronization_thread_func(int delay_seconds) {
    while (running && !camera_initialized.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    if (!running) {
        return;
    }
    
    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "相机已初始化完成，等待 " << delay_seconds << " 秒后开始处理..." << std::endl;
    }
    
    std::this_thread::sleep_for(std::chrono::seconds(delay_seconds));
    
    // 记录同步启动的时间点
    sync_start_time = std::chrono::steady_clock::now();
    
    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "等待完成，现在开始同步处理数据！" << std::endl;
    }
    
    {
        std::lock_guard<std::mutex> lock(start_mutex);
        start_processing.store(true);
    }
    start_cv.notify_all();
}

void camera_thread_func() {
    pthread_t this_thread = pthread_self();
    struct sched_param params;
    params.sched_priority = sched_get_priority_max(SCHED_FIFO);
    pthread_setschedparam(this_thread, SCHED_FIFO, &params);
        
    camera::HikCamera hikCam;
    auto time_start = std::chrono::steady_clock::now();
    std::string camera_config_path = HIK_CONFIG_FILE_PATH"/camera_config.yaml";
    std::string intrinsic_para_path = HIK_CALI_FILE_PATH"/caliResults/calibCameraData.yml";
    
    hikCam.Init(true, camera_config_path, intrinsic_para_path, time_start);
    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "海康相机初始化成功\n";
        hikCam.CamInfoShow();
    }
    
    // 清空可能已经缓存的图像
    cv::Mat temp_frame;
    uint64_t temp_device_ts;
    int64_t temp_host_ts;
    while (hikCam.ReadImg(temp_frame, &temp_device_ts, &temp_host_ts)) {
        // 持续读取直到没有新图像
    }
    
    camera_initialized.store(true);
    
    // 等待同步启动信号
    {
        std::unique_lock<std::mutex> lock(start_mutex);
        start_cv.wait(lock, []{ return start_processing.load(); });
    }
    
    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "开始处理相机图像数据" << std::endl;
    }

    cv::Mat frame;
    uint64_t device_ts;
    int64_t host_ts;
    
    while (running) {
        bool hasNew = hikCam.ReadImg(frame, &device_ts, &host_ts);
        
        if (hasNew && !frame.empty()) {
                std::cout << "Device Timestamp: " << host_ts << " ms" << std::endl;
                auto now = std::chrono::system_clock::now();
                int64_t camera_timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now.time_since_epoch()).count();
                std::cout << "Camera Timestamp: " << camera_timestamp_ms << " ms" << std::endl;
            {
                std::lock_guard<std::mutex> lock(frame_rate_mutex);
                cameraFPS.update();
            }
            // printf("相机timestamp: %ld\n", host_ts); 
           
    
            
            cv::imshow("frame", frame);
            int key = cv::waitKey(1);
            if (key == 27) { // ESC键退出
                running = false;
                break;
            }
        } 
        // 即使没有新图像也不休眠，以保持尽可能高的速度
    }
    
    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "相机线程已结束" << std::endl;
    }
}

void fps_display_thread_func() {
    // 等待同步启动信号
    {
        std::unique_lock<std::mutex> lock(start_mutex);
        start_cv.wait(lock, []{ return start_processing.load(); });
    }
    
    while(running) {

        // std::this_thread::sleep_for(std::chrono::seconds(1));
        
        // float camera_fps = 0.0f;
        // float serial_rx_fps = 0.0f;
        // float serial_tx_fps = 0.0f;
        
        // {
        //     std::lock_guard<std::mutex> lock(frame_rate_mutex);
        //     camera_fps = cameraFPS.currentFPS;
        //     serial_rx_fps = serialRxFPS.currentFPS;
        //     serial_tx_fps = serialTxFPS.currentFPS;
        // }
        
        // {
        //     std::lock_guard<std::mutex> lock(cout_mutex);
        //     std::cout << "\n----------------------------------------" << std::endl;
        //     std::cout << std::fixed << std::setprecision(2);
        //     std::cout << "相机帧率: " << camera_fps << " FPS" << std::endl;
        //     std::cout << "串口接收帧率: " << serial_rx_fps << " FPS" << std::endl;
        //     std::cout << "串口发送帧率: " << serial_tx_fps << " FPS" << std::endl;
        //     std::cout << "----------------------------------------" << std::endl;
        // }
    }
}


void serial_thread_func(const std::string& port_name, int baud_rate) {
    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "Opening serial port: " << port_name << " at " << baud_rate << " baud" << std::endl;
    }
    
    // 给串口足够的时间初始化
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    serial::Serial serial_port;
    try {
        serial_port.setPort(port_name);
        serial_port.setBaudrate(baud_rate);
        serial::Timeout timeout = serial::Timeout::simpleTimeout(1000);
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
    
    // 清空可能存在的缓冲数据
    serial_port.flush();
    
    std::queue<TimestampedPacket> packet_queue;
    std::mutex queue_mutex;
    

    std::thread serial_read_thread([&]() {
        const size_t buffer_size = 40;
        std::vector<uint8_t> buffer(buffer_size);
        
     
        {
            std::unique_lock<std::mutex> lock(start_mutex);
            start_cv.wait(lock, []{ return start_processing.load(); });
        }
        
    
        
        while (running) {
            if (serial_port.available()) {
                try {
                    // 读取可用数据
                    size_t bytes_read = serial_port.read(buffer, buffer_size);
                    
                    if (bytes_read > 0) {
                        // 获取当前时间戳
                        auto now = std::chrono::system_clock::now();
                        int64_t unix_timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            now.time_since_epoch()).count();
                        std::cout << "串口接收时间戳: " << unix_timestamp_ms << " ms" << std::endl;
                        // 更新帧率统计
                        {
                            std::lock_guard<std::mutex> lock(frame_rate_mutex);
                            serialRxFPS.update();
                        }
                 
                        if (bytes_read >= sizeof(helios::MCUPacket)) {
                            helios::MCUPacket packet;
                            memcpy(&packet, buffer.data(), sizeof(helios::MCUPacket));
                            
                            // 将数据包放入队列
                            {
                                std::lock_guard<std::mutex> lock(queue_mutex);
                                packet_queue.push(TimestampedPacket(packet, unix_timestamp_ms));
                            }
                        }
                    }
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(cout_mutex);
                    std::cerr << "Error reading from serial port: " << e.what() << std::endl;
                }
            } 
        }
    });
    

    std::thread packet_processing_thread([&]() {
   
        {
            std::unique_lock<std::mutex> lock(start_mutex);
            start_cv.wait(lock, []{ return start_processing.load(); });
        }
        
        while(running) {
            TimestampedPacket packet_data;
            bool has_packet = false;
            
            // 检查队列中是否有数据包
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                if (!packet_queue.empty()) {
                    packet_data = packet_queue.front();
                    packet_queue.pop();
                    has_packet = true;
                }
            }
            
            // 处理数据包
            if (has_packet) {
             
                int64_t previous_timestamp = last_received_timestamp.exchange(packet_data.timestamp);
                if (previous_timestamp > 0) {
                    int64_t interval = packet_data.timestamp - previous_timestamp;
                    
                    std::lock_guard<std::mutex> lock(cout_mutex);
                    // std::cout << "串口接收时间间隔: " << interval << " ms" << std::endl;
                }
            } 
        }
    });
    
    // 等待同步启动信号
    {
        std::unique_lock<std::mutex> lock(start_mutex);
        start_cv.wait(lock, []{ return start_processing.load(); });
    }
    
    // 发送数据的循环
    int counter = 0;
    const int MAX_PACKETS = 1000; 
    
    while (running && counter < MAX_PACKETS) {
        try {
            helios::TargetInfo target;
            target.gimbal_id = 0;
            target.tracking = 1;  // tracking
            target.id = 7;        // guard
            target.armors_num = 4; // normal
            
            target.x = 1.5f + 0.1f * (counter % 20);  // 循环变化以避免数值过大
            target.y = 0.5f + 0.05f * (counter % 20);
            target.z = 0.2f;
            
            target.vx = 0.2f;
            target.vy = 0.1f;
            target.vz = 0.0f;
            
            target.yaw = 30.0f * M_PI / 180.0f;  // 30度（弧度）
            target.v_yaw = 0.1f;
            target.r1 = 0.4f;
            target.r2 = 0.4f;
            target.dz = 0.15f;
            target.vision_delay = 15.0f;  // 15ms处理延迟
            
            // 将结构体转换为字节数组并发送
            std::vector<uint8_t> target_data(sizeof(helios::TargetInfo));
            memcpy(target_data.data(), &target, sizeof(helios::TargetInfo));
            
            // 添加命令ID到数据包头部（根据您的协议调整）
            // 假设您的协议是简单地将命令ID作为前几个字节
            const uint16_t cmd_id = helios::SEND_TARGET_INFO_CMD_ID;
            std::vector<uint8_t> cmd_bytes(sizeof(cmd_id));
            memcpy(cmd_bytes.data(), &cmd_id, sizeof(cmd_id));
            
            // 合并命令ID和数据
            std::vector<uint8_t> full_packet;
            full_packet.insert(full_packet.end(), cmd_bytes.begin(), cmd_bytes.end());
            full_packet.insert(full_packet.end(), target_data.begin(), target_data.end());
            
            // 发送完整的数据包
            size_t bytes_written = serial_port.write(full_packet);
            
            if (bytes_written != full_packet.size()) {
                std::lock_guard<std::mutex> lock(cout_mutex);
                std::cerr << "Failed to write all bytes to serial port." << std::endl;
            } else {
                {
                    std::lock_guard<std::mutex> lock(frame_rate_mutex);
                    serialTxFPS.update();
                }
            }
            
            counter++;
            
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        catch (const std::exception& ex) {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cerr << "Error: " << ex.what() << std::endl;
        }
    }
    
    // 等待线程结束
    serial_read_thread.join();
    packet_processing_thread.join();
    
    // 清理关闭
    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "Stopping serial communication..." << std::endl;
    }
    
    serial_port.close();
    
    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "串口线程已结束" << std::endl;
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
    
    int delay_seconds = 3;
    if (argc > 3) {
        delay_seconds = std::stoi(argv[3]);
    }
    
    // 添加一个额外的清除参数 - 如果设置为true，将尝试在启动前清空串口缓冲区
    bool clear_buffer = true;
    if (argc > 4) {
        clear_buffer = std::stoi(argv[4]) != 0;
    }
    
    std::thread sync_thread(synchronization_thread_func, delay_seconds);
    std::thread cam_thread(camera_thread_func);
    std::thread serial_thread(serial_thread_func, port_name, baud_rate);
    std::thread fps_thread(fps_display_thread_func);
    
    sync_thread.join();
    cam_thread.join();
    serial_thread.join();
    fps_thread.join();
    
    std::cout << "程序已正常退出" << std::endl;
    return 0;
}