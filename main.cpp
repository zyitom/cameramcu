#include "Serial.hpp"
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
#include "MVCameraInput.h"

std::atomic<bool> running(true);
std::mutex cout_mutex; 
std::mutex frame_rate_mutex; 


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


FrameRateStats cameraFPS;
FrameRateStats serialRxFPS;
FrameRateStats serialTxFPS;

// 相机线程函数 - 设置高优先级
void camera_thread_func() {

    pthread_t this_thread = pthread_self();
    struct sched_param params;
    params.sched_priority = sched_get_priority_max(SCHED_FIFO);
    pthread_setschedparam(this_thread, SCHED_FIFO, &params);

    
 
    MVCameraInput cam;
    if (!cam.isOpened()) {
        {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cerr << "无法打开相机！\n";
        }
        running = false;
        return;
    }

    try {
        cam.loadConfig("../camera_config.yaml");
    } catch (const std::exception& e) {
        {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "无法加载配置文件，使用默认配置\n";
        }
    }

    cv::Mat frame;
    auto fpsTimer = std::chrono::steady_clock::now();
    
    while (running) {
    
        cam >> frame;
        if (!frame.empty()) {
          
            {
                std::lock_guard<std::mutex> lock(frame_rate_mutex);
                cameraFPS.update();
            }
            
            
            cv::putText(frame, 
                       "Camera FPS: " + std::to_string(static_cast<int>(cameraFPS.currentFPS)), 
                       cv::Point(10, 30), 
                       cv::FONT_HERSHEY_SIMPLEX, 
                       1.0, 
                       cv::Scalar(0, 255, 0), 
                       2);
            
            cv::imshow("frame", frame);
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


void fps_display_thread_func() {
    while(running) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
        float camera_fps = 0.0f;
        float serial_rx_fps = 0.0f;
        float serial_tx_fps = 0.0f;
        
        {
            std::lock_guard<std::mutex> lock(frame_rate_mutex);
            camera_fps = cameraFPS.currentFPS;
            serial_rx_fps = serialRxFPS.currentFPS;
            serial_tx_fps = serialTxFPS.currentFPS;
        }
        
        {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "----------------------------------------" << std::endl;
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "相机帧率: " << camera_fps << " FPS" << std::endl;
            std::cout << "串口接收帧率: " << serial_rx_fps << " FPS" << std::endl;
            std::cout << "串口发送帧率: " << serial_tx_fps << " FPS" << std::endl;
            std::cout << "----------------------------------------" << std::endl;
        }
    }
}

void serial_thread_func(const std::string& port_name, int baud_rate) {
    helios::Serial serial;
    
    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "Opening serial port: " << port_name << " at " << baud_rate << " baud" << std::endl;
    }
    
    if (!serial.open(port_name, baud_rate)) {
        {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cerr << "Failed to open serial port" << std::endl;
        }
        running = false;
        return;
    }
    
  
    std::queue<helios::MCUPacket> packet_queue;
    std::mutex queue_mutex;
    
   
    serial.register_callback(helios::RECEIVE_AUTOAIM_RECEIVE_CMD_ID, [&packet_queue, &queue_mutex](std::vector<uint8_t> data) {
        
        {
            std::lock_guard<std::mutex> lock(frame_rate_mutex);
            serialRxFPS.update();
        }
        
        if (data.size() >= sizeof(helios::MCUPacket)) {
            helios::MCUPacket packet = fromVector<helios::MCUPacket>(data);
            
            // 将数据包放入队列，而不是直接处理
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                packet_queue.push(packet);
            }
        }
    });
    
    // 启动串口
    serial.start();
    
    // 处理队列中的数据包的线程
    std::thread packet_processing_thread([&packet_queue, &queue_mutex]() {
        while(running) {
            helios::MCUPacket packet;
            bool has_packet = false;
            
            // 检查队列中是否有数据包
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                if (!packet_queue.empty()) {
                    packet = packet_queue.front();
                    packet_queue.pop();
                    has_packet = true;
                }
            }
            
            // 处理数据包
            // if (has_packet) {
            //     std::lock_guard<std::mutex> lock(cout_mutex);
            //     std::cout << "处理MCU数据包:" << std::endl;
            //     std::cout << "  Color: " << (packet.self_color == 0 ? "Blue" : "Red") << std::endl;
            //     std::cout << "  Mode: ";
            //     switch (packet.autoaim_mode) {
            //         case 0: std::cout << "Auto-aim"; break;
            //         case 1: std::cout << "Small Energy"; break;
            //         case 2: std::cout << "Large Energy"; break;
            //         default: std::cout << "Unknown"; break;
            //     }
            //     std::cout << std::endl;
                
            //     std::cout << "  Bullet speed: " << packet.bullet_speed << " m/s" << std::endl;
            //     std::cout << "  IMU angles (yaw,pitch,roll): " 
            //             << packet.yaw << ", " 
            //             << packet.pitch << ", " 
            //             << packet.roll << std::endl;
            //     std::cout << "  Target position (x,y,z): " 
            //             << packet.x << ", " 
            //             << packet.y << ", " 
            //             << packet.z << std::endl;
            // }
            
            // 如果没有数据包处理，短暂休眠以减少CPU使用
            if (!has_packet) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    });
    

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
            
            
            serial.write(target, helios::SEND_TARGET_INFO_CMD_ID);
            
         
            {
                std::lock_guard<std::mutex> lock(frame_rate_mutex);
                serialTxFPS.update();
            }
            
            counter++;
            
          
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        catch (const std::exception& ex) {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cerr << "Error: " << ex.what() << std::endl;
        }
    }
    

    packet_processing_thread.join();
    
    // 清理关闭
    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "Stopping serial communication..." << std::endl;
    }
    
    serial.stop();
    serial.close();
    
    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "串口线程已结束" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    // 串口参数
    std::string port_name = "/dev/ttyACM0";  
    int baud_rate = 921600;                 
    
    if (argc > 1) {
        port_name = argv[1];
    }
    
    if (argc > 2) {
        baud_rate = std::stoi(argv[2]);
    }
    
    // 创建线程
    std::thread cam_thread(camera_thread_func);
    std::thread serial_thread(serial_thread_func, port_name, baud_rate);
    std::thread fps_thread(fps_display_thread_func);
    

    cam_thread.join();
    serial_thread.join();
    fps_thread.join();
    
    std::cout << "程序已正常退出" << std::endl;
    return 0;
}