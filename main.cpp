#include "Serial.hpp"
#include "Protocol.hpp"
#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <memory>

int main(int argc, char* argv[]) {

  helios::Serial serial;
  

  std::string port_name = "/dev/ttyACM0";  
  int baud_rate = 921600;                 
  

  if (argc > 1) {
    port_name = argv[1];
  }
  

  if (argc > 2) {
    baud_rate = std::stoi(argv[2]);
  }
  
  std::cout << "Opening serial port: " << port_name << " at " << baud_rate << " baud" << std::endl;
  

  if (!serial.open(port_name, baud_rate)) {
    std::cerr << "Failed to open serial port" << std::endl;
    return 1;
  }
  
  // Register callback for MCUPacket (command ID: RECEIVE_AUTOAIM_RECEIVE_CMD_ID)
  serial.register_callback(helios::RECEIVE_AUTOAIM_RECEIVE_CMD_ID, [](std::vector<uint8_t> data) {
    std::cout << "MCUPacket size: " << sizeof(helios::MCUPacket) << std::endl;
    std::cout << "Received data size: " << data.size() << std::endl;
    if (data.size() >= sizeof(helios::MCUPacket)) {
      helios::MCUPacket packet = fromVector<helios::MCUPacket>(data);
      
      std::cout << "Received MCU packet:" << std::endl;
      std::cout << "  Color: " << (packet.self_color == 0 ? "Blue" : "Red") << std::endl;
      std::cout << "  Mode: ";
      switch (packet.autoaim_mode) {
        case 0: std::cout << "Auto-aim"; break;
        case 1: std::cout << "Small Energy"; break;
        case 2: std::cout << "Large Energy"; break;
        default: std::cout << "Unknown"; break;
      }
      std::cout << std::endl;
      
      std::cout << "  Bullet speed: " << packet.bullet_speed << " m/s" << std::endl;
      std::cout << "  IMU angles (yaw,pitch,roll): " 
                << packet.yaw << ", " 
                << packet.pitch << ", " 
                << packet.roll << std::endl;
      std::cout << "  Target position (x,y,z): " 
                << packet.x << ", " 
                << packet.y << ", " 
                << packet.z << std::endl;
    } else {
      std::cerr << "Received incomplete MCU packet" << std::endl;
    }
  });
  

  serial.start();
  
  // Main loop - simulate sending target info
  int counter = 0;
  while (counter < 10) {
    try {
      // Create target info packet
      helios::TargetInfo target;
      target.gimbal_id = 0;
      target.tracking = 1;  // tracking
      target.id = 7;        // guard
      target.armors_num = 4; // normal
      
      // Set position
      target.x = 1.5f + 0.1f * counter;
      target.y = 0.5f + 0.05f * counter;
      target.z = 0.2f;
      
      // Set velocity
      target.vx = 0.2f;
      target.vy = 0.1f;
      target.vz = 0.0f;
      
      // Set orientation and other parameters
      target.yaw = 30.0f * M_PI / 180.0f;  // 30 degrees in radians
      target.v_yaw = 0.1f;
      target.r1 = 0.4f;
      target.r2 = 0.4f;
      target.dz = 0.15f;
      target.vision_delay = 15.0f;  // 15ms processing delay
      
      // CRC will be calculated by the write method
      
      // Send the target info
      serial.write(target, helios::SEND_TARGET_INFO_CMD_ID);
      
      std::cout << "Sent target info #" << counter << std::endl;
      std::cout << "  Position (x,y,z): " 
                << target.x << ", " 
                << target.y << ", " 
                << target.z << std::endl;
      
      counter++;
      
      // Wait before sending next command
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    catch (const std::exception& ex) {
      std::cerr << "Error: " << ex.what() << std::endl;
    }
  }
  
  // Clean shutdown
  std::cout << "Stopping serial communication..." << std::endl;
  serial.stop();
  serial.close();
  
  std::cout << "Example completed successfully" << std::endl;
  return 0;
}