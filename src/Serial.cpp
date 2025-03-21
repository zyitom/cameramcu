#include "Serial.hpp"
#include "CRC.h"
#include <iostream>
#include <chrono>
#include <stdexcept>
#include <algorithm>
#include <cstring>

namespace helios
{


speed_t get_baud_rate(int baud)
{
  switch (baud) {
    case 9600: return B9600;
    case 19200: return B19200;
    case 38400: return B38400;
    case 57600: return B57600;
    case 115200: return B115200;
    case 230400: return B230400;
    case 460800: return B460800;
    case 500000: return B500000;
    case 576000: return B576000;
    case 921600: return B921600;
    case 1000000: return B1000000;
    case 1152000: return B1152000;
    case 1500000: return B1500000;
    case 2000000: return B2000000;
    case 2500000: return B2500000;
    case 3000000: return B3000000;
    case 3500000: return B3500000;
    case 4000000: return B4000000;
    default: return B115200;
  }
}

Serial::Serial() 
  : port_name_(""), baud_rate_(115200) 
{

  serial_fd_ = -1;

}

Serial::Serial(const std::string& serial_name, int baud_rate) 
  : port_name_(serial_name), baud_rate_(baud_rate) 
{
  open(serial_name, baud_rate);
}

Serial::~Serial() 
{
  stop();
  close();
}

void Serial::log_error(const std::string& message) 
{
  std::cerr << "[ERROR] " << message << std::endl;
}

void Serial::log_warning(const std::string& message) 
{
  std::cerr << "[WARNING] " << message << std::endl;
}

void Serial::log_info(const std::string& message) 
{
  std::cout << "[INFO] " << message << std::endl;
}

bool Serial::open(const std::string& serial_name, int baud_rate) 
{
  port_name_ = serial_name;
  baud_rate_ = baud_rate;
  
  serial_fd_ = ::open(serial_name.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
  
  if (serial_fd_ < 0) {
    log_error("Failed to open serial port: " + port_name_);
    return false;
  }
  
  // Configure serial port parameters
  struct termios options;
  memset(&options, 0, sizeof(options));
  
  if (tcgetattr(serial_fd_, &options) != 0) {
    log_error("Failed to get terminal attributes");
    ::close(serial_fd_);
    serial_fd_ = -1;
    return false;
  }
  
  // Set baud rate
  speed_t baudrate = get_baud_rate(baud_rate);
  cfsetispeed(&options, baudrate);
  cfsetospeed(&options, baudrate);
  
  // Set character size and disable parity
  options.c_cflag |= (CLOCAL | CREAD);
  options.c_cflag &= ~PARENB;
  options.c_cflag &= ~CSTOPB;
  options.c_cflag &= ~CSIZE;
  options.c_cflag |= CS8;
  options.c_cflag &= ~CRTSCTS;
  
  // Set raw input mode
  options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
  options.c_iflag &= ~(IXON | IXOFF | IXANY);
  options.c_iflag &= ~(INLCR | ICRNL);
  options.c_oflag &= ~OPOST;
  
  // Set timeouts
  options.c_cc[VMIN] = 0;     // No minimum number of characters
  options.c_cc[VTIME] = 1;    // Timeout in deciseconds

  // Apply settings
  if (tcsetattr(serial_fd_, TCSANOW, &options) != 0) {
    log_error("Failed to set terminal attributes");
    ::close(serial_fd_);
    serial_fd_ = -1;
    return false;
  }
  
  // Clear any lingering data
  tcflush(serial_fd_, TCIOFLUSH);

  log_info("Serial port opened successfully: " + port_name_);
  return true;
}

void Serial::close() 
{
  if (serial_fd_ >= 0) {
    ::close(serial_fd_);
    serial_fd_ = -1;
  }

}

bool Serial::is_open() const 
{
  return serial_fd_ >= 0;
}

bool Serial::send_data(const std::vector<uint8_t>& data) 
{
  if (!is_open()) {
    log_error("Cannot send data: serial port not open");
    return false;
  }
  

  ssize_t bytes_written = ::write(serial_fd_, data.data(), data.size());
  if (bytes_written < 0) {
    log_error("Failed to write to serial port: " + std::string(strerror(errno)));
    return false;
  }
  return static_cast<size_t>(bytes_written) == data.size();

}

bool Serial::receive_data(std::vector<uint8_t>& data, size_t size) 
{
  if (!is_open()) {
    log_error("Cannot receive data: serial port not open");
    return false;
  }
  
  if (size == 0 || data.size() < size) {
    data.resize(size);
  }

  size_t total_bytes_read = 0;
  auto start_time = std::chrono::steady_clock::now();
  const auto timeout = std::chrono::milliseconds(500); // Timeout after 500ms
  
  while (total_bytes_read < size) {
    ssize_t bytes_read = ::read(serial_fd_, data.data() + total_bytes_read, size - total_bytes_read);
    
    if (bytes_read > 0) {
      total_bytes_read += bytes_read;
    } else if (bytes_read < 0 && errno != EAGAIN) {
      log_error("Failed to read from serial port: " + std::string(strerror(errno)));
      return false;
    }
    
    // Check for timeout
    auto current_time = std::chrono::steady_clock::now();
    if (current_time - start_time > timeout) {
      log_warning("Timeout while reading from serial port");
      return false;
    }
    
    // Small delay to prevent tight loop
    if (bytes_read <= 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  }
  
  return true;

}

void Serial::start() 
{
  if (running_) {
    return;
  }
  
  if (!is_open()) {
    log_error("Cannot start threads: serial port not open");
    return;
  }
  
  running_ = true;
  receive_thread_ = std::thread(&Serial::receive_loop, this);
  write_thread_ = std::thread(&Serial::write_loop, this);
  
  log_info("Serial communication threads started");
}

void Serial::stop() 
{
  if (!running_) {
    return;
  }
  
  running_ = false;
  write_cv_.notify_all();
  
  if (receive_thread_.joinable()) {
    receive_thread_.join();
  }
  
  if (write_thread_.joinable()) {
    write_thread_.join();
  }
  
  log_info("Serial communication threads stopped");
}

void Serial::register_callback(uint16_t cmd_id, std::function<void(std::vector<uint8_t>)> callback) 
{
  callback_map_[cmd_id] = callback;
}

void Serial::register_publisher(uint16_t cmd_id, std::shared_ptr<PublisherBase> publisher) 
{
  publisher_map_[cmd_id] = publisher;
}

bool Serial::reopen_port() 
{
  log_warning("Attempting to reopen port: " + port_name_);
  
  close();
  
  // Wait a bit before trying to reopen
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  
  for (int attempt = 0; attempt < 5; ++attempt) {
    if (open(port_name_, baud_rate_)) {
      log_info("Successfully reopened port: " + port_name_);
      return true;
    }
    
    log_warning("Failed to reopen port, retrying...");
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
  
  log_error("Failed to reopen port after multiple attempts");
  return false;
}

void Serial::receive_loop() 
{
  log_info("Starting receive loop...");
  
  std::vector<uint8_t> sof_buffer(1);
  std::vector<uint8_t> header_buffer(6);
  std::vector<uint8_t> header_with_sof(7, 0xA5);
  
  while (running_) {
    try {
      // Wait for SOF (Start of Frame) byte
      if (!receive_data(sof_buffer, 1) || sof_buffer[0] != 0xA5) {
        // Not SOF or read error, continue to next iteration
        continue;
      }
      
      // Read header after SOF
      if (!receive_data(header_buffer, 6)) {
        continue;
      }
      
      // Copy header data to a structured header (with SOF)
      std::copy(header_buffer.begin(), header_buffer.end(), header_with_sof.begin() + 1);
      FrameHeader frame_header;
      std::copy(header_with_sof.begin(), header_with_sof.end(), reinterpret_cast<uint8_t*>(&frame_header));
      
      // Sanity check on data length
      if (frame_header.data_length > 1024) {
        log_warning("Received abnormal data length: " + std::to_string(frame_header.data_length));
        continue;
      }
      
      // Verify CRC8
      if (!Verify_CRC8_Check_Sum(reinterpret_cast<uint8_t*>(&frame_header), sizeof(frame_header) - 2)) {
        log_warning("Invalid CRC8 checksum");
        continue;
      }
      
      // Read the payload plus CRC16 (2 bytes)
      std::vector<uint8_t> payload(frame_header.data_length + 2);
      if (!receive_data(payload, payload.size())) {
        log_warning("Failed to read payload");
        continue;
      }
      
      // Combine header and payload for CRC16 verification
      std::vector<uint8_t> complete_packet(header_with_sof.size() + payload.size());
      std::copy(header_with_sof.begin(), header_with_sof.end(), complete_packet.begin());
      std::copy(payload.begin(), payload.end(), complete_packet.begin() + header_with_sof.size());
      
      // Verify CRC16
      if (!Verify_CRC16_Check_Sum(complete_packet.data(), complete_packet.size())) {
        log_warning("Invalid CRC16 checksum");
        continue;
      }
      
      // Process message - pass payload without CRC16
      std::vector<uint8_t> message_data(payload.begin(), payload.end() - 2);
      auto callback_it = callback_map_.find(frame_header.cmd_id);
      if (callback_it != callback_map_.end()) {
        callback_it->second(message_data);
      } else {
        log_warning("No callback registered for cmd_id: " + std::to_string(frame_header.cmd_id));
      }
      
    } catch (const std::exception& ex) {
      log_error("Exception in receive loop: " + std::string(ex.what()));
      
      if (!reopen_port()) {
        // If we can't reopen, sleep to prevent tight loop and CPU overuse
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
    }
  }
  
  log_info("Receive loop ended");
}

void Serial::write_loop() 
{
  log_info("Starting write loop...");
  
  while (running_) {
    std::vector<uint8_t> data_to_send;
    
    // Wait for data to send
    {
      std::unique_lock<std::mutex> lock(write_mutex_);
      write_cv_.wait_for(lock, std::chrono::milliseconds(100), [this] { 
        return !write_queue_.empty() || !running_; 
      });
      
      if (!running_) {
        break;
      }
      
      if (write_queue_.empty()) {
        continue;
      }
      
      data_to_send = write_queue_.front();
      write_queue_.pop();
    }
    
    try {
      if (!send_data(data_to_send)) {
        log_warning("Failed to send data");
        
        // Put data back in the queue for retry
        std::lock_guard<std::mutex> lock(write_mutex_);
        write_queue_.push(data_to_send);
        
        if (!reopen_port()) {
          // If we can't reopen, sleep to prevent tight loop and CPU overuse
          std::this_thread::sleep_for(std::chrono::seconds(1));
        }
      }
    } catch (const std::exception& ex) {
      log_error("Exception in write loop: " + std::string(ex.what()));
      
      // Put data back in the queue for retry
      std::lock_guard<std::mutex> lock(write_mutex_);
      write_queue_.push(data_to_send);
      
      if (!reopen_port()) {
        // If we can't reopen, sleep to prevent tight loop and CPU overuse
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
    }
  }
  
  log_info("Write loop ended");
}

}  // namespace helios