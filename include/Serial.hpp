#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <queue>
#include <memory>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <future>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <vector>
#include <cstring> 
#include <termios.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <errno.h>


#include "CRC.h"

// Define the packed attribute for cross-platform compatibility
#define __packed __attribute__((__packed__))

namespace helios
{

typedef struct __packed
{
  uint8_t sof = 0xA5;
  uint16_t data_length;
  uint8_t seq;
  uint8_t crc8;
  uint16_t cmd_id;
} FrameHeader;

// Base class for messages
struct MsgBase
{
  virtual ~MsgBase() = default;
};

// Template implementation for typed messages
template <typename T>
struct MsgImpl : MsgBase
{
  MsgImpl<T>(T data) : data(data) {}
  T data;
};

// Base class for custom publishers
struct PublisherBase
{
  virtual void publish(std::shared_ptr<MsgBase> msg) = 0;
  virtual ~PublisherBase() = default;
};

// Template implementation for typed publishers
template <typename MsgType>
struct PublisherImpl : PublisherBase
{
  using CallbackType = std::function<void(const MsgType&)>;
  CallbackType callback;

  PublisherImpl(CallbackType cb) : callback(cb) {}

  void publish(std::shared_ptr<MsgBase> msg) override
  {
    auto msg_impl = std::dynamic_pointer_cast<MsgImpl<MsgType>>(msg);
    if (msg_impl == nullptr)
    {
      // Log error: "Failed to cast message"
      return;
    }
    callback(msg_impl->data);
  }
};

// Serial port handler class
class Serial
{
public:
  // Constructor and destructor
  Serial();
  Serial(const std::string& serial_name, int baud_rate);
  ~Serial();

  // Port operations
  bool open(const std::string& serial_name, int baud_rate);
  void close();
  bool is_open() const;

  // Thread control
  void start();
  void stop();

  // Message handling
  void register_callback(uint16_t cmd_id, std::function<void(std::vector<uint8_t>)> callback);
  
  // Publisher registration
  void register_publisher(uint16_t cmd_id, std::shared_ptr<PublisherBase> publisher);
  
  // Template method to publish message
  template <typename MessageT>
  void publish(std::shared_ptr<MessageT> msg, uint16_t cmd_id)
  {
    auto it = publisher_map_.find(cmd_id);
    if (it != publisher_map_.end())
    {
      auto msg_impl = std::make_shared<MsgImpl<MessageT>>(*msg);
      it->second->publish(std::dynamic_pointer_cast<MsgBase>(msg_impl));
    }
    else
    {
      // Log error: "No publisher found for cmd_id"
      log_error("No publisher found for cmd_id: " + std::to_string(cmd_id));
    }
  }

  template <typename MessageT>
  void write(MessageT& msg, uint16_t cmd_id)
  {
    FrameHeader header;
    header.data_length = sizeof(msg);
    header.cmd_id = cmd_id;
    header.seq = seq_counter_++;  // 使用序列计数器
    
    // 添加 CRC8 校验和到帧头
    Append_CRC8_Check_Sum(reinterpret_cast<uint8_t*>(&header), sizeof(header));
    
    // 转换为字节向量
    std::vector<uint8_t> head_buffer(sizeof(header));
    std::memcpy(head_buffer.data(), &header, sizeof(header));
    
    std::vector<uint8_t> data_buffer(sizeof(msg));
    std::memcpy(data_buffer.data(), &msg, sizeof(msg));
    
    // 合并帧头和数据
    std::vector<uint8_t> complete_packet(head_buffer.size() + data_buffer.size());
    std::copy(head_buffer.begin(), head_buffer.end(), complete_packet.begin());
    std::copy(data_buffer.begin(), data_buffer.end(), complete_packet.begin() + head_buffer.size());
    
    // 添加 CRC16 校验和到整个数据包
    Append_CRC16_Check_Sum(complete_packet.data(), complete_packet.size());
    
    // 将数据放入发送队列
    {
      std::lock_guard<std::mutex> lock(write_mutex_);
      write_queue_.push(complete_packet);
    }
    write_cv_.notify_one();
  }
private:

  int serial_fd_;

  // Threading control
  std::atomic<bool> running_{false};
  std::thread receive_thread_;
  std::thread write_thread_;
  std::mutex write_mutex_;
  std::condition_variable write_cv_;
  
  // Sequence counter for messages
  std::atomic<uint8_t> seq_counter_{0};

  // Message queue and callbacks
  std::queue<std::vector<uint8_t>> write_queue_;
  std::unordered_map<uint16_t, std::shared_ptr<PublisherBase>> publisher_map_;
  std::unordered_map<uint16_t, std::function<void(std::vector<uint8_t>)>> callback_map_;

  // Internal methods
  void receive_loop();
  void write_loop();
  bool reopen_port();
  
  // Low-level I/O methods
  bool send_data(const std::vector<uint8_t>& data);
  bool receive_data(std::vector<uint8_t>& data, size_t size);
  
  // Connection parameters
  std::string port_name_;
  int baud_rate_;
  
  // Error handling
  void log_error(const std::string& message);
  void log_warning(const std::string& message);
  void log_info(const std::string& message);
};

}  // namespace helios