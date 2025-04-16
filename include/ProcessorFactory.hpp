
// #pragma once

// #include <memory>
// #include <string>
// #include <unordered_map>
// #include <functional>
// #include <chrono>
// #include <opencv2/opencv.hpp>
// #include "Protocol.hpp"
// #include "ovdetector.hpp"
// #include <future>
// #include <boost/thread/mutex.hpp>
// #include <boost/thread/lock_guard.hpp>
// #include <boost/thread/condition_variable.hpp>
// #include <boost/lockfree/queue.hpp>
// #include <boost/circular_buffer.hpp>
// #include <boost/thread/thread.hpp>
// #include <boost/bind.hpp>
// #include <boost/asio.hpp>
// #include <boost/asio/serial_port.hpp>
// #include <boost/bind.hpp>
// #include <boost/thread.hpp>
// class ProcessingStrategy {
//     public:
//         virtual ~ProcessingStrategy() = default;
//         virtual void process(const cv::Mat& frame, 
//                             const helios::MCUPacket& gyro_data, 
//                             std::chrono::high_resolution_clock::time_point timestamp) = 0;
// };
    
//     // Implementation for asynchronous detector-based processing
//     class AsyncDetectorStrategy : public ProcessingStrategy {
//         private:
//             std::mutex detector_mutex_;
//             std::shared_ptr<YOLOXDetector> detector_;
        
//         public:
//             explicit AsyncDetectorStrategy(std::shared_ptr<YOLOXDetector> detector) 
//                 : detector_(std::move(detector)) {}
        
//             void process(const cv::Mat& frame, 
//                         const helios::MCUPacket& gyro_data, 
//                         std::chrono::high_resolution_clock::time_point timestamp) override {
//                 FrameData input_data(timestamp, frame);
                
//                 // Async submission using the existing pattern
//                 std::shared_future<FrameData> shared_future;
//                 {
//                     std::lock_guard<std::mutex> lock(detector_mutex_);
//                     if (detector_) {
//                         std::future<FrameData> future = detector_->submit_frame_async(input_data);
//                         shared_future = future.share();
//                     } else {
//                         std::cerr << "Detector not initialized." << std::endl;
//                         return;
//                     }
//                 }
                
//                 // Add to pending results queue
//                 {
//                     boost::lock_guard<boost::mutex> lock(pending_results_mutex);
//                     pending_results.emplace_back(timestamp, gyro_data, shared_future);
                    
//                     // Limit queue size
//                     if (pending_results.size() > MAX_PENDING_RESULTS) {
//                         pending_results.pop_front();
//                     }
//                 }
//             }
//         };