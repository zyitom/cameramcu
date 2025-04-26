#ifndef ARMOR_DETECTOR_HPP
#define ARMOR_DETECTOR_HPP

#include <algorithm>
#include <condition_variable>
#include <opencv2/core/mat.hpp>
#include <string>
#include <vector>
#include <queue>
#include <deque>
#include <thread>
#include <iostream>
#include <numeric>
#include <mutex>
#include <opencv2/opencv.hpp>
#include "Armor.hpp"
#include "openvino/openvino.hpp"
#include <future>

constexpr int NUM_APEX = 4;      
constexpr int NUM_CLASS = 7;     
constexpr int NUM_COLORS = 2;    
constexpr float CLASSIFIER_THRESHOLD = 0.65f;  
constexpr float NMS_THRESH = 0.45f;  
const std::vector<std::string> ARMOR_NUMBER_LABEL{ "guard", "1", "2", "3", "4", "5", "outpost", "base", "base" };

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

struct Object
{
    std::vector<cv::Point2f> apexes;
    cv::Rect_<float> rect;
    int label;
    int color;
    float conf;
};

class FrameData {
    public:
        std::chrono::high_resolution_clock::time_point timestamp;
        cv::Mat frame;
        std::vector<Armor> armors;
        bool processed;
        uint64_t sync_id;  // 添加同步ID
        
        FrameData() : processed(false), sync_id(UINT64_MAX) {}
        
        FrameData(std::chrono::high_resolution_clock::time_point ts, const cv::Mat& f) 
            : timestamp(ts), frame(f), processed(false), sync_id(UINT64_MAX) {}
        
        // 添加包含同步ID的构造函数
        FrameData(std::chrono::high_resolution_clock::time_point ts, const cv::Mat& f, uint64_t id) 
            : timestamp(ts), frame(f), processed(false), sync_id(id) {}
        
        void add_armor_results(const std::vector<Armor>& detected_armors) {
            armors = detected_armors;
            processed = true;
        }
        
        bool has_valid_results() const {
            return processed && !armors.empty();
        }
    };

class OVnetDetector {
public:
    OVnetDetector(const std::string& model_path, bool is_blue = true);
    ~OVnetDetector();
    
    bool initialize();
    std::future<FrameData> submit_frame_async(const FrameData& input_data);
    void set_scale(float scale_x, float scale_y);
    std::pair<int, int> get_input_size() const;
    cv::Mat visualize_detection_result(const FrameData& frame_data);
    void set_is_blue(bool is_blue);
    FrameData infer_sync(const FrameData& input_data);
private:
    // 单次的推理请求
    struct TimedIreq {
        ov::InferRequest* ireq;      // 指针代替引用
        FrameData frame_data;        // 关联的带时间戳的帧和结果
        bool has_started;            // 是否已启动推理
        
        // 构造函数
        TimedIreq(ov::InferRequest* req, const FrameData& data, bool started)
            : ireq(req), frame_data(data), has_started(started) {}
        
        // 默认构造函数
        TimedIreq() : ireq(nullptr), has_started(false) {}
    };
    
    // 预处理帧
    static ov::Tensor preprocess_frame(const cv::Mat& input_frame, const ov::Shape& input_shape);
    
    // YOLOX处理相关方法
    void init_processor(int input_width, int input_height);
    std::vector<Armor> process_output(const ov::Tensor& output_tensor);
    void generate_yolox_proposal(const float* output_buffer, std::vector<Object>& objects);
    void qsort_descent_inplace(std::vector<Object>& objects);
    void nms_sorted_bboxes(std::vector<Object>& faceobjects, std::vector<int>& picked);
    void avg_rect(std::vector<Object>& objects);
    int argmax(const float* ptr, int len);
    float intersection_area(const Object& a, const Object& b);
    void generate_grids_and_stride(const int w, const int h, const int strides[], std::vector<GridAndStride>& grid_strides);
    ArmorType judge_armor_type(const Armor& armor);
    
    std::string model_path_;
    std::string device_name_;
    bool is_blue_;
    bool initialized_;
    
    ov::Core core_;
    ov::CompiledModel compiled_model_;
    ov::Shape input_shape_;
    int model_width_;
    int model_height_;
    
    std::vector<GridAndStride> grid_strides_;
    float scale_x_;
    float scale_y_;
    
    std::vector<ov::InferRequest> ireqs_;
    
    std::mutex mutex_;
    std::condition_variable cv_;
    std::exception_ptr callback_exception_;
    std::deque<TimedIreq> available_ireqs_;
};

#endif // ARMOR_DETECTOR_HPP