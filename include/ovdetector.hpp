#include <algorithm>
#include <condition_variable>
#include <string>
#include <vector>
#include <queue>
#include <deque>
#include <thread>
#include <iostream>
#include <numeric>
#include <mutex>
#include <opencv2/opencv.hpp>
#include "openvino/openvino.hpp"


constexpr int NUM_APEX = 4;      
constexpr int NUM_CLASS = 7;     
constexpr int NUM_COLORS = 2;    
constexpr float CLASSIFIER_THRESHOLD = 0.65f;  
constexpr float NMS_THRESH = 0.45f;  


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

ov::Tensor preprocess_frame(const cv::Mat& input_frame, const ov::Shape& input_shape)
{
    cv::Mat resized_frame;
    cv::resize(input_frame, resized_frame, cv::Size(input_shape[2], input_shape[1]), 0, 0, cv::INTER_LINEAR);
    return ov::Tensor(ov::element::u8, input_shape, resized_frame.data);
}

class YOLOXProcessor {
public:
    YOLOXProcessor(float scale_x = 1.0f, float scale_y = 1.0f) 
        : scale_x_(scale_x), scale_y_(scale_y) {}
    
    void init(int input_width, int input_height) {
  
        static const int strides[3] = {8, 16, 32}; 
        grid_strides_.clear();
        generate_grids_and_stride(input_width, input_height, strides, grid_strides_);
    }
    
 
    std::vector<Object> process(const ov::Tensor& output_tensor)
    {
        const float* output_buffer = output_tensor.data<const float>();
        std::vector<Object> objects;
        generate_yolox_proposal(output_buffer, objects);

        qsort_descent_inplace(objects);
        std::vector<int> picked;
        nms_sorted_bboxes(objects, picked);

        std::vector<Object> result;
        result.reserve(picked.size());
        for (int i : picked)
        {
            nms_sorted_bboxes(objects, picked);
    
            result.push_back(objects[i]);
        }

        avg_rect(result);
        return result;
    }

private:
    std::vector<GridAndStride> grid_strides_;
    float scale_x_;
    float scale_y_;
    

    void generate_yolox_proposal(const float* output_buffer, std::vector<Object>& objects)
    {
        const int num_anchors = grid_strides_.size();
        const int class_start = 2 * NUM_APEX + 1;

        for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
        {
            const int basic_pos = anchor_idx * (class_start + NUM_CLASS + NUM_COLORS);
            float box_conf = output_buffer[basic_pos + 2 * NUM_APEX];

            if (box_conf >= CLASSIFIER_THRESHOLD)
            {
                const GridAndStride& gs = grid_strides_[anchor_idx];
                Object obj;
                obj.apexes.reserve(NUM_APEX);
                
                for (int i = 0; i < NUM_APEX; i++)
                {
                    float x = (output_buffer[basic_pos + 0 + i * 2] + gs.grid0) * gs.stride / scale_x_;
                    float y = (output_buffer[basic_pos + 1 + i * 2] + gs.grid1) * gs.stride / scale_y_;
                    obj.apexes.emplace_back(x, y);
                }

                int color_idx = argmax(output_buffer + basic_pos + class_start, NUM_COLORS);
                int class_idx = argmax(output_buffer + basic_pos + class_start + NUM_COLORS, NUM_CLASS);
                float color_conf = output_buffer[basic_pos + class_start + color_idx];
                float class_conf = output_buffer[basic_pos + class_start + NUM_COLORS + class_idx];

                obj.rect = cv::boundingRect(obj.apexes);
                obj.label = class_idx;
                obj.color = color_idx;
                obj.conf = box_conf * ((class_conf + color_conf) / 2);

                objects.push_back(std::move(obj));
            }
        }
    }

    void qsort_descent_inplace(std::vector<Object>& objects)
    {
        std::sort(objects.begin(), objects.end(), 
                 [](const Object& a, const Object& b) { return a.conf > b.conf; });
    }


    void nms_sorted_bboxes(std::vector<Object>& faceobjects, std::vector<int>& picked)
    {
        picked.clear();
        const int n = faceobjects.size();
        picked.reserve(n);

        for (int i = 0; i < n; i++)
        {
            const Object& a = faceobjects[i];

            int keep = 1;
            for (int j = 0; j < (int)picked.size(); j++)
            {
                const Object& b = faceobjects[picked[j]];

                float inter_area = intersection_area(a, b);
                float union_area = a.rect.area() + b.rect.area() - inter_area;
                float iou = inter_area / union_area;

                if (iou > NMS_THRESH || std::isnan(iou))
                {
                    keep = 0;
                    if (iou > 0.9 && std::abs(a.conf - b.conf) < 0.2 && a.label == b.label && a.color == b.color)
                    {
                        faceobjects[picked[j]].apexes.insert(faceobjects[picked[j]].apexes.end(), 
                                                           a.apexes.begin(), a.apexes.end());
                    }
                    break;
                }
            }

            if (keep)
            {
                picked.push_back(i);
            }
        }
    }

    // 平均关键点
    void avg_rect(std::vector<Object>& objects)
    {
        for (auto& object : objects)
        {
            std::size_t N = object.apexes.size();

            if (N >= 2 * NUM_APEX)
            {
                std::vector<cv::Point2f> fin_point(NUM_APEX, cv::Point2f(0, 0));

                for (size_t i = 0; i < N; i++)
                {
                    fin_point[i % NUM_APEX] += object.apexes[i];
                }

                for (int i = 0; i < NUM_APEX; i++)
                {
                    fin_point[i] /= static_cast<float>(N / NUM_APEX);
                }

                object.apexes = std::move(fin_point);
                object.rect = cv::boundingRect(object.apexes);
            }
        }
    }

    // 辅助函数
    int argmax(const float* ptr, int len)
    {
        return std::max_element(ptr, ptr + len) - ptr;
    }

    float intersection_area(const Object& a, const Object& b)
    {
        cv::Rect_<float> inter = a.rect & b.rect;
        return inter.area();
    }

    // 生成网格和步长
    void generate_grids_and_stride(const int w, const int h, const int strides[],
                                 std::vector<GridAndStride>& grid_strides)
    {
        grid_strides.clear();
        grid_strides.reserve(w * h / 64 + w * h / 256 + w * h / 1024);
        
        for (int i = 0; i < 3; i++)
        {
            int num_grid_w = w / strides[i];
            int num_grid_h = h / strides[i];

            for (int g1 = 0; g1 < num_grid_h; g1++)
            {
                for (int g0 = 0; g0 < num_grid_w; g0++)
                {
                    grid_strides.push_back({g0, g1, strides[i]});
                }
            }
        }
    }
};

// 在图像上绘制检测结果
cv::Mat visualize_detection(const cv::Mat& frame, const std::vector<Object>& objects) 
{
    static const std::vector<cv::Scalar> colors = {
        cv::Scalar(255, 0, 0),     // 蓝色
        cv::Scalar(0, 255, 0),     // 绿色
        cv::Scalar(0, 0, 255),     // 红色
        cv::Scalar(255, 255, 0),   // 青色
        cv::Scalar(255, 0, 255),   // 品红
        cv::Scalar(0, 255, 255),   // 黄色
        cv::Scalar(128, 128, 128), // 灰色
        cv::Scalar(255, 255, 255)  // 白色
    };
    
    cv::Mat vis = frame.clone();
    
    for (const auto& obj : objects) {
        // 绘制多边形
        std::vector<cv::Point> pts;
        for (const auto& p : obj.apexes) {
            pts.emplace_back(static_cast<int>(p.x), static_cast<int>(p.y));
        }
        
        // 根据标签选择颜色
        cv::Scalar color = colors[obj.label % colors.size()];
        
        // 绘制多边形
        cv::polylines(vis, pts, true, color, 2);
        
        // 绘制边界框
        cv::rectangle(vis, obj.rect, color, 1);
        
        // 添加标签和置信度
        char text[64];
        sprintf(text, "id:%d %.1f%%", obj.label, obj.conf * 100);
        
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(vis, 
                   cv::Point(obj.rect.x, obj.rect.y - text_size.height - 5),
                   cv::Point(obj.rect.x + text_size.width, obj.rect.y),
                   color, -1);
        cv::putText(vis, text, cv::Point(obj.rect.x, obj.rect.y - 5),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
    
    return vis;
}

class YOLOXDetector {
    public:
        YOLOXDetector(const std::string& model_path, const std::string& device_name = "GPU") 
            : model_path_(model_path), device_name_(device_name), initialized_(false) {}
        
        ~YOLOXDetector() {
            finished_ireqs_.clear();
            ireqs_.clear();
            compiled_model_ = ov::CompiledModel();
            core_ = ov::Core();
        }
        
        bool initialize() {
            try {
                // 加载模型
                core_ = ov::Core();
                std::cout << "加载模型: " << model_path_ << std::endl;
                
                std::shared_ptr<ov::Model> model = core_.read_model(model_path_);
                
                // 配置预处理
                ov::preprocess::PrePostProcessor ppp(model);
                ppp.input()
                    .tensor()
                    .set_element_type(ov::element::u8)
                    .set_layout("NHWC")
                    .set_color_format(ov::preprocess::ColorFormat::BGR);
                ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB);
                ppp.input().model().set_layout("NCHW");
                ppp.output().tensor().set_element_type(ov::element::f32);
                
                model = ppp.build();
                
                ov::AnyMap tput{{ov::hint::performance_mode.name(), ov::hint::PerformanceMode::THROUGHPUT}};
                compiled_model_ = core_.compile_model(model, device_name_, tput);
                
                input_shape_ = compiled_model_.input().get_shape();
                model_height_ = input_shape_[1];  
                model_width_ = input_shape_[2];
                
                // 初始化YOLOX处理器
                yolox_processor_.init(model_width_, model_height_);
                
                uint32_t nireq = compiled_model_.get_property(ov::optimal_number_of_infer_requests);
                std::cout << "创建 " << nireq << " 个推理请求" << std::endl;
                
                ireqs_.resize(nireq);
                std::generate(ireqs_.begin(), ireqs_.end(), [&] {
                    return compiled_model_.create_infer_request();
                });
                
                
                // 初始化完成的推理请求队列
                for (auto& ireq : ireqs_) {
                    finished_ireqs_.push_back({ireq, cv::Mat(), false});
                }
                
                initialized_ = true;
                return true;
            } 
            catch (const std::exception& ex) {
                std::cerr << "初始化异常: " << ex.what() << std::endl;
                return false;
            }
        }
        
        // 接收cv::Mat作为输入，返回检测结果
        std::vector<Object> detect(const cv::Mat& frame) {
            if (!initialized_) {
                std::cerr << "检测器未初始化" << std::endl;
                return {};
            }
            
            std::vector<Object> detected_objects;
            try {
                // 执行推理
                std::unique_lock<std::mutex> lock(mutex_);
                
                // 等待有完成的推理请求
                while (!callback_exception_ && finished_ireqs_.empty()) {
                    cv_.wait(lock);
                }
                
                if (callback_exception_) {
                    std::rethrow_exception(callback_exception_);
                }
                
                if (!finished_ireqs_.empty()) {
                    // 获取一个已完成的推理请求
                    TimedIreq timedIreq = finished_ireqs_.front();
                    finished_ireqs_.pop_front();
                    lock.unlock();
                    
                    ov::InferRequest& ireq = timedIreq.ireq;
                    
                    // 如果已经启动过推理，则这是一个已完成的推理结果
                    if (timedIreq.has_started && !timedIreq.frame.empty()) {
                        // 获取模型输出并执行YOLOX后处理
                        ov::Tensor output_tensor = ireq.get_output_tensor();
                        detected_objects = yolox_processor_.process(output_tensor);
                    }
                    
                    try {
                        // 准备输入张量
                        ov::Tensor input_tensor = preprocess_frame(frame, input_shape_);
                        ireq.set_input_tensor(input_tensor);
                        
                        // 设置回调
                        ireq.set_callback(
                            [&ireq, frame, this](std::exception_ptr ex) {
                                std::unique_lock<std::mutex> lock(this->mutex_);
                                try {
                                    if (ex) {
                                        std::rethrow_exception(ex);
                                    }
                                    this->finished_ireqs_.push_back({ireq, frame, true});
                                } catch (const std::exception&) {
                                    if (!this->callback_exception_) {
                                        this->callback_exception_ = std::current_exception();
                                    }
                                }
                                this->cv_.notify_one();
                            });
                        
                        ireq.start_async();
                    } catch (const std::exception& e) {
                        std::cerr << "推理异常: " << e.what() << std::endl;
                        std::unique_lock<std::mutex> lock(mutex_);
                        finished_ireqs_.push_back({ireq, cv::Mat(), false});
                        cv_.notify_one();
                    }
                }
                
                return detected_objects;
            } 
            catch (const std::exception& ex) {
                std::cerr << "检测异常: " << ex.what() << std::endl;
                return {};
            }
        }
        
        // 设置输入缩放系数
        void set_scale(float scale_x, float scale_y) {
            yolox_processor_ = YOLOXProcessor(scale_x, scale_y);
            yolox_processor_.init(model_width_, model_height_);
        }
        
        // 获取模型输入尺寸
        std::pair<int, int> get_input_size() const {
            return {model_width_, model_height_};
        }
        
        // 预处理帧
        static ov::Tensor preprocess_frame(const cv::Mat& input_frame, const ov::Shape& input_shape) {
            cv::Mat resized_frame;
            cv::resize(input_frame, resized_frame, cv::Size(input_shape[2], input_shape[1]), 0, 0, cv::INTER_LINEAR);
            return ov::Tensor(ov::element::u8, input_shape, resized_frame.data);
        }
        
    private:
        // 表示带时间戳的推理请求
        struct TimedIreq {
            ov::InferRequest& ireq;  // 引用
            cv::Mat frame;           // 关联的帧
            bool has_started;        // 是否已启动推理
        };
        
        std::string model_path_;
        std::string device_name_;
        bool initialized_;
        
        ov::Core core_;
        ov::CompiledModel compiled_model_;
        ov::Shape input_shape_;
        int model_width_;
        int model_height_;
        
        YOLOXProcessor yolox_processor_;
        std::vector<ov::InferRequest> ireqs_;
        
        std::mutex mutex_;
        std::condition_variable cv_;
        std::exception_ptr callback_exception_;
        std::deque<TimedIreq> finished_ireqs_;
    };


