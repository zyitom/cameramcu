#include "ovdetector.hpp"

OVnetDetector::OVnetDetector(const std::string& model_path, bool is_blue)
    : model_path_(model_path), device_name_("GPU"), is_blue_(is_blue), 
      initialized_(false), scale_x_(1.0f), scale_y_(1.0f) {}

OVnetDetector::~OVnetDetector() {
    for (auto& ireq : ireqs_) {
        try {
            ireq.cancel();
        } catch (...) {
            // 忽略异常
        }
    }
    
    // 清空所有资源
    available_ireqs_.clear();
    ireqs_.clear();
    compiled_model_ = ov::CompiledModel();
    core_ = ov::Core();
}

bool OVnetDetector::initialize() {
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
        
        init_processor(model_width_, model_height_);
        
        uint32_t nireq = compiled_model_.get_property(ov::optimal_number_of_infer_requests);
        std::cout << "创建 " << nireq << " 个推理请求" << std::endl;
        
        ireqs_.resize(nireq);
        std::generate(ireqs_.begin(), ireqs_.end(), [&] {
            return compiled_model_.create_infer_request();
        });
        
        // 初始化完成的推理请求队列
        for (auto& ireq : ireqs_) {
            available_ireqs_.push_back(TimedIreq(&ireq, FrameData{}, false));
        }
        
        initialized_ = true;
        return true;
    } 
    catch (const std::exception& ex) {
        std::cerr << "初始化异常: " << ex.what() << std::endl;
        return false;
    }
}

void OVnetDetector::init_processor(int input_width, int input_height) {
    static const int strides[3] = {8, 16, 32}; 
    grid_strides_.clear();
    generate_grids_and_stride(input_width, input_height, strides, grid_strides_);
    scale_x_ = static_cast<float>(input_width) / 1440;
    scale_y_ = static_cast<float>(input_height) / 1080;
}

ov::Tensor OVnetDetector::preprocess_frame(const cv::Mat& input_frame, const ov::Shape& input_shape) {
    cv::Mat resized_frame;
    cv::resize(input_frame, resized_frame, cv::Size(input_shape[2], input_shape[1]), 0, 0, cv::INTER_LINEAR);
    return ov::Tensor(ov::element::u8, input_shape, resized_frame.data);
}
FrameData OVnetDetector::infer_sync(const FrameData& input_data) {
    if (!initialized_) {
        throw std::runtime_error("Detector not initialized. Call initialize() first.");
    }
    
    // Create a copy of input to preserve timestamp and frame
    FrameData result = input_data;
    
    try {
        // Preprocess the frame
        ov::Tensor preprocessed = preprocess_frame(input_data.frame, input_shape_);
        
        // Get an available inference request
        std::unique_lock<std::mutex> lock(mutex_);
        if (available_ireqs_.empty()) {
            // Create a new inference request if none are available
            auto new_ireq = compiled_model_.create_infer_request();
            available_ireqs_.emplace_back(&new_ireq, FrameData(), false);
        }
        
        auto& timed_ireq = available_ireqs_.front();
        auto* ireq = timed_ireq.ireq;
        available_ireqs_.pop_front();  // Remove from available queue
        lock.unlock();
        
        // Set the input tensor for inference
        ireq->set_input_tensor(preprocessed);
        
        // Start synchronous inference
        ireq->infer();
        
        // Get the output tensor
        auto output_tensor = ireq->get_output_tensor();
        
        // Process the output to get armor detections
        std::vector<Armor> detected_armors = process_output(output_tensor);
        
        // Add results to the output frame
        result.add_armor_results(detected_armors);
        
        // Return the inference request to the available pool
        lock.lock();
        available_ireqs_.push_back(timed_ireq);
        lock.unlock();
        
    } catch (const std::exception& e) {
        std::cerr << "Error during synchronous inference: " << e.what() << std::endl;
        result.processed = true;  // Mark as processed but with empty armor vector
    }
    
    return result;
}
std::future<FrameData> OVnetDetector::submit_frame_async(const FrameData& input_data) {
    auto promise_ptr = std::make_shared<std::promise<FrameData>>();
    std::future<FrameData> future = promise_ptr->get_future();
    
    if (!initialized_ || input_data.frame.empty()) {
        promise_ptr->set_value(input_data);  // 返回原始帧
        return future;
    }
    
    std::unique_lock<std::mutex> lock(mutex_);
    
    // 等待有空闲的推理请求
    while (!callback_exception_ && available_ireqs_.empty()) {
        cv_.wait(lock);
    }
    
    if (callback_exception_) {
        promise_ptr->set_exception(callback_exception_);
        return future;
    }
    
    if (!available_ireqs_.empty()) {
        TimedIreq timedIreq = available_ireqs_.front();
        available_ireqs_.pop_front();
        lock.unlock();
        
        ov::InferRequest* ireq_ptr = timedIreq.ireq;
        
        try {
            ov::Tensor input_tensor = preprocess_frame(input_data.frame, input_shape_);
            ireq_ptr->set_input_tensor(input_tensor);
            
            ireq_ptr->set_callback(
                [ireq_ptr, input_data, promise_ptr, this](std::exception_ptr ex) {
                    std::unique_lock<std::mutex> lock(this->mutex_);
                    try {
                        ov::Tensor output_tensor = ireq_ptr->get_output_tensor();
                        std::vector<Armor> detected_objects = this->process_output(output_tensor);
                        
                        FrameData result = input_data;
                        result.add_armor_results(detected_objects);
                        
                        // 设置结果
                        promise_ptr->set_value(result);
                        
                        // 将推理请求放回空闲队列
                        this->available_ireqs_.push_back(TimedIreq(ireq_ptr, FrameData{}, false));
                    } catch (const std::exception& e) {
                        promise_ptr->set_exception(std::current_exception());
                        this->available_ireqs_.push_back(TimedIreq(ireq_ptr, FrameData{}, false));
                    }
                    this->cv_.notify_one();
                });
            
            ireq_ptr->start_async();
        } catch (const std::exception& e) {
            std::cerr << "推理异常: " << e.what() << std::endl;
            promise_ptr->set_exception(std::current_exception());
            std::unique_lock<std::mutex> lock(mutex_);
            available_ireqs_.push_back(TimedIreq(ireq_ptr, FrameData{}, false));
            cv_.notify_one();
        }
    } else {
        promise_ptr->set_value(input_data);  // 如果没有可用的推理请求，返回原始帧
    }
    
    return future;
}

std::vector<Armor> OVnetDetector::process_output(const ov::Tensor& output_tensor) {
    const float* output_buffer = output_tensor.data<const float>();
    std::vector<Object> objects;
    generate_yolox_proposal(output_buffer, objects);

    qsort_descent_inplace(objects);
    std::vector<int> picked;
    nms_sorted_bboxes(objects, picked);

    std::vector<Object> result;
    result.reserve(picked.size());
    for (int i : picked) {
        result.push_back(objects[i]);
    }

    avg_rect(result);
    std::vector<Armor> armors; 
    
    for (const auto& object : result) {
        // 如果是蓝色队伍检测蓝色装甲板，红色队伍检测红色装甲板
        // if ((is_blue_ && object.color != 0) || (!is_blue_ && object.color != 1)) {
        //     continue;
        // }
        
        Armor armor_target;
        armor_target.confidence = object.conf;
        armor_target.number = ARMOR_NUMBER_LABEL[object.label];
        
        armor_target.left_light.bottom = object.apexes[1];
        armor_target.left_light.top = object.apexes[0];
        armor_target.right_light.top = object.apexes[3];
        armor_target.right_light.bottom = object.apexes[2];
        armor_target.center = (object.apexes[0] + object.apexes[1] + object.apexes[2] + object.apexes[3]) * 0.25f;
        armor_target.type = judge_armor_type(armor_target);
        
        armors.push_back(armor_target);
    }
    
    return armors;
}

void OVnetDetector::generate_yolox_proposal(const float* output_buffer, std::vector<Object>& objects) {
    const int num_anchors = grid_strides_.size();
    const int class_start = 2 * NUM_APEX + 1;

    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) {
        const int basic_pos = anchor_idx * (class_start + NUM_CLASS + NUM_COLORS);
        float box_conf = output_buffer[basic_pos + 2 * NUM_APEX];

        if (box_conf >= CLASSIFIER_THRESHOLD) {
            const GridAndStride& gs = grid_strides_[anchor_idx];
            Object obj;
            obj.apexes.reserve(NUM_APEX);
            
            for (int i = 0; i < NUM_APEX; i++) {
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

void OVnetDetector::qsort_descent_inplace(std::vector<Object>& objects) {
    std::sort(objects.begin(), objects.end(), 
             [](const Object& a, const Object& b) { return a.conf > b.conf; });
}

void OVnetDetector::nms_sorted_bboxes(std::vector<Object>& faceobjects, std::vector<int>& picked) {
    picked.clear();
    const int n = faceobjects.size();
    picked.reserve(n);

    for (int i = 0; i < n; i++) {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++) {
            const Object& b = faceobjects[picked[j]];

            float inter_area = intersection_area(a, b);
            float union_area = a.rect.area() + b.rect.area() - inter_area;
            float iou = inter_area / union_area;

            if (iou > NMS_THRESH || std::isnan(iou)) {
                keep = 0;
                if (iou > 0.9 && std::abs(a.conf - b.conf) < 0.2 && a.label == b.label && a.color == b.color) {
                    faceobjects[picked[j]].apexes.insert(faceobjects[picked[j]].apexes.end(), 
                                                       a.apexes.begin(), a.apexes.end());
                }
                break;
            }
        }

        if (keep) {
            picked.push_back(i);
        }
    }
}

void OVnetDetector::avg_rect(std::vector<Object>& objects) {
    for (auto& object : objects) {
        std::size_t N = object.apexes.size();

        if (N >= 2 * NUM_APEX) {
            std::vector<cv::Point2f> fin_point(NUM_APEX, cv::Point2f(0, 0));

            for (size_t i = 0; i < N; i++) {
                fin_point[i % NUM_APEX] += object.apexes[i];
            }

            for (int i = 0; i < NUM_APEX; i++) {
                fin_point[i] /= static_cast<float>(N / NUM_APEX);
            }

            object.apexes = std::move(fin_point);
            object.rect = cv::boundingRect(object.apexes);
        }
    }
}

int OVnetDetector::argmax(const float* ptr, int len) {
    return std::max_element(ptr, ptr + len) - ptr;
}

float OVnetDetector::intersection_area(const Object& a, const Object& b) {
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

void OVnetDetector::generate_grids_and_stride(const int w, const int h, const int strides[],
                                         std::vector<GridAndStride>& grid_strides) {
    grid_strides.clear();
    grid_strides.reserve(w * h / 64 + w * h / 256 + w * h / 1024);
    
    for (int i = 0; i < 3; i++) {
        int num_grid_w = w / strides[i];
        int num_grid_h = h / strides[i];

        for (int g1 = 0; g1 < num_grid_h; g1++) {
            for (int g0 = 0; g0 < num_grid_w; g0++) {
                grid_strides.push_back({g0, g1, strides[i]});
            }
        }
    }
}

ArmorType OVnetDetector::judge_armor_type(const Armor& armor) {
    cv::Point2f light_center1 = (armor.left_light.top + armor.left_light.bottom) / 2.0;
    cv::Point2f light_center2 = (armor.right_light.top + armor.right_light.bottom) / 2.0;
    float light_length1 = cv::norm(armor.left_light.top - armor.left_light.bottom);
    float light_length2 = cv::norm(armor.right_light.top - armor.right_light.bottom);

    float avg_light_length = (light_length1 + light_length2) / 2.0;
    float center_distance = cv::norm(light_center1 - light_center2) / avg_light_length;
    
    // 这里用的是定制判断比例
    return center_distance > 3.6 ? ArmorType::LARGE : ArmorType::SMALL;
}

void OVnetDetector::set_scale(float scale_x, float scale_y) {
    scale_x_ = scale_x;
    scale_y_ = scale_y;
    init_processor(model_width_, model_height_);
}

std::pair<int, int> OVnetDetector::get_input_size() const {
    return {model_width_, model_height_};
}

cv::Mat OVnetDetector::visualize_detection_result(const FrameData& frame_data) {
    if (!frame_data.processed || frame_data.frame.empty() || frame_data.armors.empty()) {
        return frame_data.frame.clone();  
    }
    
    cv::Mat vis = frame_data.frame.clone();
    
    for (const auto& armor : frame_data.armors) {
        cv::Point left_top(armor.left_light.top.x, armor.left_light.top.y);
        cv::Point right_top(armor.right_light.top.x, armor.right_light.top.y);
        cv::Point right_bottom(armor.right_light.bottom.x, armor.right_light.bottom.y);
        cv::Point left_bottom(armor.left_light.bottom.x, armor.left_light.bottom.y);
        
        std::vector<cv::Point> pts = {left_top, right_top, right_bottom, left_bottom};
        
        cv::Scalar color;
        if (armor.type == ArmorType::SMALL) {
            color = cv::Scalar(0, 255, 0); 
        } else {
            color = cv::Scalar(0, 0, 255); 
        }
        
        cv::polylines(vis, pts, true, color, 2);
        
        cv::putText(vis, "LT", left_top, cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 0), 1);
        cv::putText(vis, "RT", right_top, cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 0), 1);
        cv::putText(vis, "RB", right_bottom, cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 0), 1);
        cv::putText(vis, "LB", left_bottom, cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 0), 1);
        
        cv::circle(vis, left_top, 4, cv::Scalar(255, 0, 0), -1);      // 左上 - 蓝色
        cv::circle(vis, right_top, 4, cv::Scalar(0, 255, 255), -1);   // 右上 - 黄色
        cv::circle(vis, right_bottom, 4, cv::Scalar(255, 0, 255), -1); // 右下 - 紫色
        cv::circle(vis, left_bottom, 4, cv::Scalar(0, 165, 255), -1); // 左下 - 橙色
        
        cv::Point center(armor.center.x, armor.center.y);
        cv::circle(vis, center, 3, cv::Scalar(0, 255, 0), -1);
        
        char text[64];
        sprintf(text, "%s %.1f%%", armor.number.c_str(), armor.confidence * 100);
        
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::Point text_pos(center.x - text_size.width/2, center.y - 10);
        
        cv::rectangle(vis, 
                    cv::Point(text_pos.x, text_pos.y - text_size.height),
                    cv::Point(text_pos.x + text_size.width, text_pos.y + baseline),
                    color, -1);
        cv::putText(vis, text, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
    
    cv::imshow("Detection Points Check", vis);
    cv::waitKey(1);
    return vis;
}

void OVnetDetector::set_is_blue(bool is_blue) {
    is_blue_ = is_blue;
}