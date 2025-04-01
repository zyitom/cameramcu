#ifndef CAMERA_INPUT_H
#define CAMERA_INPUT_H

#include "ImageInput.h"

#include "CameraApi.h"

#include <iostream>
#include <string>
#include <yaml-cpp/yaml.h>
class MVCameraInput : public ImageInput {
private:
    unsigned char*          g_pRgbBuffer = nullptr; // 处理后图像输出的数据缓存区地址
    CameraHandle            hCamera;                // 相机句柄

public:
    MVCameraInput();
    bool        init()      final;
    cv::Mat     read()      final;
    void loadConfig(const std::string& config_file);

    void updateParameters();
    int exposure_time;
    int gamma;
    int rgb_gain_r;
    int rgb_gain_g;
    int rgb_gain_b;
    int analog_gain;
    bool flip_image;

    ~MVCameraInput();
};

#endif 
