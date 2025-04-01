#include "MVCameraInput.h"
MVCameraInput::MVCameraInput()
{
    if(this->init()){

        tSdkCameraCapbility     tCapability;       // 设备描述信息

        if(CameraGetCapability(hCamera, &tCapability)){};

        this->setResolution(cv::Size(
            tCapability.sResolutionRange.iWidthMax,
            tCapability.sResolutionRange.iHeightMax
        ));
        std::cout << "Camera resolution: " << tCapability.sResolutionRange.iWidthMax << "x" << tCapability.sResolutionRange.iHeightMax << std::endl;
    }else{
        std::cerr << "Failed to initialize the camera!\n";
    };
}

bool MVCameraInput::init()
{
    tSdkCameraDevInfo       tCameraEnumList;   // 相机列表
    int                     iCameraCounts = 1; // 摄像头数量
    int                     iStatus = -1;      // 相机初始化返回值
    tSdkCameraCapbility     tCapability;       // 设备描述信息
    int                     nChannel = 3;

    CameraSdkInit(1);

    //枚举设备，并建立设备列表
    //如果需要连接多个设备，则将tCameraEnumList改为数组，同时修改iCameraCounts的值
    CameraEnumerateDevice(&tCameraEnumList, &iCameraCounts);
    std::cout<<tCameraEnumList.acProductName<<std::endl;
    //没有连接设备
    if (iCameraCounts == 0)
    {
        std::cerr << "No camera was found!\n";
        this->opened = false;
        return false;
    }

    do {
        //相机初始化。初始化成功后，才能调用任何其他相机相关的操作接口
        //该示例中，我们只假设连接了一个相机。因此，只初始化第一个相机。
        //(-1,-1)表示加载上次退出前保存的参数，如果是第一次使用该相机，则加载默认参数.
        iStatus = CameraInit(&tCameraEnumList, -1, -1, &hCamera);
        if (iStatus != CAMERA_STATUS_SUCCESS) { 
            std::cerr << "Failed to init the camera! Error code is " + std::to_string(iStatus) + "\n"; 
            std::cerr << CameraGetErrorString(iStatus);
        }
    } while (iStatus != CAMERA_STATUS_SUCCESS); //初始化失败

    //获得相机的特性描述结构体。该结构体中包含了相机可设置的各种参数的范围信息。决定了相关函数的参数
    CameraGetCapability(hCamera, &tCapability);

    if (tCapability.sIspCapacity.bMonoSensor) {
        nChannel = 1;
        CameraSetIspOutFormat(hCamera,CAMERA_MEDIA_TYPE_MONO8);
    } else {
        nChannel = 3;
        CameraSetIspOutFormat(hCamera,CAMERA_MEDIA_TYPE_BGR8);
    }

    // 分配图像缓冲区
    g_pRgbBuffer =  new unsigned char[tCapability.sResolutionRange.iHeightMax * tCapability.sResolutionRange.iWidthMax * nChannel]; 

    /*让SDK进入工作模式，开始接收来自相机发送的图像
    数据。如果当前相机是触发模式，则需要接收到
    触发帧以后才会更新图像。    */
    CameraPlay(hCamera);
    CameraSetExposureTime(hCamera, 3000);

    /*其他的相机参数设置
    例如 CameraSetExposureTime   CameraGetExposureTime  设置/读取曝光时间
         CameraSetImageResolution  CameraGetImageResolution 设置/读取分辨率
         CameraSetGamma、CameraSetConrast、CameraSetGain等设置图像伽马、对比度、RGB数字增益等等。
         更多的参数的设置方法，，清参考MindVision_Demo。本例程只是为了演示如何将SDK中获取的图像，转成OpenCV的图像格式,以便调用OpenCV的图像处理函数进行后续开发
    */
    //printf("iFrameSpeedDesc = %d\n",tCapability.iFrameSpeedDesc);         //frame rate option number,when set,iframeSpeed range (0,iFrameSpeedDec-1)
    CameraSetFrameSpeed(hCamera, 2);          //set frame rate,the bigger,the quicker(0~2)
    int currentTriggerMode;

    CameraSdkStatus triggerStatus = CameraSetTriggerMode(hCamera, 0);

    if (triggerStatus != CAMERA_STATUS_SUCCESS) {
        std::cerr << "Failed to set trigger mode!\n";
        exit(-1);
    }
    //CameraSetAnalogGain(hCamera, 4);           //mo ni zeng yi
    //int analoggain;
    //CameraGetAnalogGain(hCamera, &analoggain);
    //std::cout << "analoggain = " << analoggain << std::endl;
    CameraSetAeState(hCamera, false);             //设置为手动曝光
    //if (CameraSetExposureTime(hCamera, 1000) == 0) //单位为微秒
    //    std::cout << "set exposure successful!\n";
    //else
    //    std::cout << "set exposure failed!\n";
    //double expTime = 0;
    //CameraGetExposureTime(hCamera, &expTime);
    //std::cout << "explore time = " << expTime << "us" << std::endl;
    
    updateParameters();

    this->opened = true;
    return true;
}

void MVCameraInput::loadConfig(const std::string& config_file)
{
    try {
        YAML::Node config = YAML::LoadFile(config_file);
        exposure_time = config["exposure_time"].as<int>();
        gamma = config["gamma"].as<int>();
    
        if (config["rgb_gain"]["rgb_camera"].as<bool>()) {
            rgb_gain_r = config["rgb_gain"]["r"].as<int>();
            rgb_gain_g = config["rgb_gain"]["g"].as<int>();
            rgb_gain_b = config["rgb_gain"]["b"].as<int>();
        } else {
            rgb_gain_r = rgb_gain_g = rgb_gain_b = 0;
        }

        analog_gain = config["analog_gain"].as<int>();
        flip_image = config["flip_image"].as<bool>();
        std::cout << "Loaded camera parameters from " << config_file << std::endl;
        std::cout << "Exposure time: " << exposure_time << std::endl;
        std::cout << "Gamma: " << gamma << std::endl;
        std::cout << "RGB gain: " << rgb_gain_r << ", " << rgb_gain_g << ", " << rgb_gain_b << std::endl;
        std::cout << "Analog gain: " << analog_gain << std::endl;
        std::cout << "Flip image: " << flip_image << std::endl;
        updateParameters();
    } catch (const YAML::Exception& e) {
        std::cerr << "Error loading YAML file: " << e.what() << std::endl;
    }
}

void MVCameraInput::updateParameters()
{
    if (!opened) return;

    CameraSetExposureTime(hCamera, exposure_time);
    CameraSetGamma(hCamera, gamma);
    CameraSetGain(hCamera, rgb_gain_r, rgb_gain_g, rgb_gain_b);
    CameraSetAnalogGain(hCamera, analog_gain);
    
    if (flip_image) {
        CameraSetMirror(hCamera, 0, true);  // Vertical flip
    } else {
        CameraSetMirror(hCamera, 0, false);
    }
    
}

// cv::Mat MVCameraInput::read() {
//     INT iWidth, iHeight;
//     unsigned char* pRGBBuffer;
//     cv::Mat matImg;
    
//     // 使用CameraGetImageBufferEx获取图像数据，超时时间100ms
//     pRGBBuffer = CameraGetImageBufferEx(hCamera, &iWidth, &iHeight, 100);
    
//     if (pRGBBuffer) {
//         // 创建OpenCV Mat对象
//         // 注意：CameraGetImageBufferEx返回的是RGB格式，而不是BGR格式
//         // 因此需要设置正确的色彩空间
//         cv::Mat tempImg(cv::Size(iWidth, iHeight), CV_8UC3, pRGBBuffer);
        
//         // 如果需要，转换颜色空间从RGB到BGR (OpenCV默认使用BGR)
//         cv::cvtColor(tempImg, matImg, cv::COLOR_RGB2BGR);
        
//         // 如果需要翻转图像
//         if (flip_image) {
//             cv::flip(matImg, matImg, 0);  // 垂直翻转
//         }
        
//         // 调整图像大小
//         cv::resize(matImg, matImg, this->imgResolution);
        
//         // 注意：使用CameraGetImageBufferEx时不需要调用CameraReleaseImageBuffer
//     }
    
//     return matImg;
// }

cv::Mat MVCameraInput::read()
{
    tSdkFrameHead           sFrameInfo;        //输入图像的帧头信息
    BYTE*                   pbyBuffer;         //输出图像数据的缓冲区地址
    CameraSdkStatus         status;

    cv::Mat matImg;

    // 第四个参数为超时时间，单位为ms
    if (CameraGetImageBuffer(hCamera, &sFrameInfo, &pbyBuffer, 100) == CAMERA_STATUS_SUCCESS) 
    {
        status = CameraImageProcess(hCamera, pbyBuffer, g_pRgbBuffer, &sFrameInfo);

        if (status == CAMERA_STATUS_SUCCESS) {

        if(flip_image){
                    CameraFlipFrameBuffer(g_pRgbBuffer, &sFrameInfo, 1);
        }

            matImg = cv::Mat(
                cv::Size(sFrameInfo.iWidth, sFrameInfo.iHeight),
                sFrameInfo.uiMediaType == CAMERA_MEDIA_TYPE_MONO8 ? CV_8UC1 : CV_8UC3,
                g_pRgbBuffer
            );

            auto begin_timestamp = cv::getTickCount();

            cv::resize(matImg, matImg, this->imgResolution);

            auto elapsed_seconds = (cv::getTickCount() - begin_timestamp) / cv::getTickFrequency();
        }

        //在成功调用CameraGetImageBuffer后，必须调用CameraReleaseImageBuffer来释放获得的buffer。
        //否则再次调用CameraGetImageBuffer时，程序将被挂起一直阻塞，直到其他线程中调用CameraReleaseImageBuffer来释放了buffer
        CameraReleaseImageBuffer(hCamera, pbyBuffer);
    }
    
    return matImg;
}


MVCameraInput::~MVCameraInput()
{
    CameraUnInit(hCamera);
    //注意，先反初始化后再释放图像缓冲区
    // delete g_pRgbBuffer;
}
