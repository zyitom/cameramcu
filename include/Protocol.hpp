#pragma once

#include <cstdint>
#include <cmath>

#define __packed __attribute__((__packed__))

namespace helios
{

typedef enum CMD_ID
{
  SEND_TARGET_INFO_CMD_ID = 0x0002,
  RECEIVE_AUTOAIM_RECEIVE_CMD_ID = 0x0107
} CMD_ID;

typedef struct __packed
{
  uint8_t gimbal_id;  // 0为左云台，1为右云台
  uint8_t tracking;   // tracking = 1或0是指当前上位机的predictor的整车观测器当前是否观测到了车辆或者打符是否在观测中
  uint8_t id;         // outpost = 8  guard = 7  base = 9   energy = 10
  uint8_t armors_num; // 2-balance 3-outpost 4-normal  5-energy  1-armor
  float x;            // 车辆中心x  (能量机关装甲板x)
  float y;            // 车辆中心y  (能量机关装甲板y)
  float z;            // 车辆中心z  (能量机关装甲板z)
  float yaw;          // 面对我们的装甲板的yaw值
  float vx;           // 类推
  float vy;
  float vz;
  float v_yaw;
  float r1;           // 车辆第一个半径
  float r2;           // 车辆第二个半径
  float dz;           // 两对装甲板之间的高低差值
  float vision_delay; // 视觉处理消耗的全部时间
} TargetInfo;

typedef struct __packed
{
  uint8_t self_color;     // 自身颜色 0 蓝 1 红
  uint8_t autoaim_mode;   // 0自瞄 1 小符 2 大符
  uint8_t use_traditional;// 0 use net, 1 use traditional
  float bullet_speed;     // 弹速
  float yaw;              // 直接转发陀螺仪yaw即可
  float pitch;            // 直接转发陀螺仪pitch即可
  float roll;             // 直接转发陀螺仪的roll即可
  float x;                // final predicted target x
  float y;                // final predicted target y
  float z;                // final predicted target z
} MCUPacket;

} // namespace helios