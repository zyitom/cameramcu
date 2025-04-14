cmake_minimum_required(VERSION 3.10)
project(helios_serial VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Add compile options
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Add minimal ROS 2 packages needed for compilation
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)

# Original project dependencies
add_subdirectory(hikvision)
add_subdirectory(serial)

find_package(Threads REQUIRED)
find_package(OpenVINO REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP)
find_package(gflags REQUIRED)
find_package(TBB REQUIRED)
find_package(Eigen3 REQUIRED)
if(EIGEN3_FOUND)
  message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIR}")
  include_directories(${EIGEN3_INCLUDE_DIR})
else()
  set(EIGEN3_INCLUDE_DIR "/usr/include/eigen3")
  if(EXISTS "${EIGEN3_INCLUDE_DIR}")
    message(STATUS "Using manual Eigen3 path: ${EIGEN3_INCLUDE_DIR}")
    include_directories(${EIGEN3_INCLUDE_DIR})
  else()
    message(FATAL_ERROR "Eigen3 not found. Please install libeigen3-dev or specify the correct path.")
  endif()
endif()

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread system chrono)

set(SERIAL_SOURCES
    ${SRC_DIR}/Serial.cpp
    ${SRC_DIR}/CRC.cpp
    ${SRC_DIR}/PnPsolver.cpp
)

set(SERIAL_HEADERS
    ${INCLUDE_DIR}/Serial.hpp
    ${INCLUDE_DIR}/CRC.h
    ${INCLUDE_DIR}/Armor.hpp
    ${INCLUDE_DIR}/PnPSolver.hpp
)

add_library(helios_serial ${SERIAL_SOURCES})

target_include_directories(helios_serial PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${OpenVINO_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}
    ${gflags_INCLUDE_DIR}
    ${TBB_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${CERES_INCLUDE_DIRS}
    ${rclcpp_INCLUDE_DIRS}
)

add_executable(serial_example ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(serial_example PRIVATE helios_serial)

target_link_libraries(serial_example PRIVATE
    hikvision_camera
    Threads::Threads
    serial
    openvino::runtime
    ${OpenCV_LIBS}
    gflags
    pthread
    TBB::tbb
    ${Boost_LIBRARIES}
    ${CERES_LIBRARIES}
    ${rclcpp_LIBRARIES}
)

# Minimal ROS 2 installation targets
install(
    TARGETS helios_serial serial_example
    RUNTIME DESTINATION lib/${PROJECT_NAME}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Finalize ROS 2 package
ament_package()