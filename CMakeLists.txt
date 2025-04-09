cmake_minimum_required(VERSION 3.10)
project(HeliosSerial VERSION 1.0.0 LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 14)
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
add_subdirectory(hikvision)
add_subdirectory(serial)
find_package(Threads REQUIRED)
find_package(OpenVINO REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP)
find_package(gflags REQUIRED)
find_package(TBB REQUIRED)

message(STATUS "Configuring for Unix/Linux platform")


set(SERIAL_SOURCES
    ${SRC_DIR}/Serial.cpp
    ${SRC_DIR}/CRC.cpp
)


set(SERIAL_HEADERS
    ${INCLUDE_DIR}/Serial.hpp
    ${INCLUDE_DIR}/CRC.h
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
    TBB::tbb)

include(GNUInstallDirs)

