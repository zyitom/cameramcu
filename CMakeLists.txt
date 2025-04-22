cmake_minimum_required(VERSION 3.10)
project(helios_serial VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE "Release")

if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()


find_package(ament_cmake_auto REQUIRED)


ament_auto_find_build_dependencies()


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
else()
  set(EIGEN3_INCLUDE_DIR "/usr/include/eigen3")
  if(EXISTS "${EIGEN3_INCLUDE_DIR}")
    message(STATUS "Using manual Eigen3 path: ${EIGEN3_INCLUDE_DIR}")
  else()
    message(FATAL_ERROR "Eigen3 not found. Please install libeigen3-dev or specify the correct path.")
  endif()
endif()

find_package(Ceres REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread system chrono)


set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(SERIAL_SOURCES
    ${SRC_DIR}/Serial.cpp
    ${SRC_DIR}/CRC.cpp
    ${SRC_DIR}/PnPsolver.cpp
    ${SRC_DIR}/ovdetector.cpp
)


ament_auto_add_library(helios_serial SHARED ${SERIAL_SOURCES})

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
)


ament_auto_add_executable(serial_example ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

target_link_libraries(helios_serial
    ${OpenVINO_LIBRARIES}
    ${OpenCV_LIBS}
    ${gflags_LIBRARIES}
    ${TBB_LIBRARIES}
    ${Boost_LIBRARIES}
    ${CERES_LIBRARIES}
)

target_link_libraries(serial_example
    helios_serial
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
)

install(DIRECTORY 
  description
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_auto_package()