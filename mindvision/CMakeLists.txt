cmake_minimum_required(VERSION 3.1)
project(mindvision)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

add_library(mindvision_input
    src/MVCameraInput.cpp
    src/VideoInput.cpp
)

target_include_directories(mindvision_input
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/MVSDK/include
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${YAML_CPP_INCLUDE_DIR}
)

target_link_libraries(mindvision_input
    PUBLIC
        ${OpenCV_LIBS}
        yaml-cpp
)

find_library(MVSDK_LIB MVSDK)
if(NOT MVSDK_LIB)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        set(MVSDK_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/MVSDK/lib/arm64)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
        set(MVSDK_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/MVSDK/lib/x64)
    else()
        message(FATAL_ERROR "Unsupported CPU architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()

    find_library(MVSDK_LIB MVSDK PATHS ${MVSDK_LIB_DIR} NO_DEFAULT_PATH)
endif()

if(MVSDK_LIB)
    target_link_libraries(mindvision_input PUBLIC ${MVSDK_LIB})
    message(STATUS "Found MVSDK library: ${MVSDK_LIB}")
else()
    message(FATAL_ERROR "MVSDK library not found")
endif()
