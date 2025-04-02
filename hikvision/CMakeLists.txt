find_package(OpenCV 4 REQUIRED)
find_package(yaml-cpp REQUIRED)


set(HIKVISION_LIB_DIR /opt/MVS/lib/64)
include_directories(/opt/MVS/include)

set(HIKVISION_CAMERA_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(HIK_CONFIG_FILE_PATH ${HIKVISION_CAMERA_DIR}/config)
set(HIK_CALI_FILE_PATH ${HIKVISION_CAMERA_DIR}/config/cameraCaliData)

include_directories(${HIKVISION_CAMERA_DIR}/include)
include_directories(${HIKVISION_CAMERA_DIR}/tool/include)

file(GLOB_RECURSE HIKVISION_SOURCES 
    "${HIKVISION_CAMERA_DIR}/src/*.cpp" 
    "${HIKVISION_CAMERA_DIR}/tool/src/*.cpp"
)

add_library(hikvision_camera STATIC ${HIKVISION_SOURCES})

target_link_directories(hikvision_camera PUBLIC ${HIKVISION_LIB_DIR})

target_compile_definitions(hikvision_camera PUBLIC 
    "HIK_CONFIG_FILE_PATH=\"${HIK_CONFIG_FILE_PATH}\""
    "HIK_CALI_FILE_PATH=\"${HIK_CALI_FILE_PATH}\""
)

target_link_libraries(hikvision_camera PUBLIC 
    ${OpenCV_LIBS}
    MvCameraControl
    yaml-cpp
)

set(HIK_LIB_DIRS ${HIKVISION_LIB_DIR} PARENT_SCOPE)

target_include_directories(hikvision_camera PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/tool/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(hikvision_camera INTERFACE MvCameraControl)