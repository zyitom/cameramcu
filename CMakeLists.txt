cmake_minimum_required(VERSION 3.10)
project(HeliosSerial VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Set directories
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Add compile options
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Check platform for platform-specific libraries
if(WIN32)
    # Windows-specific settings
    message(STATUS "Configuring for Windows platform")
    # No additional libraries needed for Windows serial port
else()
    # Linux/Unix-specific settings
    message(STATUS "Configuring for Unix/Linux platform")
    # No additional libraries needed for POSIX serial port
endif()

# Define library sources
set(SERIAL_SOURCES
    ${SRC_DIR}/Serial.cpp
    ${SRC_DIR}/CRC.cpp
)

# Define library headers
set(SERIAL_HEADERS
    ${INCLUDE_DIR}/Serial.hpp
    ${INCLUDE_DIR}/CRC.h
)

# Create the library
add_library(helios_serial ${SERIAL_SOURCES})

# Set include directories
target_include_directories(helios_serial PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Add example executable
add_executable(serial_example ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(serial_example PRIVATE helios_serial)

# Link with platform-specific libraries
if(WIN32)
    # Windows requires additional system libraries
    target_link_libraries(helios_serial PRIVATE setupapi)
else()
    # Linux might need pthread explicitly
    find_package(Threads REQUIRED)
    target_link_libraries(helios_serial PRIVATE Threads::Threads)
    target_link_libraries(serial_example PRIVATE Threads::Threads)
endif()

# Installation rules
include(GNUInstallDirs)

install(TARGETS helios_serial serial_example
    EXPORT HeliosSerialTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${SERIAL_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/helios)