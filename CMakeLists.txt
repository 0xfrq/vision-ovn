cmake_minimum_required(VERSION 3.5)
project(vision_ovn)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Find catkin and required ROS packages
find_package(catkin REQUIRED COMPONENTS
    roscpp
    roslib
    std_msgs
    v2_detection
)

find_package(OpenCV REQUIRED)

# OpenVINO 2020.4 paths
set(OPENVINO_DIR /opt/intel/l_openvino_toolkit_runtime_ubuntu16_p_2020.4.287)
set(IE_INCLUDE_DIR ${OPENVINO_DIR}/inference_engine/include)
set(IE_LIB_DIR ${OPENVINO_DIR}/inference_engine/lib/intel64)

# Catkin package configuration
catkin_package(
    CATKIN_DEPENDS roscpp roslib std_msgs v2_detection
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${IE_INCLUDE_DIR}
)

link_directories(${IE_LIB_DIR})

add_executable(vision_ovn
    src/main.cpp
    src/yolo_openvino.cpp
)

target_link_libraries(vision_ovn
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    inference_engine
    inference_engine_legacy
    pthread
)