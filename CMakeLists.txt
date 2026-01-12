cmake_minimum_required(VERSION 3.5)
project(vision_ovn)

set(CMAKE_CXX_STANDARD 14)
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

# Find OpenVINO 2021.4 (InferenceEngine)
find_package(InferenceEngine REQUIRED)

# Catkin package configuration
catkin_package(
    CATKIN_DEPENDS roscpp roslib std_msgs v2_detection
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${InferenceEngine_INCLUDE_DIRS}
)

add_executable(vision
    src/main.cpp
    src/yolo_openvino.cpp
)

target_link_libraries(vision
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${InferenceEngine_LIBRARIES}
)