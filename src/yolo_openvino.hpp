#ifndef YOLO_OPENVINO_HPP
#define YOLO_OPENVINO_HPP

#include <ie_core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>

struct Detection {
    cv::Rect box;
    float conf;
    int class_id;
};

class YoloOpenVINO {
private:
    InferenceEngine::Core core;
    InferenceEngine::ExecutableNetwork executable_network;
    InferenceEngine::InferRequest infer_request;
    
    std::string input_name;
    std::string output_name;
    
    int input_width = 416;
    int input_height = 416;
    int blob_size = 416;
    
    // cache untuk hindari rekalkulasi saat blobsize sama
    int last_blob_size = -1;
    int last_orig_w = -1;
    int last_orig_h = -1;
    float cached_scale = 1.0f;
    int cached_new_w = 0;
    int cached_new_h = 0;
    int cached_pad_x = 0;
    int cached_pad_y = 0;
    
    // pre-allocated buffers
    cv::Mat padded_buffer;
    
    void printModelInfo();

public:
    YoloOpenVINO(const std::string& model_path, const std::string& device = "CPU");
    std::vector<Detection> infer(const cv::Mat& image);
    void setInputSize(int size);
};

#endif
