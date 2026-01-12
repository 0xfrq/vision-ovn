#include "yolo_openvino.hpp"

using namespace std;
using namespace InferenceEngine;

YoloOpenVINO::YoloOpenVINO(const string& model_path, const string& device)
{
    cout << "Loading OpenVINO 2021.4 model: " << model_path << endl;
    
    // load network dari file IR (.xml + .bin)
    CNNNetwork network = core.ReadNetwork(model_path);
    
    // get input info
    InputsDataMap inputs_info = network.getInputsInfo();
    for(auto& item : inputs_info) {
        input_name = item.first;
        auto input_data = item.second;
        
        // set precision dan layout untuk performa optimal
        input_data->setPrecision(Precision::FP32);
        input_data->setLayout(Layout::NCHW);
        
        // get input dimensions
        auto dims = input_data->getTensorDesc().getDims();
        if(dims.size() == 4) {
            input_height = dims[2];
            input_width = dims[3];
        }
    }
    
    // get output info
    OutputsDataMap outputs_info = network.getOutputsInfo();
    for(auto& item : outputs_info) {
        output_name = item.first;
        item.second->setPrecision(Precision::FP32);
    }
    
    // konfigurasi performa untuk CPU
    map<string, string> config;
    if(device == "CPU") {
        config[PluginConfigParams::KEY_CPU_THREADS_NUM] = "4";
        config[PluginConfigParams::KEY_CPU_BIND_THREAD] = PluginConfigParams::YES;
        config[PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS] = "1";
    }
    
    // load executable network
    executable_network = core.LoadNetwork(network, device, config);
    
    // buat inference request
    infer_request = executable_network.CreateInferRequest();
    
    // pre-allocate buffer
    padded_buffer = cv::Mat::zeros(input_height, input_width, CV_8UC3);
    
    printModelInfo();
    cout << "OpenVINO 2021.4 model loaded successfully on " << device << endl;
}

void YoloOpenVINO::printModelInfo() {
    cout << "Input name: " << input_name << endl;
    cout << "Input size: " << input_width << "x" << input_height << endl;
    cout << "Output name: " << output_name << endl;
}

vector<Detection> YoloOpenVINO::infer(const cv::Mat& image)
{
    int orig_w = image.cols;
    int orig_h = image.rows;
    
    // hitung scale dan padding hanya jika blobsize atau ukuran gambar berubah
    if(blob_size != last_blob_size || orig_w != last_orig_w || orig_h != last_orig_h) {
        cached_scale = (float)blob_size / max(orig_w, orig_h);
        cached_new_w = (int)(orig_w * cached_scale);
        cached_new_h = (int)(orig_h * cached_scale);
        cached_pad_x = (input_width - cached_new_w) / 2;
        cached_pad_y = (input_height - cached_new_h) / 2;
        last_blob_size = blob_size;
        last_orig_w = orig_w;
        last_orig_h = orig_h;
        
        // reset padded buffer ke hitam
        padded_buffer.setTo(cv::Scalar(0, 0, 0));
    }
    
    // resize langsung ke roi dalam padded buffer
    cv::Mat roi = padded_buffer(cv::Rect(cached_pad_x, cached_pad_y, cached_new_w, cached_new_h));
    cv::resize(image, roi, cv::Size(cached_new_w, cached_new_h), 0, 0, cv::INTER_LINEAR);
    
    // get input blob dan isi data
    Blob::Ptr input_blob = infer_request.GetBlob(input_name);
    auto input_data = input_blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
    
    // normalize dan convert BGR->RGB dalam satu pass (NCHW format)
    const uchar* src = padded_buffer.ptr<uchar>();
    float* r_ptr = input_data;
    float* g_ptr = r_ptr + input_width * input_height;
    float* b_ptr = g_ptr + input_width * input_height;
    
    const float inv255 = 1.0f / 255.0f;
    int total = input_width * input_height;
    for(int i = 0; i < total; i++) {
        int idx = i * 3;
        r_ptr[i] = src[idx + 2] * inv255;  // R
        g_ptr[i] = src[idx + 1] * inv255;  // G
        b_ptr[i] = src[idx + 0] * inv255;  // B
    }
    
    // run inference
    infer_request.Infer();
    
    // get output blob
    Blob::Ptr output_blob = infer_request.GetBlob(output_name);
    auto output_data = output_blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
    
    // get output dimensions
    auto out_dims = output_blob->getTensorDesc().getDims();
    int num_boxes = out_dims[1];
    int elements = out_dims[2];
    
    float inv_scale = 1.0f / cached_scale;
    
    vector<Detection> detections;
    detections.reserve(20);
    
    // threshold rendah di sini, filter final dilakukan di main.cpp (0.5f seperti Python)
    const float conf_threshold = 0.25f;
    
    for(int i = 0; i < num_boxes; i++) {
        float conf = output_data[i * elements + 4];
        if(conf < conf_threshold) continue;
        
        float cx = output_data[i * elements + 0] - cached_pad_x;
        float cy = output_data[i * elements + 1] - cached_pad_y;
        float w = output_data[i * elements + 2];
        float h = output_data[i * elements + 3];
        
        // expand box sedikit untuk capture full ball (10% padding)
        float expand = 0.1f;
        w *= (1.0f + expand);
        h *= (1.0f + expand);
        
        int x = (int)((cx - w * 0.5f) * inv_scale);
        int y = (int)((cy - h * 0.5f) * inv_scale);
        int width = (int)(w * inv_scale);
        int height = (int)(h * inv_scale);
        
        // clamp ke batas frame
        x = max(0, min(x, orig_w - 1));
        y = max(0, min(y, orig_h - 1));
        width = min(width, orig_w - x);
        height = min(height, orig_h - y);
        
        if(width > 5 && height > 5) {
            Detection det;
            det.box = cv::Rect(x, y, width, height);
            det.conf = conf;
            det.class_id = 0;
            detections.push_back(det);
        }
    }
    
    // simple NMS untuk gabungkan deteksi overlapping
    if(detections.size() > 1) {
        vector<Detection> nms_result;
        vector<bool> suppressed(detections.size(), false);
        
        // sort by confidence
        sort(detections.begin(), detections.end(), 
             [](const Detection& a, const Detection& b) { return a.conf > b.conf; });
        
        for(size_t i = 0; i < detections.size(); i++) {
            if(suppressed[i]) continue;
            nms_result.push_back(detections[i]);
            
            for(size_t j = i + 1; j < detections.size(); j++) {
                if(suppressed[j]) continue;
                
                // hitung IoU
                cv::Rect inter = detections[i].box & detections[j].box;
                float inter_area = inter.area();
                float union_area = detections[i].box.area() + detections[j].box.area() - inter_area;
                float iou = inter_area / union_area;
                
                if(iou > 0.4f) suppressed[j] = true;
            }
        }
        return nms_result;
    }
    
    return detections;
}

void YoloOpenVINO::setInputSize(int size) {
    blob_size = size;
}
