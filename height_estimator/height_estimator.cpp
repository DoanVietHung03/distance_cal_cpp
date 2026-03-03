#include "height_estimator.h"

#include "../config_file/json.hpp"
using json = nlohmann::json;
#include <fstream>

// Constructor khởi tạo giá trị ban đầu
HeightEstimator::HeightEstimator() : fx(0.0), fy(0.0), loaded(false) {}

bool HeightEstimator::load_focal_length(const std::string& calib_file, double current_width) {
    try {
        // Try to support both OpenCV FileStorage (YAML/XML) and plain JSON files.
        cv::Mat K;
        double orig_w = 1280.0; // fallback if resolution not found

        // determine file extension (simple check)
        auto pos = calib_file.find_last_of('.');
        std::string ext = (pos == std::string::npos) ? std::string() : calib_file.substr(pos);

        if (ext == ".json" || ext == ".JSON") {
            // Parse with nlohmann::json to avoid OpenCV asserting on unexpected JSON structure
            std::ifstream ifs(calib_file);
            if (!ifs.is_open()) {
                throw std::runtime_error("Khong the mo file calibration.");
            }

            json j;
            ifs >> j;

            // camera_matrix can be represented as a 3x3 array or an object with a "data" array
            if (j.contains("camera_matrix")) {
                auto cm = j["camera_matrix"];
                if (cm.is_array() && cm.size() == 3 && cm[0].is_array()) {
                    K = cv::Mat(3, 3, CV_64F);
                    for (int r = 0; r < 3; ++r)
                        for (int c = 0; c < 3; ++c)
                            K.at<double>(r, c) = cm[r][c].get<double>();
                }
                else if (cm.is_object() && cm.contains("data") && cm["data"].is_array()) {
                    auto data = cm["data"];
                    if (data.size() >= 9) {
                        K = cv::Mat(3, 3, CV_64F);
                        for (int r = 0; r < 3; ++r)
                            for (int c = 0; c < 3; ++c)
                                K.at<double>(r, c) = data[r * 3 + c].get<double>();
                    }
                }
            }

            if (j.contains("image_resolution") && j["image_resolution"].is_array() && j["image_resolution"].size() >= 1) {
                orig_w = j["image_resolution"][0].get<double>();
            }
            else if (j.contains("image_width")) {
                orig_w = j["image_width"].get<double>();
            }
        }
        else {
            // Use OpenCV FileStorage for YAML/XML style files
            cv::FileStorage fs(calib_file, cv::FileStorage::READ);
            if (!fs.isOpened()) {
                throw std::runtime_error("Khong the mo file calibration.");
            }

            // Safely attempt to read camera_matrix and image_resolution
            try {
                fs["camera_matrix"] >> K;
            }
            catch (const cv::Exception&) {
                // If operator[] asserts because the root is not a map, treat as missing
                K = cv::Mat();
            }

            cv::FileNode res_node = fs["image_resolution"];
            if (!res_node.empty() && res_node.isSeq()) {
                orig_w = (double)res_node[0];
            }
        }

        if (K.empty()) {
            throw std::runtime_error("Khong tim thay camera_matrix trong file.");
        }

        // Tính tỉ lệ resize
        double scale_ratio = current_width / orig_w;

        // K là ma trận 3x3 kiểu double (CV_64F)
        fx = K.at<double>(0, 0) * scale_ratio;
        fy = K.at<double>(1, 1) * scale_ratio;
        loaded = true;

        std::cout << "[HeightEstimator] Focal Length loaded: fx=" << fx << ", fy=" << fy << std::endl;
        return true;

    }
    catch (const std::exception& e) {
        std::cerr << "[HeightEstimator] ERR: " << e.what() << std::endl;
        // Fallback
        fx = current_width;
        fy = current_width;
        loaded = false; // Đánh dấu false như logic gốc
        return false;
    }
}

std::pair<double, double> HeightEstimator::calculate(
    const cv::Point2f& head_pt,
    const cv::Point2f& foot_pt,
    const cv::Mat& homography_matrix,
    const cv::Point2f& cam_real_pos)
{
    if (!loaded || homography_matrix.empty()) {
        return { 0.0, 0.0 };
    }

    // 1. Tính chiều cao trên ảnh (Pixel) dùng khoảng cách Euclidean
    double h_pixel = std::hypot(head_pt.x - foot_pt.x, head_pt.y - foot_pt.y);

    // 2. Tính khoảng cách thực từ Camera tới điểm Chân (Distance D)
    // C++ cần đưa điểm vào std::vector để truyền qua hàm perspectiveTransform
    std::vector<cv::Point2f> foot_arr = { foot_pt };
    std::vector<cv::Point2f> real_pt_arr;

    cv::perspectiveTransform(foot_arr, real_pt_arr, homography_matrix);
    cv::Point2f real_pt = real_pt_arr[0]; // Lấy kết quả đầu tiên (X_real, Y_real)

    // Tính khoảng cách
    double dx = real_pt.x - cam_real_pos.x;
    double dy = real_pt.y - cam_real_pos.y;
    double distance_D = std::sqrt(dx * dx + dy * dy);

    // 3. Áp dụng công thức Sliding Scale: H_real = h_pixel * (D / f)
    if (fy == 0.0) return { 0.0, 0.0 };

    double height_real = h_pixel * (distance_D / fy);
    double TILT_FACTOR = 1.0;

    return { height_real * TILT_FACTOR, distance_D };
}