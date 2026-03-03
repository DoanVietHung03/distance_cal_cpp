#ifndef HEIGHT_ESTIMATOR_H
#define HEIGHT_ESTIMATOR_H

#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <string>
#include <utility> // Để dùng std::pair

class HeightEstimator {
private:
    double fx;
    double fy;
    bool loaded;

public:
    // Constructor
    HeightEstimator();

    // Hàm load focal length từ file JSON
    bool load_focal_length(const std::string& calib_file, double current_width);

    // Hàm tính toán chiều cao và khoảng cách
    // Trả về một std::pair chứa: <Chiều_cao_mét, Khoảng_cách_mét>
    std::pair<double, double> calculate(
        const cv::Point2f& head_pt,
        const cv::Point2f& foot_pt,
        const cv::Mat& homography_matrix,
        const cv::Point2f& cam_real_pos
    );
};

#endif // HEIGHT_ESTIMATOR_H