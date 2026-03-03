#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <queue>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/tracking.hpp>

// Giả định bạn đã có file này từ project chuyển đổi trước đó
#include "height_estimator.h" 

// Nếu dùng nlohmann json
#include "../config_file/json.hpp"
using json = nlohmann::json;

// ================= CẤU HÌNH =================
const std::string VIDEO_PATH = "./vid_test/cam_2.mp4";
const std::string CALIB_FILE = "./config_file/calibration.json";
const std::string CONFIG_FILE = "./config_file/config.json";
const int TARGET_W = 1200;
const int YOLO_SKIP_FRAMES = 2;

// ================= THREAD-SAFE QUEUE =================
template <typename T>
class SafeQueue {
private:
    std::queue<T> q;
    std::mutex mtx;
    std::condition_variable cv;
    size_t max_size;

public:
    SafeQueue(size_t size = 1) : max_size(size) {}

    bool push(const T& item) {
        std::unique_lock<std::mutex> lock(mtx);
        if (q.size() >= max_size) return false;
        q.push(item);
        cv.notify_one();
        return true;
    }

    bool pop(T& item, int timeout_ms = 0) {
        std::unique_lock<std::mutex> lock(mtx);
        if (timeout_ms > 0) {
            if (!cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] { return !q.empty(); })) {
                return false;
            }
        }
        else {
            if (q.empty()) return false;
        }
        item = q.front();
        q.pop();
        return true;
    }

    void clear() {
        std::unique_lock<std::mutex> lock(mtx);
        std::queue<T> empty;
        std::swap(q, empty);
    }

    bool empty() {
        std::unique_lock<std::mutex> lock(mtx);
        return q.empty();
    }
};

// ================= PROFILER =================
class LatencyProfiler {
private:
    std::map<std::string, std::vector<double>> records;
    std::mutex mtx;

public:
    void update(const std::string& name, double elapsed_ms);
    void print_report();
};

// ================= VIDEO STREAM =================
class VideoStream {
private:
    cv::VideoCapture stream;
    cv::Mat current_frame;
    std::atomic<bool> stopped{ false };
    std::atomic<bool> ret{ false };
    double fps;
    std::thread worker_thread;
    std::mutex frame_mtx;

public:
    VideoStream(const std::string& src);
    ~VideoStream();
    void start();
    void update();
    bool read(cv::Mat& frame);
    void stop();
    void reset();
    cv::VideoCapture& getStream() { return stream; }
};

// ================= CẤU TRÚC KẾT QUẢ YOLO =================
struct YoloResult {
    cv::Rect box;
    cv::Point head;
    cv::Point foot;
};

// ================= YOLO WORKER =================
class YoloWorker {
private:
    LatencyProfiler* profiler;
    std::string model_path;
    cv::dnn::Net net;
    std::atomic<bool> stopped{ false };
    std::thread worker_thread;

    SafeQueue<cv::Mat> input_queue{ 1 };
    SafeQueue<std::vector<YoloResult>> output_queue{ 1 };

public:
    YoloWorker(const std::string& path, LatencyProfiler* prof);
    ~YoloWorker();

    void put_frame(const cv::Mat& frame);
    bool get_results(std::vector<YoloResult>& results);
    void run();
    void stop();
};

// ================= MAIN APP =================
struct DetectedObject {
    cv::Rect box;
    cv::Point head;
    cv::Point foot;
    double h_real = 0.0;
    double d_to_target = 0.0;
};

class VideoDistanceApp {
public:
    LatencyProfiler profiler;
    std::string mode = "DISTANCE";
    bool paused = false;
    int frame_count = 0;

    cv::Mat gray_anchor;
    std::vector<cv::Point2f> p0_anchor;
    std::vector<cv::Point2f> roi_points_initial;
    std::vector<cv::Point2f> roi_points_curr;
    std::vector<cv::Point2f> target_point_initial;
    std::vector<cv::Point2f> target_point_curr;
    std::vector<cv::Rect> last_known_boxes;
    cv::Ptr<cv::Tracker> target_tracker;

    HeightEstimator height_tool;
    std::unique_ptr<YoloWorker> yolo_worker;

    std::map<std::string, double> real_world;
    cv::Point2f cam_real_pos{ 0.5f, -18.0f };
    cv::Mat matrix_homography;
    double scale_px_per_meter = 1.0;
    cv::Mat map1, map2;

    cv::Mat current_frame;
    std::vector<DetectedObject> detected_objects;
    double prev_time = 0.0;
    double current_fps = 0.0;

    VideoDistanceApp();
    void init_calibration_maps(cv::Size original_size);
    bool load_config();
    std::vector<cv::Point2f> get_quadrilateral_coords(double l1, double l2, double l3, double l4, double d13);
    void compute_homography(const std::vector<cv::Point2f>& current_pts_array);
    double calculate_distance_points(cv::Point2f p1, cv::Point2f p2);

    void init_anchor(const cv::Mat& gray_frame);
    void update_stabilizer(const cv::Mat& gray_curr);
    cv::Point get_current_target_tuple();
    cv::Rect update_target_tracker(const cv::Mat& frame);

    void process_async_detections();
    void process_frame(const cv::Mat& raw_frame, cv::Mat& out_frame, cv::Rect& out_target_box);
    void draw_overlays(cv::Mat& img, const cv::Rect& target_box);
};