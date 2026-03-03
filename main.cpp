#include "main.h"
#include <fstream>
#include <numeric>
#include <algorithm>
#include <cmath>

// ================= PROFILER =================
void LatencyProfiler::update(const std::string& name, double elapsed_ms) {
    std::lock_guard<std::mutex> lock(mtx);
    records[name].push_back(elapsed_ms);
}

void LatencyProfiler::print_report() {
    std::cout << "\n=============================================\n";
    std::cout << "MULTI-THREAD VIDEO STABILIZER REPORT (Unit: ms)\n";
    std::cout << "=============================================\n";
    std::lock_guard<std::mutex> lock(mtx);
    for (const auto& pair : records) {
        if (pair.second.empty()) continue;
        auto vals = pair.second;
        if (vals.size() > 5) vals.erase(vals.begin(), vals.begin() + 5);

        double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
        double mean = sum / vals.size();
        double min_v = *std::min_element(vals.begin(), vals.end());
        double max_v = *std::max_element(vals.begin(), vals.end());

        std::sort(vals.begin(), vals.end());
        double p99 = vals[static_cast<int>(vals.size() * 0.99)];

        printf("%-15s | Mean: %-6.2f | Min: %-6.2f | Max: %-6.2f | P99: %-6.2f\n",
            pair.first.c_str(), mean, min_v, max_v, p99);
    }
    std::cout << "=============================================\n\n";
}

// ================= VIDEO STREAM =================
VideoStream::VideoStream(const std::string& src) : stream(src) {
    stream.read(current_frame);
    fps = stream.get(cv::CAP_PROP_FPS);
    if (fps <= 0 || std::isnan(fps)) fps = 30;
}

VideoStream::~VideoStream() { stop(); }

void VideoStream::start() {
    worker_thread = std::thread(&VideoStream::update, this);
}

void VideoStream::update() {
    double delay = 1.0 / fps;
    while (!stopped) {
        auto start_time = std::chrono::high_resolution_clock::now();
        cv::Mat frame;
        bool grabbed = stream.read(frame);

        if (!grabbed) {
            ret = false;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(frame_mtx);
            current_frame = frame.clone();
            ret = true;
        }

        auto elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
        double time_to_wait = delay - elapsed;
        if (time_to_wait > 0) {
            std::this_thread::sleep_for(std::chrono::duration<double>(time_to_wait));
        }
    }
}

bool VideoStream::read(cv::Mat& frame) {
    if (!ret) return false;
    std::lock_guard<std::mutex> lock(frame_mtx);
    frame = current_frame.clone();
    return true;
}

void VideoStream::stop() {
    stopped = true;
    if (worker_thread.joinable()) worker_thread.join();
    stream.release();
}

void VideoStream::reset() {
    stream.set(cv::CAP_PROP_POS_FRAMES, 0);
}

// ================= YOLO WORKER =================
YoloWorker::YoloWorker(const std::string& path, LatencyProfiler* prof) : model_path(path), profiler(prof) {
    try {
        net = cv::dnn::readNetFromONNX(model_path);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        std::cout << "[INFO] YOLO Worker loaded." << std::endl;
    }
    catch (const cv::Exception& e) {
        std::cerr << "[ERROR] Could not load YOLO model: " << e.what() << std::endl;
    }
    worker_thread = std::thread(&YoloWorker::run, this);
}

YoloWorker::~YoloWorker() { stop(); }

void YoloWorker::put_frame(const cv::Mat& frame) {
    input_queue.push(frame);
}

bool YoloWorker::get_results(std::vector<YoloResult>& results) {
    return output_queue.pop(results);
}

void YoloWorker::run() {
    while (!stopped) {
        cv::Mat frame;
        if (input_queue.pop(frame, 100)) {
            auto t0 = std::chrono::high_resolution_clock::now();

            // YOLO Inference
            cv::Mat blob;
            cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);
            net.setInput(blob);
            std::vector<cv::Mat> outputs;
            net.forward(outputs, net.getUnconnectedOutLayersNames());

            // Post-processing output shape [1, 56, 8400]
            cv::Mat output = outputs[0];
            int dimensions = output.size[1];
            int rows = output.size[2];

            // Transpose matrix sang [8400, 56] để duyệt dễ hơn
            cv::Mat output_T(rows, dimensions, CV_32F);
            for (int i = 0; i < dimensions; ++i) {
                for (int j = 0; j < rows; ++j) {
                    output_T.at<float>(j, i) = output.at<float>(0, i, j);
                }
            }

            std::vector<cv::Rect> boxes;
            std::vector<float> confidences;
            std::vector<std::vector<float>> keypoints_list;

            float x_scale = (float)frame.cols / 640.0f;
            float y_scale = (float)frame.rows / 640.0f;

            for (int i = 0; i < rows; ++i) {
                float* data = output_T.ptr<float>(i);
                float confidence = data[4]; // objectness / score
                if (confidence >= 0.5f) {
                    float cx = data[0] * x_scale;
                    float cy = data[1] * y_scale;
                    float w = data[2] * x_scale;
                    float h = data[3] * y_scale;
                    int left = int(cx - 0.5 * w);
                    int top = int(cy - 0.5 * h);

                    boxes.push_back(cv::Rect(left, top, int(w), int(h)));
                    confidences.push_back(confidence);

                    std::vector<float> kpts(51);
                    for (int k = 0; k < 51; ++k) {
                        if (k % 3 == 0) kpts[k] = data[5 + k] * x_scale;
                        else if (k % 3 == 1) kpts[k] = data[5 + k] * y_scale;
                        else kpts[k] = data[5 + k];
                    }
                    keypoints_list.push_back(kpts);
                }
            }

            // NMS
            std::vector<int> indices;
            cv::dnn::NMSBoxes(boxes, confidences, 0.5f, 0.4f, indices);

            std::vector<YoloResult> parsed_results;
            for (int idx : indices) {
                YoloResult res;
                res.box = boxes[idx];
                auto& kpts = keypoints_list[idx];

                cv::Point head_pt(res.box.x + res.box.width / 2, res.box.y);
                cv::Point foot_pt(res.box.x + res.box.width / 2, res.box.y + res.box.height);

                // Index 15 (x:45, y:46, c:47), Index 16 (x:48, y:49, c:50)
                if (kpts[47] > 0.5 && kpts[50] > 0.5) {
                    foot_pt = cv::Point((kpts[45] + kpts[48]) / 2, (kpts[46] + kpts[49]) / 2);
                }
                else if (kpts[47] > 0.5) {
                    foot_pt = cv::Point(kpts[45], kpts[46]);
                }
                else if (kpts[50] > 0.5) {
                    foot_pt = cv::Point(kpts[48], kpts[49]);
                }

                // Index 0 (x:0, y:1, c:2), Index 1 (x:3, y:4, c:5), Index 2 (x:6, y:7, c:8)
                if (kpts[2] > 0.5) {
                    head_pt = cv::Point(kpts[0], kpts[1]);
                }
                else if (kpts[5] > 0.5 && kpts[8] > 0.5) {
                    head_pt = cv::Point((kpts[3] + kpts[6]) / 2, (kpts[4] + kpts[7]) / 2);
                }

                res.head = head_pt;
                res.foot = foot_pt;
                parsed_results.push_back(res);
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            profiler->update("YOLO_Infer", std::chrono::duration<double, std::milli>(t1 - t0).count());

            output_queue.clear();
            output_queue.push(parsed_results);
        }
    }
}

void YoloWorker::stop() {
    stopped = true;
    if (worker_thread.joinable()) worker_thread.join();
}

// ================= MAIN APP =================
VideoDistanceApp::VideoDistanceApp() {
    yolo_worker = std::make_unique<YoloWorker>("./weights/yolo11n-pose.onnx", &profiler);
    if (yolo_worker) {
        std::cerr << "Load successfully" << std::endl;
    }
}

void VideoDistanceApp::init_calibration_maps(cv::Size original_size) {
    std::ifstream f(CALIB_FILE);
    if (!f.is_open()) return;
    try {
        json data = json::parse(f);
        cv::Mat K = cv::Mat::zeros(3, 3, CV_64F);
        auto k_arr = data["camera_matrix"];
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) K.at<double>(i, j) = k_arr[i][j];

        cv::Mat D = cv::Mat::zeros(1, 5, CV_64F);
        auto d_arr = data["distortion_coefficients"];
        for (size_t i = 0; i < d_arr.size(); ++i) D.at<double>(0, i) = d_arr[i];

        double scale_factor = (double)TARGET_W / original_size.width;
        int target_h = (int)(original_size.height * scale_factor);
        cv::Size target_size(TARGET_W, target_h);

        if (data.contains("image_resolution")) {
            int calib_w = data["image_resolution"][0];
            int calib_h = data["image_resolution"][1];
            double total_scale_x = ((double)original_size.width / calib_w) * scale_factor;
            double total_scale_y = ((double)original_size.height / calib_h) * scale_factor;
            K.at<double>(0, 0) *= total_scale_x; K.at<double>(1, 1) *= total_scale_y;
            K.at<double>(0, 2) *= total_scale_x; K.at<double>(1, 2) *= total_scale_y;
        }
        else {
            K.at<double>(0, 0) *= scale_factor; K.at<double>(1, 1) *= scale_factor;
            K.at<double>(0, 2) *= scale_factor; K.at<double>(1, 2) *= scale_factor;
        }

        cv::Mat new_K = cv::getOptimalNewCameraMatrix(K, D, target_size, 1, target_size);
        cv::initUndistortRectifyMap(K, D, cv::Mat(), new_K, target_size, CV_32FC1, map1, map2);

        height_tool.load_focal_length(CALIB_FILE, TARGET_W); // Bỏ comment nếu class HeightEstimator hỗ trợ
    }
    catch (const std::exception& e) {
        std::cerr << "Error Init Calib: " << e.what() << std::endl;
    }
}

bool VideoDistanceApp::load_config() {
    std::ifstream f(CONFIG_FILE);
    if (!f.is_open()) return false;
    try {
        json data = json::parse(f);
        real_world = data["real_world"].get<std::map<std::string, double>>();
        cam_real_pos = cv::Point2f(data["camera"]["real_x"], data["camera"]["real_y"]);
        if (data["settings"].contains("scale_px_per_meter")) {
            scale_px_per_meter = data["settings"]["scale_px_per_meter"];
        }

        if (data.contains("points_px")) {
            for (auto& p : data["points_px"]) roi_points_initial.push_back(cv::Point2f(p[0], p[1]));
            roi_points_curr = roi_points_initial;
            compute_homography(roi_points_curr);
        }
        if (data.contains("target_point")) {
            target_point_initial.push_back(cv::Point2f(data["target_point"][0], data["target_point"][1]));
            target_point_curr = target_point_initial;
        }
        return true;
    }
    catch (...) {
        return false;
    }
}

std::vector<cv::Point2f> VideoDistanceApp::get_quadrilateral_coords(double l1, double l2, double l3, double l4, double d13) {
    if (l1 + l2 < d13 || std::abs(l1 - l2) > d13) return {};
    cv::Point2f p1(0.0f, 0.0f);
    cv::Point2f p2((float)l1, 0.0f);
    double cos_alpha = (l1 * l1 + d13 * d13 - l2 * l2) / (2 * l1 * d13);
    cos_alpha = std::max(-1.0, std::min(1.0, cos_alpha));
    double alpha = std::acos(cos_alpha);
    cv::Point2f p3((float)(d13 * std::cos(alpha)), (float)(d13 * std::sin(alpha)));

    double d = d13;
    double a = (l4 * l4 - l3 * l3 + d * d) / (2 * d);
    double h = std::sqrt(std::max(0.0, l4 * l4 - a * a));

    double x0 = p1.x + a * (p3.x - p1.x) / d;
    double y0 = p1.y + a * (p3.y - p1.y) / d;
    double vx = (p3.x - p1.x) / d;
    double vy = (p3.y - p1.y) / d;

    cv::Point2f p4((float)(x0 - h * vy), (float)(y0 + h * vx));
    return { p1, p2, p3, p4 };
}

void VideoDistanceApp::compute_homography(const std::vector<cv::Point2f>& current_pts_array) {
    if (current_pts_array.size() < 4) return;
    auto rw = real_world;
    auto real_coords = get_quadrilateral_coords(rw["L1"], rw["L2"], rw["L3"], rw["L4"], rw["diag_13"]);
    if (real_coords.empty()) return;

    std::vector<cv::Point2f> dst_pts;
    for (auto& pt : real_coords) {
        dst_pts.push_back(cv::Point2f(pt.x * scale_px_per_meter, pt.y * scale_px_per_meter));
    }
    matrix_homography = cv::getPerspectiveTransform(current_pts_array, dst_pts);
}

double VideoDistanceApp::calculate_distance_points(cv::Point2f p1, cv::Point2f p2) {
    if (matrix_homography.empty()) return 0.0;
    std::vector<cv::Point2f> pts = { p1, p2 }, trans_pts;
    cv::perspectiveTransform(pts, trans_pts, matrix_homography);
    return cv::norm(trans_pts[0] - trans_pts[1]) / scale_px_per_meter;
}

void VideoDistanceApp::init_anchor(const cv::Mat& gray_frame) {
    gray_anchor = gray_frame.clone();
    cv::Mat mask = cv::Mat::ones(gray_frame.size(), CV_8UC1) * 255;

    if (!roi_points_initial.empty()) {
        std::vector<cv::Point> pts_int;
        for (auto& p : roi_points_initial) pts_int.push_back(p);
        cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{pts_int}, cv::Scalar(0));
    }

    for (const auto& box : last_known_boxes) cv::rectangle(mask, box, cv::Scalar(0), -1);
    cv::goodFeaturesToTrack(gray_anchor, p0_anchor, 300, 0.01, 10, mask);
}

void VideoDistanceApp::update_stabilizer(const cv::Mat& gray_curr) {
    if (p0_anchor.size() < 10) {
        init_anchor(gray_curr);
        return;
    }

    std::vector<cv::Point2f> p1_anchor;
    std::vector<uchar> status;
    std::vector<float> err;
    cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.01);

    cv::calcOpticalFlowPyrLK(gray_anchor, gray_curr, p0_anchor, p1_anchor, status, err, cv::Size(21, 21), 3, criteria);

    std::vector<cv::Point2f> good_new, good_old;
    for (size_t i = 0; i < p0_anchor.size(); i++) {
        if (status[i] == 1) {
            bool is_dirty = false;
            for (const auto& box : last_known_boxes) {
                cv::Rect expanded_box(box.x - 10, box.y - 10, box.width + 20, box.height + 20);
                if (expanded_box.contains(p1_anchor[i])) { is_dirty = true; break; }
            }
            if (!is_dirty) { good_new.push_back(p1_anchor[i]); good_old.push_back(p0_anchor[i]); }
        }
    }

    if (good_new.size() > 10) {
        cv::Mat M = cv::findHomography(good_old, good_new, cv::RANSAC, 5.0);
        if (!M.empty()) {
            if (!roi_points_initial.empty()) {
                cv::perspectiveTransform(roi_points_initial, roi_points_curr, M);
                compute_homography(roi_points_curr);
            }
            if (!target_point_initial.empty()) cv::perspectiveTransform(target_point_initial, target_point_curr, M);
        }
    }

    if ((double)good_new.size() / (p0_anchor.size() + 1e-5) < 0.3) init_anchor(gray_curr);

    gray_anchor = gray_curr.clone();
    p0_anchor = good_new;
}

cv::Point VideoDistanceApp::get_current_target_tuple() {
    if (!target_point_curr.empty()) return cv::Point(target_point_curr[0].x, target_point_curr[0].y);
    return cv::Point(-1, -1);
}

cv::Rect VideoDistanceApp::update_target_tracker(const cv::Mat& frame) {
    cv::Rect tracked_box;
    if (target_tracker) {
        if (target_tracker->update(frame, tracked_box)) {
            target_point_curr.clear();
            target_point_curr.push_back(cv::Point2f(tracked_box.x + tracked_box.width / 2.0f, tracked_box.y + tracked_box.height / 2.0f));
        }
        else {
            target_tracker.release();
        }
    }
    return tracked_box;
}

void VideoDistanceApp::process_async_detections() {
    std::vector<YoloResult> results;
    if (!yolo_worker->get_results(results)) return;

    detected_objects.clear();
    last_known_boxes.clear();
    cv::Point target_pt = get_current_target_tuple();

    for (const auto& r : results) {
        last_known_boxes.push_back(r.box);
        DetectedObject obj{ r.box, r.head, r.foot, 0.0, 0.0 };

        if (mode == "HEIGHT") {
            obj.h_real = height_tool.calculate(obj.head, obj.foot, matrix_homography, cam_real_pos).first;
        }
        else if (mode == "DISTANCE" && target_pt.x != -1) {
            obj.d_to_target = calculate_distance_points(obj.foot, target_pt);
        }
        detected_objects.push_back(obj);
    }
}

void VideoDistanceApp::process_frame(const cv::Mat& raw_frame, cv::Mat& out_frame, cv::Rect& out_target_box) {
    auto t_start = std::chrono::high_resolution_clock::now();
    double curr_time = cv::getTickCount() / cv::getTickFrequency();
    current_fps = (prev_time > 0) ? 1.0 / (curr_time - prev_time) : 0;
    prev_time = curr_time;

    double scale = (double)TARGET_W / raw_frame.cols;
    int new_h = (int)(raw_frame.rows * scale);
    cv::Mat frame_resized;
    cv::resize(raw_frame, frame_resized, cv::Size(TARGET_W, new_h));

    cv::Mat frame_clean = frame_resized;
    if (!map1.empty() && !map2.empty()) cv::remap(frame_resized, frame_clean, map1, map2, cv::INTER_LINEAR);

    cv::Mat frame_gray;
    cv::cvtColor(frame_clean, frame_gray, cv::COLOR_BGR2GRAY);

    auto t_stab = std::chrono::high_resolution_clock::now();
    if (gray_anchor.empty()) init_anchor(frame_gray);
    else update_stabilizer(frame_gray);
    profiler.update("Stabilizer", std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_stab).count());

    out_target_box = update_target_tracker(frame_clean);

    if (frame_count % YOLO_SKIP_FRAMES == 0) yolo_worker->put_frame(frame_clean.clone());
    process_async_detections();

    profiler.update("Main_Display", std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_start).count());

    out_frame = frame_clean;
    frame_count++;
}

void VideoDistanceApp::draw_overlays(cv::Mat& img, const cv::Rect& target_box) {
    cv::putText(img, "FPS: " + std::to_string((int)current_fps), cv::Point(TARGET_W - 120, 80),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

    if (!roi_points_curr.empty()) {
        std::vector<cv::Point> pts;
        for (auto& p : roi_points_curr) pts.push_back(p);
        cv::polylines(img, pts, true, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }

    cv::Point target_pt = get_current_target_tuple();
    if (mode == "DISTANCE" && target_pt.x != -1) {
        cv::circle(img, target_pt, 5, cv::Scalar(0, 255, 255), -1);
        cv::circle(img, target_pt, 12, cv::Scalar(0, 255, 255), 2);
        if (target_box.width > 0) cv::rectangle(img, target_box, cv::Scalar(0, 255, 255), 1);
    }

    for (const auto& obj : detected_objects) {
        cv::rectangle(img, obj.box, cv::Scalar(100, 100, 100), 1);
        if (mode == "HEIGHT") {
            cv::line(img, obj.head, obj.foot, cv::Scalar(0, 255, 0), 2);
            char buf[32]; snprintf(buf, sizeof(buf), "H: %.2fm", obj.h_real);
            cv::putText(img, buf, cv::Point(obj.box.x, obj.box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        }
        else if (mode == "DISTANCE") {
            cv::circle(img, obj.foot, 4, cv::Scalar(255, 0, 0), -1);
            if (target_pt.x != -1) {
                cv::line(img, obj.foot, target_pt, cv::Scalar(0, 165, 255), 2);
                cv::Point mid((obj.foot.x + target_pt.x) / 2, (obj.foot.y + target_pt.y) / 2);
                char buf[32]; snprintf(buf, sizeof(buf), "%.2fm", obj.d_to_target);
                cv::putText(img, buf, mid, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 165, 255), 2);
            }
        }
    }

    cv::rectangle(img, cv::Rect(0, 0, TARGET_W, 60), cv::Scalar(0, 0, 0), -1);
    cv::putText(img, "MODE: " + mode, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
    cv::putText(img, paused ? "PAUSED" : "PLAYING", cv::Point(350, 40), cv::FONT_HERSHEY_SIMPLEX, 0.6,
        paused ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0), 1);
}

// ================= ENTRY POINT =================
VideoDistanceApp app;

void mouse_event_video(int event, int x, int y, int flags, void* userdata) {
    if (event == cv::EVENT_LBUTTONDOWN && app.mode == "DISTANCE") {
        app.target_point_curr.clear();
        app.target_point_curr.push_back(cv::Point2f(x, y));

        int box_size = 20;
        cv::Rect bbox(std::max(0, x - box_size / 2), std::max(0, y - box_size / 2), box_size, box_size);

        app.target_tracker = cv::TrackerKCF::create();
        app.target_tracker->init(app.current_frame, bbox);
    }
}

int main() {
    VideoStream vs(VIDEO_PATH);
    vs.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    cv::Mat dummy_frame;
    vs.read(dummy_frame);
    if (!dummy_frame.empty()) {
        app.init_calibration_maps(cv::Size(dummy_frame.cols, dummy_frame.rows));
    }
    app.load_config();

    cv::namedWindow("Anchor Tracking");
    cv::setMouseCallback("Anchor Tracking", mouse_event_video);

    while (true) {
        cv::Rect target_box;
        if (!app.paused) {
            cv::Mat frame;
            if (!vs.read(frame)) {
                std::cout << "End of video, resetting..." << std::endl;
                vs.reset();
                app.gray_anchor = cv::Mat();
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                continue;
            }
            app.process_frame(frame, app.current_frame, target_box);
        }

        if (!app.current_frame.empty()) {
            cv::Mat display_img = app.current_frame.clone();
            app.draw_overlays(display_img, target_box);
            cv::imshow("Anchor Tracking", display_img);
        }

        char key = (char)cv::waitKey(1);
        if (key == 'q') {
            app.yolo_worker->stop();
            app.profiler.print_report();
            vs.stop();
            break;
        }
        if (key == ' ') app.paused = !app.paused;
        if (app.paused) {
            if (key == 'h') app.mode = "HEIGHT";
            if (key == 'd') app.mode = "DISTANCE";
        }
    }

    cv::destroyAllWindows();
    return 0;
}