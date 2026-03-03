# DISTANCE CALCULATION BY C++ (CUDA Accelerated)

## 1. Mô tả dự án:
- Ứng dụng C++ tính toán khoảng cách và ước lượng độ cao (distance/height estimation) sử dụng Computer Vision. Dự án được tối ưu hóa để chạy trên GPU NVIDIA thông qua thư viện OpenCV CUDA.
- Tính năng chính: Ước lượng chiều cao/khoảng cách từ hình ảnh/video.
- Mô-đun chính: height_estimator (xử lý logic tính toán).
- Deep Learning: Sử dụng YOLO (ONNX) qua OpenCV DNN module (CUDA backend).
- Environment: Visual Studio 2026, CMake, Windows 10/11.

## 2. Yêu cầu hệ thống (Prerequisites):
### Phần cứng:
- GPU NVIDIA (Test trên GTX 1650 Ti).
- RAM >= 8GB.
### Phần mềm & Thư viện:
- Visual Studio 2026 (MSVC v19.4x/19.5x).
- CMake >= 3.10.
- CUDA Toolkit: v12.4.
- cuDNN: v8.9.7 (Đã merge vào thư mục cài đặt CUDA).
- OpenCV: v4.12.0 + OpenCV Contrib (Custom Build).

## 3. Cấu trúc dự án:
```
distance_cal_cpp/
├── CMakeLists.txt       # Cấu hình build cho dự án
├── main.cpp             # Entrypoint (Điểm vào chương trình)
├── main.h               # Khai báo chung
├── build                # thư mục chứa các file build từ cmake (QUAN TRỌNG)
├──config_file
│   ├── config.json          # Cấu hình tham số chạy (quan trọng)
│   └── calibration.json     # Thông số camera
├── height_estimator/        # Mã nguồn mô-đun chính
│   ├── height_estimator.cpp
│   └── height_estimator.h
├── weights/             # Chứa file model (yolo11n-pose.onnx)
├── vid_test/            # Chứa video test
└── README.md            # Tài liệu hướng dẫn
```

## 4. Hướng dẫn cài đặt môi trường (Quan trọng):
- Do sử dụng Visual Studio 2026 (chưa được CUDA hỗ trợ chính thức), OpenCV cần được build thủ công với các tham số đặc biệt:
        Source: OpenCV 4.12.0 + Contrib.
        CMake Configure Flags:
        WITH_CUDA=ON, WITH_CUDNN=ON, OPENCV_DNN_CUDA=ON
        CUDA_ARCH_BIN=7.5 (Cho GTX 1650 Ti)
        BUILD_opencv_world=ON
        CUDA_NVCC_FLAGS thêm cờ: -allow-unsupported-compiler (Để vượt lỗi check version của VS 2026).
        Installation Path: ./OpenCV_CUDA/build/install

## 5. Hướng dẫn Build & Run:
- Mở Developer PowerShell for VS 2026 tại thư mục gốc dự án và chạy các lệnh sau:

Bước 1: Cấu hình (Configure)
        -> cmake -B build -G "Visual Studio 18 2026" -A x64
        Lưu ý: Nếu CMake không tìm thấy OpenCV, hãy đảm bảo file CMakeLists.txt đã set đường dẫn:
        set(OpenCV_DIR "D:/OpenCV_CUDA/build/install")
        
Bước 2: Biên dịch (Build)
        -> cmake --build build --config Release
        
Bước 3: Setup trước khi chạy (Tránh lỗi Crash)
- Để chương trình chạy được, file .exe cần tìm thấy thư viện động .dll và file config:
Copy DLL: Copy file opencv_world4120.dll từ D:\OpenCV_CUDA\build\install\bin vào thư mục build\Release.
Config & Model: Đảm bảo file config.json và thư mục weights nằm ở vị trí mà code có thể đọc được (Khuyên dùng đường dẫn tuyệt đối trong code C++ để tránh lỗi File Not Found).

Bước 4: Chạy chương trình
        -> .\build\Release\build_app.exe
        
## 6. Các lỗi thường gặp (Troubleshooting):
- Lỗi: Chương trình tắt ngay lập tức (Silent Crash). Nguyên nhân: Thiếu file opencv_world4120.dll cạnh file .exe hoặc chưa add đường dẫn bin vào System PATH. Khắc phục: Copy file .dll vào cùng thư mục với file .exe là nhanh nhất.

Lỗi: Xuất hiện file gcapi.dll, service.conf.lock. Nguyên nhân: NVIDIA Overlay (GeForce Experience) can thiệp vào tiến trình build. Khắc phục: Xóa các file này đi, không ảnh hưởng đến dự án. Tắt Overlay trong setting của NVIDIA nếu muốn.
