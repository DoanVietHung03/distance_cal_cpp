# DISTANCE CALCULATION BY C++

## Mô tả:
Ứng dụng C++ để tính toán khoảng cách/độ cao (distance/height estimation). Dự án dùng CMake để cấu hình và Ninja làm generator theo cấu hình hiện tại. Mô-đun chính liên quan: height_estimator (bao gồm height_estimator.h và height_estimator.cpp). Điểm vào chương trình: main.cpp. Tập tin cấu hình mẫu: config.json.

## Yêu cầu:
•	C++17
•	CMake >= 3.10
•	Ninja (nếu dùng generator Ninja) hoặc một generator tương thích (ví dụ Visual Studio trên Windows)
•	Trình biên dịch tương thích (MSVC/GCC/Clang)

## Cấu trúc dự án (tổng quan):
•	CMakeLists.txt — tập tin cấu hình dự án
•	main.cpp, main.h — entrypoint và khai báo chung
•	height_estimator/ — mã nguồn mô-đun ước lượng độ cao
•	config.json và calibration.json — cấu hình đầu vào

## Hướng dẫn build (ví dụ):
1.	Tạo thư mục build và cấu hình trong VS: cmake -B build -G "Visual Studio 18 2026" -A x64
2.	Biên dịch: cmake --build build --config Release
3.	(Tùy chọn) Chạy binary: .\build\Release\<name>.exe

## Cấu hình:
Chỉnh tham số tại config.json (định dạng JSON). File này chứa các tham số chạy/thuật toán (xem ví dụ trong repo để biết các trường cần thiết).

## Chạy và kiểm thử:
•	Sau khi build, chạy binary với các tham số nếu cần (tuỳ vào cách main.cpp xử lý argv).
•	Nếu cần thêm test, bổ sung test harness và target trong CMakeLists.txt.
