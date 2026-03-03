cmake_minimum_required(VERSION 3.10)
project(distance_cal_cpp VERSION 0.1.0)

include(CTest)
enable_testing()

# Yêu cầu C++17 trở lên
set(CMAKE_CXX_STANDARD 17)

# -----------------------------------------------------------------------
# [CÁCH SỬA LỖI] KHAI BÁO THỦ CÔNG (MANUAL MAPPING)
# Bỏ qua find_package tự động, trỏ thẳng vào nơi chứa file
# -----------------------------------------------------------------------

# 1. Đường dẫn gốc đến thư mục cài đặt OpenCV
set(OPENCV_ROOT "D:/OpenCV_CUDA/build/install")

# 2. Khai báo đường dẫn đến file .h (Header)
include_directories(${OPENCV_ROOT}/include)

# 3. Khai báo đường dẫn đến file .lib (Library)
link_directories(${OPENCV_ROOT}/lib)
link_directories(${OPENCV_ROOT}/x64/vc16/lib) # Thêm dự phòng nếu nó nằm trong folder con
link_directories(${OPENCV_ROOT}/x64/vc17/lib) # Thêm dự phòng

# -----------------------------------------------------------------------

find_package(Threads REQUIRED)

# Tạo file chạy tên là "build_app"
add_executable(build_app 
    main.cpp
    height_estimator/height_estimator.cpp 
    "main.h")

target_include_directories(build_app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/height_estimator)

# -----------------------------------------------------------------------
# [QUAN TRỌNG] Link trực tiếp file opencv_world
# Tên file có thể là opencv_world4120.lib hoặc opencv_world4120d.lib (Debug)
# -----------------------------------------------------------------------
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_link_libraries(build_app PRIVATE opencv_world4120d Threads::Threads)
else()
    target_link_libraries(build_app PRIVATE opencv_world4120 Threads::Threads)
endif()