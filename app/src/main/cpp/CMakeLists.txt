cmake_minimum_required(VERSION 3.22.1)
project("yolo_jni" C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags for release builds
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

# Build libyuv as static library (disable its executables and unit tests)
set(UNIT_TEST OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/libyuv EXCLUDE_FROM_ALL)

# Our JNI shared library
add_library(yolo_jni SHARED
    yuv_preprocessing.cpp
)

target_include_directories(yolo_jni PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libyuv/include
)

target_link_libraries(yolo_jni
    yuv          # libyuv static library
    android      # Android NDK
    log          # Android logging
   jnigraphics  # For bitmap access if needed
)
