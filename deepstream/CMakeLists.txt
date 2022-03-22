cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(retinaface LANGUAGES CXX)
enable_language(CXX)
include(FindCUDA)
set(CMAKE_CXX_STANDARD 14)
find_package(CUDA REQUIRED)
enable_language(CUDA)
find_package(OpenCV REQUIRED)
find_package(PkgConfig)
pkg_check_modules(GST REQUIRED gstreamer-1.0>=1.4
                               gstreamer-sdp-1.0>=1.4
                               gstreamer-video-1.0>=1.4
                               gstreamer-app-1.0>=1.4)

if(DEFINED TensorRT_DIR)
  include_directories("${TensorRT_DIR}/include")
  link_directories("${TensorRT_DIR}/lib")
endif(DEFINED TensorRT_DIR)
if(DEFINED DeepStream_DIR)
include_directories("${DeepStream_DIR}/sources/includes")
  link_directories("${DeepStream_DIR}/lib")
endif(DEFINED DeepStream_DIR)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${GST_INCLUDE_DIRS})

message(STATUS "====================${PLATFORM_TEGRA}")

add_library(nvdsparsebbox_yolo SHARED
../cpp/csrc/cuda/decode_iou.h
../cpp/csrc/cuda/decode_iou.cu
../cpp/csrc/cuda/nms_iou.h
../cpp/csrc/cuda/nms_iou.cu
../cpp/csrc/cuda/utils.h
../cpp/engine.cpp
  nvdsparsebbox_yolo.cpp
)
set_target_properties(nvdsparsebbox_yolo PROPERTIES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(nvdsparsebbox_yolo PUBLIC nvinfer nvinfer_plugin cuda ${CUDA_LIBRARIES})

add_executable(detect detect.cpp )
target_link_libraries(detect PRIVATE nvdsparsebbox_yolo cuda ${CUDA_LIBRARIES} nvinfer nvinfer_plugin nvonnxparser ${GST_LIBRARIES} nvdsgst_meta nvds_meta nvbufsurface)

if(DEFINED PLATFORM_TEGRA)
  add_definitions(-DPLATFORM_TEGRA=1)
endif(DEFINED PLATFORM_TEGRA)
#cmake -DDeepStream_DIR=/opt/nvidia/deepstream/deepstream-6.0 \
#-DCMAKE_CUDA_FLAGS="--expt-extended-lambda -std=c++14" \
#-DPLATFORM_TEGRA=ON ..

