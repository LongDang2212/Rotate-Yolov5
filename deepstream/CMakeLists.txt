cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(ryolo LANGUAGES CXX)
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
include_directories("${PROJECT_SOURCE_DIR}/includes")
  link_directories("${DeepStream_DIR}/lib")
endif(DEFINED DeepStream_DIR)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${GST_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/sort_tracking/include)
include_directories(${DeepStream_DIR}/sources/apps/apps-common/includes)
include_directories(/usr/include/opencv4)


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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp") 
file(GLOB SOURCE_FILES ${PROJECT_SOURCE_DIR}/sort_tracking/src/*.cpp)
target_link_libraries(nvdsparsebbox_yolo PUBLIC nvinfer nvinfer_plugin cuda ${CUDA_LIBRARIES})
file(GLOB DS_SOURCE_FILES ${DeepStream_DIR}/sources/apps/apps-common/src/*.cpp)
add_executable(detect detect.cpp ${SOURCE_FILES} ${DS_SOURCE_FILES})
target_link_libraries(detect PRIVATE nvdsparsebbox_yolo cuda ${CUDA_LIBRARIES} nvinfer nvinfer_plugin nvonnxparser ${GST_LIBRARIES} nvdsgst_meta nvds_meta nvbufsurface  ${OpenCV_LIBS})
if(DEFINED PLATFORM_TEGRA)
  add_definitions(-DPLATFORM_TEGRA=1)
endif(DEFINED PLATFORM_TEGRA)


