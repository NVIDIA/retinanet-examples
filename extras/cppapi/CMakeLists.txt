cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(odtk_infer LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
find_package(CUDA REQUIRED)
enable_language(CUDA)
find_package(OpenCV REQUIRED)

if(DEFINED TensorRT_DIR)
  include_directories("${TensorRT_DIR}/include")
  link_directories("${TensorRT_DIR}/lib")
endif(DEFINED TensorRT_DIR)
include_directories(${CUDA_INCLUDE_DIRS})

add_library(odtk SHARED
  ../../csrc/cuda/decode.h
  ../../csrc/cuda/decode.cu
  ../../csrc/cuda/nms.h
  ../../csrc/cuda/nms.cu
  ../../csrc/cuda/decode_rotate.h
  ../../csrc/cuda/decode_rotate.cu
  ../../csrc/cuda/nms_iou.h
  ../../csrc/cuda/nms_iou.cu
  ../../csrc/cuda/utils.h
  ../../csrc/engine.h
  ../../csrc/engine.cpp
  ../../csrc/calibrator.h
)
set_target_properties(odtk PROPERTIES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_ARCHITECTURES 60 61 70 72 75 80 86
)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(odtk PUBLIC nvinfer nvonnxparser ${OpenCV_LIBS})

add_executable(export export.cpp)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(export PRIVATE odtk ${OpenCV_LIBS})

add_executable(infer infer.cpp)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(infer PRIVATE odtk ${OpenCV_LIBS} cuda ${CUDA_LIBRARIES})

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  add_executable(infervideo infervideo.cpp)
  include_directories(${OpenCV_INCLUDE_DIRS})
  target_link_libraries(infervideo PRIVATE odtk ${OpenCV_LIBS} cuda ${CUDA_LIBRARIES})
endif()
