cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(retinanet_infer LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)
find_package(CUDA REQUIRED)
enable_language(CUDA)

if(DEFINED TensorRT_DIR)
  include_directories("${TensorRT_DIR}/include")
  link_directories("${TensorRT_DIR}/lib")
endif(DEFINED TensorRT_DIR)
include_directories(${CUDA_INCLUDE_DIRS})

add_library(retinanet
  ../../csrc/cuda/decode.h
  ../../csrc/cuda/decode.cu
  ../../csrc/cuda/nms.h
  ../../csrc/cuda/nms.cu
  ../../csrc/cuda/utils.h
  ../../csrc/engine.h
  ../../csrc/engine.cpp
  ../../csrc/calibrator.h
)
set_target_properties(retinanet PROPERTIES 
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_link_libraries(retinanet PUBLIC nvinfer nvonnxparser ${OPENCV_LIBS})

find_package(OpenCV REQUIRED)
add_executable(export export.cpp)
target_link_libraries(export PRIVATE retinanet ${OpenCV_LIBS})

find_package(OpenCV REQUIRED)

add_executable(infer infer.cpp)
target_link_libraries(infer PRIVATE retinanet ${OpenCV_LIBS} cuda ${CUDA_LIBRARIES})

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  add_executable(infervideo infervideo.cpp)
  target_link_libraries(infervideo PRIVATE retinanet ${OpenCV_LIBS} cuda ${CUDA_LIBRARIES})
endif()
