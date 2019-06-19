cmake_minimum_required(VERSION 3.5.1)

project(deepstream-retinanet)
enable_language(CXX)
include(FindCUDA)

set(CMAKE_CXX_STANDARD 11)
find_package(CUDA REQUIRED)

if(DEFINED TensorRT_DIR)
  include_directories("${TensorRT_DIR}/include")
  link_directories("${TensorRT_DIR}/lib")
endif(DEFINED TensorRT_DIR)
if(DEFINED DeepStream_DIR)
  include_directories("${DeepStream_DIR}/sources/includes")
endif(DEFINED DeepStream_DIR)
include_directories(${CUDA_INCLUDE_DIRS})

if(NOT DEFINED ARCH)
  set(ARCH "sm_70")
endif(NOT DEFINED ARCH)

cuda_add_library(nvdsparsebbox_retinanet SHARED
  ../../../csrc/cuda/decode.h
  ../../../csrc/cuda/decode.cu
  ../../../csrc/cuda/nms.h
  ../../../csrc/cuda/nms.cu
  ../../../csrc/cuda/utils.h
  ../../../csrc/engine.cpp
  nvdsparsebbox_retinanet.cpp
  OPTIONS -arch ${ARCH} -std=c++11 --expt-extended-lambda
)
target_link_libraries(nvdsparsebbox_retinanet ${CUDA_LIBRARIES} nvinfer nvinfer_plugin nvonnxparser)
