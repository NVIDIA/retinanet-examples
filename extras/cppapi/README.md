# RetinaNet C++ Inference API - Sample Code

The C++ API allows you to build a TensorRT engine for inference using the ONNX export of a core model.

The following shows how to build and run code samples for exporting an ONNX core model (from RetinaNet or other toolkit supporting the same sort of core model structure) to a TensorRT engine and doing inference on images.

## Building

Building the example requires the following toolkits and libraries to be set up properly on your system:
* A proper C++ toolchain (MSVS on Windows)
* [CMake](https://cmake.org/download/) version 3.9 or later
* NVIDIA [CUDA](https://developer.nvidia.com/cuda-toolkit)
* NVIDIA [CuDNN](https://developer.nvidia.com/cudnn)
* NVIDIA [TensorRT](https://developer.nvidia.com/tensorrt)
* [OpenCV](https://opencv.org/releases.html)

### Linux
```bash
mkdir build && cd build
cmake -DCMAKE_CUDA_FLAGS="--expt-extended-lambda -std=c++14" ..
make
```

### Windows
```bash
mkdir build && cd build
cmake -G "Visual Studio 15 2017" -A x64 -T host=x64,cuda=10.0 -DTensorRT_DIR="C:\path\to\tensorrt" -DOpenCV_DIR="C:\path\to\opencv\build" ..
msbuild odtk_infer.sln
```

## Running

If you don't have an ONNX core model, generate one from your RetinaNet model:
```bash
odtk export model.pth model.onnx
```

Load the ONNX core model and export it to a RetinaNet TensorRT engine (using FP16 precision):
```bash
export{.exe} model.onnx engine.plan
```

You can also export the ONNX core model to an INT8 TensorRT engine if you have already done INT8 calibration:
```bash
export{.exe} model.onnx engine.plan INT8CalibrationTable
```

Run a test inference (default output if none provided: "detections.png"):
```bash
infer{.exe} engine.plan image.jpg [<OUTPUT>.png]
```

Note: make sure the TensorRT, CuDNN and OpenCV libraries are available in your environment and path.

We have verified these steps with the following configurations:
* DGX-1V using the provided Docker container (CUDA 10, cuDNN 7.4.2, TensorRT 5.0.2, OpenCV 3.4.3)
* Jetson AGX Xavier with JetPack 4.1.1 Developer Preview (CUDA 10, cuDNN 7.3.1, TensorRT 5.0.3, OpenCV 3.3.1)




