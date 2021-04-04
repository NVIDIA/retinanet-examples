# Deploying RetinaNet in DeepStream 4.0

This shows how to export a trained RetinaNet model to TensorRT and deploy it in a video analytics application using NVIDIA DeepStream 4.0.

## Prerequisites
* A GPU supported by DeepStream: Jetson Xavier, Tesla P4/P40/V100/T4
* A trained PyTorch RetinaNet model.
* A video source, either `.mp4` files or a webcam.

## Tesla GPUs
Setup instructions:

#### 1. Download DeepStream 4.0 
Download DeepStream 4.0 SDK for Tesla "Download .tar" from [https://developer.nvidia.com/deepstream-download](https://developer.nvidia.com/deepstream-download) and place in the `extras/deepstream` directory. 

This file should be called `deepstream_sdk_v4.0.2_x86_64.tbz2`.

#### 2. Unpack DeepStream
You may need to adjust the permissions on the `.tbz2` file before you can extract it. 

```
cd extras/deepstream
mkdir DeepStream_Release
tar -xvf deepstream_sdk_v4.0.2_x86_64.tbz2 -C DeepStream_Release/
```

#### 3. Build and enter the DeepStream docker container
```
docker build -f <your_path>/retinanet-examples/Dockerfile.deepstream -t ds_odtk:latest <your_path>/retinanet-examples
docker run --gpus all -it --rm --ipc=host -v <dir containing your data>:/data ds_odtk:latest
```

#### 4. Export your trained PyTorch RetinaNet model to TensorRT per the [INFERENCE](https://github.com/NVIDIA/retinanet-examples/blob/master/INFERENCE.md) instructions:
```
odtk export <PyTorch model> <engine> --batch n

OR

odtk export <PyTorch model> <engine> --int8 --calibration-images <example images> --batch n
```

#### 5. Run deepstream-app
Once all of the config files have been modified, launch the DeepStream application: 
```
cd /workspace/retinanet-examples/extras/deepstream/deepstream-sample/
LD_PRELOAD=build/libnvdsparsebbox_odtk.so deepstream-app -c <config file>
```

## Jetson AGX Xavier
Setup instructions.

#### 1. Flash Jetson Xavier with [Jetpack 4.3](https://developer.nvidia.com/embedded/jetpack)

**Ensure that you tick the DeepStream box, under Additional SDKs**

#### 2. (on host) Covert PyTorch model to ONNX.

```bash
odtk export model.pth model.onnx
```

#### 3. Copy ONNX RetinaNet model and config files to Jetson Xavier

Use `scp` or a memory card.

#### 4. (on Jetson) Make the C++ API

```bash
cd extras/cppapi
mkdir build && cd build
cmake -DCMAKE_CUDA_FLAGS="--expt-extended-lambda -std=c++14" ..
make
```

#### 5. (on Jetson) Make the RetinaNet plugin

```bash
cd extras/deepstream/deepstream-sample
mkdir build && cd build
cmake -DDeepStream_DIR=/opt/nvidia/deepstream/deepstream-4.0 .. && make -j
```

#### 6. (on Jetson) Build the TensorRT Engine

```bash
cd extras/cppapi/build
./export model.onnx engine.plan
```

#### 7. (on Jetson) Modify the DeepStream config files
As described in the "preparing the DeepStream config file" section below. 

#### 8. (on Jetson) Run deepstream-app
Once all of the config files have been modified, launch the DeepStream application: 
```
cd extras/deepstream/deepstream-sample
LD_PRELOAD=build/libnvdsparsebbox_odtk.so deepstream-app -c <config file>
```

## Preparing the DeepStream config file:
We have included two example DeepStream config files in `deepstream-sample`.
- `ds_config_1vids.txt`: Performs detection on a single video, using the detector specified by `infer_config_batch1.txt`.
- `ds_config_8vids.txt`: Performs detection on multiple video streams simultaneously, using the detector specified by `infer_config_batch8.txt`. Frames from each video are combined into a single batch and passed to the detector for inference.

The `ds_config_*` files are DeepStream config files. They describe the overall processing. `infer_config_*` files define the individual detectors, which can be chained in series.

Before they can be used, these config files must be modified to specify the correct paths to the input and output videos files, and the TensorRT engines.

* **Input files** are specified in the deepstream config files by the `uri=file://<path>` parameter.

* **Output files** are specified in the deepstream config files by the `output-file=<path>` parameter.

* **TensorRT engines** are specified in both the DeepStream config files, and also the detector config files, by the `model-engine-file=<path>` parameters. 

On Xavier, you can optionally set `enable=1` to `[sink1]` in `ds_config_*` files to display the processed video stream.


## Convert output video file to mp4
You can convert the outputted `.mkv` file to `.mp4` using `ffmpeg`.
```
ffmpeg -i /data/output/file1.mkv -c copy /data/output/file2.mp4
```
