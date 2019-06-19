# Deploying RetinaNet in DeepStream 3.0

This shows how to export a trained RetinaNet model to TensorRT and deploy it in a video analytics application using NVIDIA DeepStream 3.0.

## Prerequisites
* A GPU supported by DeepStream: Jetson Xavier, Tesla P4/P40/V100/T4
* Download DeepStream 3.0 SDK from the NVIDIA [DeepStream 3.0](https://developer.nvidia.com/deepstream-sdk) website.
    * **Note**: DeepStream 3.0 requires CUDA 10, cuDNN 7.3, and TensorRT 5.0. 
* A trained PyTorch RetinaNet model.
* One or more video files in .mp4 format.


## Setup instructions for Tesla GPUs
1.) First, ensure that DeepStream 3.0 SDK for Tesla is downloaded to this directory: `DeepStreamSDK-Tesla-v3.0.tbz2`

2.) Unpack the DeepStream 3.0 SDK for Tesla:
```
tar -xvf DeepStreamSDK-Tesla-v3.0-tbz2 -C DeepStream_Release/
```

3.) We strongly recommend using the provided Dockerfile to configure the environment for Tesla GPUs.
```
docker build -f <your_path>/retinanet-examples/Dockerfile.deepstream -t ds_retinanet:latest <your_path>/retinanet-examples
docker run --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video -it \
    --rm --ipc=host -v <dir containing your data>:/data ds_retinanet:latest
```

4.) Finally, export your trained PyTorch RetinaNet model to TensorRT per the [INFERENCE](https://github.com/NVIDIA/retinanet-examples/blob/master/INFERENCE.md) instructions:
```
retinanet export <PyTorch model> <engine> --opset 8 --batch n

OR

retinanet export <PyTorch model> <engine> --opset 8 --int8 --calibration-images <example images> --batch n
```

## Setup instructions for Jetson Xavier
1.) First, flash Jetson Xavier with [Jetpack 4.1.1](https://developer.nvidia.com/embedded/jetpack-4-1-1)

2.) Install additional DeepStream dependencies:
```
sudo apt install \
    libssl1.0.0 \
    libgstreamer1.0-0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libgstrtspserver-1.0-0 \
    libjansson4=2.11-1
    librdkafka1=0.11.3-1build1
```
3.) Download DeepStream 3.0 SDK for Jetson Xavier onto your device: `DeepStreamSDK-Jetson-3.0_EA_beta5.0.tbz2`

4.) Unpack DeepStream 3.0 SDK for Jetson Xavier:
```
tar -xpvf DeepStreamSDK-Jetson-3.0_EA_beta5.0.tbz2

cd deepstream_sdk_on_jetson/ && \
   sudo tar -xvf binaries.tbz2 -C / && \ 
   sudo ldconfig
```

5.) Export trained PyTorch RetinaNet model to ONNX on your host system (i.e. not on your Jetson), specifying (`--opset 8`):
```
   retinanet export model.pth model.onnx --opset 8
```

5.) Copy ONNX RetinaNet model to Jetson Xavier, and export to TensorRT using the [cppapi sample code](https://github.com/NVIDIA/retinanet-examples/tree/master/extras/cppapi#running)


## Preparing the DeepStream config file:
We have included two example DeepStream config files in `deepstream-sample`.
- `ds_config_1vids.txt`: Performs detection on a single video, using the detector specified by `infer_config_batch1.txt`.
- `ds_config_8vids.txt`: Performs detection on multiple video streams simultaneously, using the detector specified by `infer_config_batch8.txt`. Frames from each video are combined into a single batch and passed to the detector for inference.

The `ds_config_*` files are DeepStream config files. They describe the overall processing. `infer_config_*` files define the individual detectors, which can be chained in series.

Before they can be used, these config files must be modified to specify the correct paths to the input and output videos files, and the TensorRT engines.

**Input files** are specified in the deepstream config files by the `uri=file://<path>` parameter.

**Output files** are specified in the deepstream config files by the `output-file=<path>` parameter.

**TensorRT engines** are specified in both the DeepStream config files, and also the detector config files, by the `model-engine-file=<path>` parameters. 

On Xavier, you can optionally set `enable=1` to `[sink1]` in `ds_config_*` files to display the processed video stream.

## Run deepstream-app
Once all of the config files have been modified, launch the DeepStream application: 
```
LD_PRELOAD=<your_local_path>/retinanet-examples/extras/deepstream/build/libnvdsparsebbox_retinanet.so \
    deepstream-app -c <config file>
```

## Convert output video file to mp4
You can convert the outputted `.mkv` file to `.mp4` using `ffmpeg`.
```
ffmpeg -i /data/output/file1.mkv -c copy /data/output/file2.mp4
```
