# Inference

We provide two ways inferring using `odtk`:
* PyTorch inference using a trained model (FP32 or FP16 precision)
* Export trained pytorch model to TensorRT for optimized inference (FP32, FP16 or INT8 precision)

`odtk infer` will run distributed inference across all available GPUs. When using PyTorch, the default behavior is to run inference with mixed precision. The precision used when running inference with a TensorRT engine will correspond to the precision chosen when the model was exported to TensorRT (see [TensorRT section](#exporting-trained-pytorch-model-to-tensorrt) below). 

**NOTE**: Availability of HW support for fast FP16 and INT8 precision like [NVIDIA Tensor Cores](https://www.nvidia.com/en-us/data-center/tensorcore/) depends on your GPU architecture: Volta or newer GPUs support both FP16 and INT8, and Pascal GPUs can support either FP16 or INT8. 

## PyTorch Inference

Evaluate trained PyTorch detection model on COCO 2017 (mixed precision):

```bash
odtk infer model.pth --images=/data/coco/val2017 --annotations=instances_val2017.json --batch 8
```
**NOTE**: `--batch N` specifies *global* batch size to be used for inference. The batch size per GPU will be `N // num_gpus`.

Use full precision (FP32) during evaluation:

```bash
odtk infer model.pth --images=/data/coco/val2017 --annotations=instances_val2017.json --full-precision
```

Evaluate PyTorch detection model with a small input image size:

```bash
odtk infer model.pth --images=/data/coco/val2017 --annotations=instances_val2017.json  --resize 400 --max-size 640
```
Here, the shorter side of the input images will be resized to `resize` as long as the longer side doesn't get larger than `max-size`, otherwise the longer side of the input image will be resized to `max-size`.

**NOTE**: To get best accuracy, training the model at the preferred export size is encouraged.

Run inference using your own dataset:

```bash
odtk infer model.pth --images=/data/your_images --output=detections.json
```

## Exporting trained PyTorch model to TensorRT

`odtk` provides an simple workflow to optimize a trained PyTorch model for inference deployment using TensorRT. The PyTorch model is exported to [ONNX](https://github.com/onnx/onnx), and then the ONNX model is consumed and optimized by TensorRT.
To learn more about TensorRT optimization, refer here: https://developer.nvidia.com/tensorrt

**NOTE**: When a model is optimized with TensorRT, the output is a TensorRT engine (.plan file) that can be used for deployment. This TensorRT engine has several fixed properties that are specified during the export process.
* Input image size: TensorRT engines only support a fixed input size.
* Precision: TensorRT supports FP32, FP16, or INT8 precision.
* Target GPU: TensorRT optimizations are tied to the type of GPU on the system where optimization is performed. They are not transferable across different types of GPUs. Put another way, if you aim to deploy your TensorRT engine on a Tesla T4 GPU, you must run the optimization on a system with a T4 GPU. 

The workflow for exporting a trained PyTorch detection model to TensorRT is as simple as:

```bash
odtk export model.pth model_fp16.plan --size 1280
```
This will create a TensorRT engine optimized for batch size 1, using an input size of 1280x1280. By default, the engine will be created to run in FP16 precision.

Export your model to use full precision using a non-square input size:
```bash
odtk export model.pth model_fp32.plan --full-precision --size 800 1280
```

In order to use INT8 precision with TensorRT, you need to provide calibration images (images that are representative of what will be seen at runtime) that will be used to rescale the network.
```bash
odtk export model.pth model_int8.plan --int8 --calibration-images /data/val/ --calibration-batches 10 --calibration-table model_calibration_table
```

This will randomly select 20 images from `/data/val/` to calibrate the network for INT8 precision. The results from calibration will be saved to `model_calibration_table` that can be used to create subsequent INT8 engines for this model without needed to recalibrate. 

Build an INT8 engine for a previously calibrated model:
```bash
odtk export model.pth model_int8.plan --int8 --calibration-table model_calibration_table
```

## Deployment with TensorRT on NVIDIA Jetson AGX Xavier

We provide a path for deploying trained models with TensorRT onto embedded platforms like [NVIDIA Jetson AGX Xavier](https://developer.nvidia.com/embedded/buy/jetson-agx-xavier-devkit), where PyTorch is not readily available. 

You will need to export your trained PyTorch model to ONNX representation on your host system, and copy the resulting ONNX model to your Jetson AGX Xavier:
```bash
odtk export model.pth model.onnx --size 800 1280
```

Refer to additional documentation on using the example cppapi code to build the TensorRT engine and run inference here: [cppapi example code](extras/cppapi/README.md)

## Rotated detections

*Rotated ODTK* allows users to train and infer rotated bounding boxes in imagery. 

### Inference

An example command:
```
odtk infer model.pth --images /data/val --annotations /data/val_rotated.json --output /data/detections.json \ 
    --resize 768 --rotated-bbox
```

### Export

Rotated bounding box models can be exported to create TensorRT engines by using the axis aligned command with the addition of `--rotated-bbox`.