# RetinaNet Examples

**Fast** and **accurate** single stage object detection with end-to-end GPU optimization.

## Description

[RetinaNet](#references) is a single shot object detector with multiple backbones offering various performance/accuracy trade-offs.

It is optimized for end-to-end GPU processing using:
* The [PyTorch](https://pytorch.org) deep learning framework with [ONNX](https://onnx.ai) support
* NVIDIA [Apex](https://github.com/NVIDIA/apex) for mixed precision and distributed training
* NVIDIA [DALI](https://github.com/NVIDIA/DALI) for optimized data pre-processing
* NVIDIA [TensorRT](https://developer.nvidia.com/tensorrt) for high-performance inference
* NVIDIA [DeepStream](https://developer.nvidia.com/deepstream-sdk) for optimized real-time video streams support

## Disclaimer

This is a research project, not an official NVIDIA product.

## Performance

The detection pipeline allows the user to select a specific backbone depending on the latency-accuracy trade-off preferred.

Backbone | Resize | mAP @[IoU=0.50:0.95] | Training Time on [DGX1v](https://www.nvidia.com/en-us/data-center/dgx-1/) | TensorRT Inference Latency FP16 on [V100](https://www.nvidia.com/en-us/data-center/tesla-v100/) | TensorRT Inference Latency INT8 on [T4](https://www.nvidia.com/en-us/data-center/tesla-t4/)
--- | :---: | :---: | :---: | :---: | :---:
ResNet18FPN | 800 | 0.318 | 5 hrs  | 12 ms/im | 12 ms/im
ResNet34FPN | 800 | 0.343 | 6 hrs  | 14 ms/im | 14 ms/im
ResNet50FPN | 800 | 0.358 | 7 hrs  | 16 ms/im | 16 ms/im
ResNet101FPN | 800 | 0.376 | 10 hrs | 20 ms/im | 20 ms/im
ResNet152FPN | 800 | 0.393 | 12 hrs | 25 ms/im | 24 ms/im

Training results for [COCO 2017](http://cocodataset.org/#detection-2017) (train/val) after full training schedule with default parameters. Inference results include bounding boxes post-processing for a batch size of 1.

## Installation

For best performance, we encourage using the latest [PyTorch NGC docker container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch):
```bash
nvidia-docker run --rm --ipc=host -it nvcr.io/nvidia/pytorch:19.05-py3
```

From the container, simply install retinanet using `pip`:
```bash
pip install --no-cache-dir git+https://github.com/nvidia/retinanet-examples
```

Or you can clone this repository, build and run your own image:
```bash
git clone https://github.com/nvidia/retinanet-examples
docker build -t retinanet:latest retinanet/
nvidia-docker run --rm --ipc=host -it retinanet:latest
```

## Usage

Training, inference, evaluation and model export can be done through the `retinanet` utility.

For more details refer to the [INFERENCE](INFERENCE.md) and [TRAINING](TRAINING.md) documentation.

### Training

Train a detection model on [COCO 2017](http://cocodataset.org/#download) from pre-trained backbone:
```bash
retinanet train retinanet_rn50fpn.pth --backbone ResNet50FPN \
    --images /coco/images/train2017/ --annotations /coco/annotations/instances_train2017.json \
    --val-images /coco/images/val2017/ --val-annotations /coco/annotations/instances_val2017.json
```

### Fine Tuning

Fine-tune a pre-trained model on your dataset. In the example below we use [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html) with [JSON annotations](https://storage.googleapis.com/coco-dataset/external/PASCAL_VOC.zip):
```bash
retinanet train model_mydataset.pth \
    --fine-tune retinanet_rn50fpn.pth \
    --classes 20 --iters 10000 --val-iters 1000 --lr 0.0005 \
    --resize 512 --jitter 480 640 --images /voc/JPEGImages/ \
    --annotations /voc/pascal_train2012.json --val-annotations /voc/pascal_val2012.json
```

Note: the shorter side of the input images will be resized to `resize` as long as the longer side doesn't get larger than `max-size`. During training, the images will be randomly randomly resized to a new size within the `jitter` range.

### Inference

Evaluate your detection model on [COCO 2017](http://cocodataset.org/#download):
```bash
retinanet infer retinanet_rn50fpn.pth --images /coco/images/val2017/ --annotations /coco/annotations/instances_val2017.json
```

Run inference on [your dataset](#datasets):
```bash
retinanet infer retinanet_rn50fpn.pth --images /dataset/val --output detections.json
```

### Optimized Inference with TensorRT

For faster inference, export the detection model to an optimized FP16 TensorRT engine:
```bash
retinanet export model.pth engine.plan
```
Note: for older versions of TensorRT (prior to TensorRT 5.1 / 19.03 containers) the ONNX opset version should be specified (using `--opset 8` for instance).

Evaluate the model with TensorRT backend on [COCO 2017](http://cocodataset.org/#download):
```bash
retinanet infer engine.plan --images /coco/images/val2017/ --annotations /coco/annotations/instances_val2017.json
```

### INT8 Inference with TensorRT

For even faster inference, do INT8 calibration to create an optimized INT8 TensorRT engine:
```bash
retinanet export model.pth engine.plan --int8 --calibration-images /coco/images/val2017/
```
This will create an INT8CalibrationTable file that can be used to create INT8 TensorRT engines for the same model later on without needing to do calibration.

Or create an optimized INT8 TensorRT engine using a cached calibration table:
```bash
retinanet export model.pth engine.plan --int8 --calibration-table /path/to/INT8CalibrationTable
```

## Datasets

RetinaNet supports annotations in the [COCO JSON format](http://cocodataset.org/#format-data).
When converting the annotations from your own dataset into JSON, the following entries are required:
```
{
    "images": [{
        "id" : int,
        "file_name" : str
    }],
    "annotations": [{
        "id" : int,
        "image_id" : int, 
        "category_id" : int,
        "bbox" : [x, y, w, h]
    }],
    "categories": [{
        "id" : int
    ]}
}
```

## References

- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002).
  Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár.
  ICCV, 2017.
- [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677).
  Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, Kaiming He.
  June 2017.
- [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144).
  Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie.
  CVPR, 2017.
- [Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385).
  Kaiming He, Xiangyu Zhang, Shaoqing Renm Jian Sun.
  CVPR, 2016.
