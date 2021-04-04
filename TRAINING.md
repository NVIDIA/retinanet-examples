# Training

There are two main ways to train a model with `odtk`:
* Fine-tuning the detection model using a model already trained on a large dataset (like MS-COCO)
* Fully training the detection model from random initialization using a pre-trained backbone (usually ImageNet)

## Fine-tuning

Fine-tuning an existing model trained on COCO allows you to use transfer learning to get a accurate model for your own dataset with minimal training.
When fine-tuning, we re-initialize the last layer of the classification head so the network will re-learn how to map features to classes scores regardless of the number of classes in your own dataset.

You can fine-tune a pre-trained model on your dataset. In the example below we take a model trained on COCO, and then fine-tune using [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html) with [JSON annotations](https://storage.googleapis.com/coco-dataset/external/PASCAL_VOC.zip):
```bash
odtk train model_mydataset.pth \
    --fine-tune retinanet_rn50fpn.pth \
    --classes 20 --iters 10000 --val-iters 1000 --lr 0.0005 \
    --resize 512 --jitter 480 640 --images /voc/JPEGImages/ \
    --annotations /voc/pascal_train2012.json --val-annotations /voc/pascal_val2012.json
```

Even though the COCO model was trained on 80 classes, we can easily use tranfer learning to fine-tune it on the Pascal VOC model representing only 20 classes.

The shorter side of the input images will be resized to `resize` as long as the longer side doesn't get larger than `max-size`.
During training the images will be randomly resized to a new size within the `jitter` range.

We usually want to fine-tune the model with a lower learning rate `lr` than during full training and for less iterations `iters`.

## Full Training

If you do not have a pre-trained model, if your dataset is substantially large, or if you have written your own backbone, then you should fully train the detection model.

Full training usually starts from a pre-trained backbone (automatically downloaded with the current backbones we offer) that has been pre-trained on a classification task with a large dataset like [ImageNet](http://www.image-net.org).
This is especially necessary for backbones using batch normalization as they require large batch sizes during training that cannot be provided when training on the detection task as the input images have to be relatively large.

Train a detection model on [COCO 2017](http://cocodataset.org/#download) from pre-trained backbone:
```bash
odtk train retinanet_rn50fpn.pth --backbone ResNet50FPN \
    --images /coco/images/train2017/ --annotations /coco/annotations/instances_train2017.json \
    --val-images /coco/images/val2017/ --val-annotations /coco/annotations/instances_val2017.json
```

## Training arguments

### Positional arguments
* The only positional argument is the name of the model. This can be a full path, or relative to the current directory.
```bash
odtk train model.pth
```

### Other arguments
The following arguments are available during training:

* `--annotations` (str): Path to COCO style annotations (required).
* `--images` (str): Path to a directory of images (required).
* `--lr` (float): Sets the learning rate. Default: 0.01.
* `--full-precision`: By default we train using mixed precision. Include this argument to instead train in full precision.
* `--warmup` (int): The number of initial iterations during which we want to linearly ramp-up the learning rate to avoid early divergence of the loss. Default: 1000
* `--backbone` (str): Specify one of the supported backbones. Default: `ResNet50FPN`
* `--classes` (int): The number of classes in your dataset. Default: 80
* `--batch` (int): The size of each training batch. Default: 2 x number of GPUs.
* `--max-size` (int): The longest edge of your training image will be resized, so that it is always less than or equal to `max-size`. Default: 1333. 
* `--jitter` (int int): The shortest edge of your training images will be resized to int1 >= shortest edge >= int2, unless the longest edge exceeds `max-size`, in which case the longest edge will be resized to `max-size` and the shortest length will be sized to keep the aspect ratio constant. Default: 640 1024.
* `--resize` (int): During validation inference, the shortest edge of your training images will be resized to int, unless the longest edge exceeds `max-size`, in which case the longest edge will be resized to `max-size` and the shortest length will be sized to keep the aspect ratio constant. Default: 800.
* `--iters` (int): The number of iterations to process. An iteration is the processing (forward and backward pass) of one batch. Number of epochs is (`iters` x `batch`) / `len(data)`. Default: 90000.
* `--milestones` (int int): The learning rate is multiplied by `--gamma` every time it reaches a milestone. Default: 60000 80000.
* `--gamma` (float): The learning rate is multiplied by `--gamma` every time it reaches a milestone. Default: 0.1.
* `--override`: Do not continue training from `model.pth`, instead overwrite it.
* `--val-annotations` (str): Path to COCO style annotations. If supplied, `pycocotools` will be used to give validation mAP.
* `--val-images` (str): Path to directory of validation images.
* `--val-iters` (int): Run inference on the validation set every int iterations.
* `--fine-tune` (str): Fine tune from a model at path str.
* `--with-dali`: Load data using DALI.
* `--augment-rotate`: Randomly rotates the training images by 0&deg;, 90&deg;, 180&deg; or 270&deg;.
* `--augment-brightness` (float): Randomly adjusts brightness of image. The value sets the standard deviation of a Gaussian distribution. The degree of augmentation is selected from this distribution. Default: 0.002
* `--augment-contrast` (float): Randomly adjusts contrast of image. The value sets the standard deviation of a Gaussian distribution. The degree of augmentation is selected from this distribution. Default: 0.002
* `--augment-hue` (float): Randomly adjusts hue of image. The value sets the standard deviation of a Gaussian distribution. The degree of augmentation is selected from this distribution. Default: 0.0002
* `--augment-saturation` (float): Randomly adjusts saturation of image. The value sets the standard deviation of a Gaussian distribution. The degree of augmentation is selected from this distribution. Default: 0.002
* `--regularization-l2` (float): Sets the L2 regularization of the optimizer. Default: 0.0001

You can also monitor the loss and learning rate schedule of the training using TensorBoard bu specifying a `logdir` path.

## Rotated detections

*Rotated ODTK* allows users to train and infer rotated bounding boxes in imagery. 

### Dataset
Annotations need to conform to the COCO standard, with the addition of an angle (radians) in the bounding box (bbox) entry `[xmin, ymin, width, height, **theta**]`. `xmin`, `ymin`, `width` and `height` are in the axis aligned coordinates, ie floats, measured from the top left of the image. `theta` is in radians, measured anti-clockwise from the x-axis. We constrain theta between - \pi/4 and \pi/4.

In order for the validation metrics to calculate, you also need to fill the `segmentation` entry with the coordinates of the corners of your bounding box.

If using the `--rotated-bbox` flag for rotated detections, add an additional float `theta` to the annotations. To get validation scores you also need to fill the `segmentation` section.
```
        "bbox" : [x, y, w, h, theta]    # all floats, where theta is measured in radians anti-clockwise from the x-axis.
        "segmentation" : [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                                        # Required for validation scores.
``` 

### Anchors

As with all single shot detectors, the anchor boxes may need to be adjusted to suit your dataset. You may need to adjust the anchors in `odtk/model.py`

The default anchors are:

```python
self.ratios = [0.5, 1.0, 2.0]
self.scales = [4 * 2**(i/3) for i in range(3)]
self.angles = [-np.pi/6, 0, np.pi/6] 
```

### Training

We recommend reducing your learning rate, for example using `--lr 0.0005`. 

An example training command for training remote sensing imagery. Note that `--augment-rotate` has been used to randomly rotated the imagery during training.
```
odtk train model.pth --images /data/train --annotations /data/train_rotated.json  --backbone ResNet50FPN \ 
    --lr 0.00005 --fine-tune /data/saved_models/retinanet_rn50fpn.pth \ 
    --val-images /data/val --val-annotations /data/val_rotated.json --classes 1 \ 
    --jitter 688 848 --resize 768 \ 
    --augment-rotate --augment-brightness 0.01 --augment-contrast 0.01 --augment-hue 0.002 \ 
    --augment-saturation 0.01 --batch 16 --regularization-l2 0.0001  --val-iters 20000 --rotated-bbox
```

