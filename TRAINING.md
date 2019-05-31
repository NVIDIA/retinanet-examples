# Training

There are two main ways to train a model with `retinanet-examples`:
* Fine-tuning the detection model using a model already trained on a large dataset (like MS-COCO)
* Fully training the detection model from random initialization using a pre-trained backbone (usually on ImageNet)

## Fine-tuning

Fine-tuning an existing model trained on COCO allows you to use transfer learning to get a accurate model for your own dataset with minimal training.
When fine-tuning, we re-initialize the last layer of the classification head so the network will re-learn how to map features to classes scores regardless of the number of classes in your own dataset.

You can fine-tune a pre-trained model on your dataset. In the example below we use [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html) with [JSON annotations](https://storage.googleapis.com/coco-dataset/external/PASCAL_VOC.zip):
```bash
retinanet train model_mydataset.pth \
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
retinanet train retinanet_rn50fpn.pth --backbone ResNet50FPN \
    --images /coco/images/train2017/ --annotations /coco/annotations/instances_train2017.json \
    --val-images /coco/images/val2017/ --val-annotations /coco/annotations/instances_val2017.json
```

We use mixed precision training by default. Full precision training can be used by providing the `full-precision` option although it doesn't provide improved accuracy in our experience.

If you want to setup your own training schedule, the following options are useful:
* `iters` is the total number of iterations you want to train the model for (1 iteration with a `batch` size of 16 correspond to going through 16 images of your dataset)
* `milestone` is a list of number of iteration at which we want to decay the learning rate
* `lr` represents the initial learning rate and `gamma` is the factor by which we multiply the learning rate at each decay milestone
* `schedule` is a float value that `iters` and `milestones` will be multiplied with to easily scale the learning schedule
* `warmup` is the number of initial iterations during which we want to linearly ramp-up the learning rate to avoid early divergence of the loss.

You can also monitor the loss and learning rate schedule of the training using TensorBoard bu specifying a `logdir` path.
