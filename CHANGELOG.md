# NVIDIA ODTK change log

## Version 0.2.6 -- 2021-04-04

### Added
* `--no-apex` option to `odtk train` and `odtk infer`.
  * This parameter allows you to switch to Pytorch native AMP and DistributedDataParallel.
* Adding validation stats to TensorBoard.

### Changed
* Pytorch Docker container 20.11 from 20.06
* Added training and inference support for PyTorch native AMP, and torch.nn.parallel.DistributedDataParallel (use `--no-apex`).
* Switched the Pytorch Model and Data Memory Format to Channels Last. (see [Memory Format Tutorial](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html))
* Bug fixes:
  * Workaround for `'No detections!'` during vlidation added. (see [#52663](https://github.com/pytorch/pytorch/issues/52663))
  * Freeze unused parameters from torchvision models from autograd gradient calculations.
  * Make tensorboard writer exclusive to the master process to prevent race conditions.
* Renamed instances of `retinanet` to `odtk` (folder, C++ namepsaces, etc.)


## Version 0.2.5 -- 2020-06-27

### Added
* `--dynamic-batch-opts` option to `odtk export`.
  * This parameter allows you to provide TensorRT Optimiation Profile batch sizes for engine export (min, opt, max).

### Changed
* Updated TensorRT plugins to allow for dynamic batch sizes (see https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work_dynamic_shapes and https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_plugin_v2_dynamic_ext.html).


## Version 0.2.4 -- 2020-04-20

### Added
* `--anchor-ious` option to `odtk train`.
  * This parameter allows you to adjust the background and foreground anchor IoU threshold. The default values are `[0.4, 0.5].` 
  * Example `--anchor-ious 0.3 0.5`. This would mean that any anchor with an IoU of less than 0.3 is assigned to background, 
  and that any anchor with an IoU of greater than 0.5 is assigned to the foreground object, which is atmost one.

## Version 0.2.3 -- 2020-04-14

### Added
* `MobileNetV2FPN` backbone

## Version 0.2.2 -- 2020-04-01

### Added
* Rotated bounding box detections models can now be exported to ONNX and TensorRT using `odtk export model.pth model.plan --rotated-bbox`
* The `--rotated-bbox` flag is automatically applied when running `odtk infer` or `odtk export` _on a model trained with ODTK version 0.2.2 or later_. 

### Changed

* Improvements to the rotated IoU calculations.

### Limitations

* The C++ API cannot currently infer rotated bounding box models.

## Version 0.2.1 -- 2020-03-18

### Added
* The DALI dataloader (flag `--with-dali`) now supports image augmentation using:
   * `--augment-brightness` : Randomly adjusts brightness of image
   * `--augment-contrast` : Randomly adjusts contrast of image
   * `--augment-hue` : Randomly adjusts hue of image
   * `--augment-saturation` : Randomly adjusts saturation of image

### Changed
* The code in `box.py` for generating anchors has been improved.

## Version 0.2.0 -- 2020-03-13

Version 0.2.0 introduces rotated detections.

### Added
* `train arguments`:
  * `--rotated-bbox`: Trains a model is predict rotated bounding boxes `[x, y, w, h, theta]` instead of axis aligned boxes `[x, y, w, h]`.
* `infer arguments`:
  * `--rotated-bbox`: Infer a rotated model.

### Changed
The project has reverted to the name **Object Detection Toolkit** (ODTK), to better reflect the multi-network nature of the repo.
* `retinanet` has been replaced with `odtk`. All subcommands remain the same. 

### Limitations
* Models trained using the `--rotated-bbox` flag cannot be exported to ONNX or a TensorRT Engine.
* PyTorch raises two warnings which can be ignored:

Warning 1: NCCL watchdog
```
[E ProcessGroupNCCL.cpp:284] NCCL watchdog thread terminated
```

Warning 2: Save state warning
```
/opt/conda/lib/python3.6/site-packages/torch/optim/lr_scheduler.py:201: UserWarning: Please also save or load the state of the optimzer when saving or loading the scheduler.
  warnings.warn(SAVE_STATE_WARNING, UserWarning)
```

## Version 0.1.1 -- 2020-03-06

### Added
 * `train` arguments
   * `--augment-rotate`: Randomly rotates the training images by 0&deg;, 90&deg;, 180&deg; or 270&deg;.
   * `--augment-brightness` : Randomly adjusts brightness of image
   * `--augment-contrast` : Randomly adjusts contrast of image
   * `--augment-hue` : Randomly adjusts hue of image
   * `--augment-saturation` : Randomly adjusts saturation of image
   * `--regularization-l2` : Sets the L2 regularization of the optimizer.
