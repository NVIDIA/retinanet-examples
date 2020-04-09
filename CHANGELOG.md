# NVIDIA ODTK change log

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
