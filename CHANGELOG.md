# NVIDIA PyTorch RetinaNet change log

## Version 0.1.1 -- 2020-03-06

### Added
 * `train` arguments
   * `--augment-rotate`: Randomly rotates the training images by 0&deg;, 90&deg;, 180&deg; or 270&deg;.
   * `--augment-brightness` : Randomly adjusts brightness of image
   * `--augment-contrast` : Randomly adjusts contrast of image
   * `--augment-hue` : Randomly adjusts hue of image
   * `--augment-saturation` : Randomly adjusts saturation of image
   * `--regularization-l2` : Sets the L2 regularization of the optimizer.
