# NVIDIA RetinaNet Pytorch Repo CHANGELOG

## 2020-02-24

* Add training argument `--augment-rotate`
  * This randomly rotates the training images by 0&deg;, 90&deg;, 180&deg; or 270&deg;.
  * Omitting the `--augment-rotate` flag will keep the default behaviour of no rotation (0&deg;).
  * This flag is in addition to the default horizontal flip with probability P(hflip) = 0.5.