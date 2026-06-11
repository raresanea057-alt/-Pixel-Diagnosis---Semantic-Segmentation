# Animal Image Segmentation

## Task

Given a **256×256 RGB image** of an animal at a veterinary clinic, predict the class of every pixel.

| Class ID | Description |
|----------|-------------|
| **0** | Background (anything that is not the animal) |
| **1** | Pet (the animal's body) |
| **2** | Border (a thin contour around the animal) |

### Evaluation Metric
**Mean Intersection over Union (mIoU)** averaged across all three classes.

### Dataset

- **230** training images
- **50** test images
- Resolution: **256×256**
- Format: **PNG**
- Segmentation masks contain values **{0, 1, 2}**

---

# Approach

Two architectures were compared using the **same training setup** and **same train/validation split** (`random.seed(42)`):

- **200** training images
- **30** validation images

## 1. TinyUNet-ResNet18 (`segmentation_fixed.ipynb`)

A standard U-Net decoder attached to a pretrained **ResNet18** encoder (ImageNet weights).

### Encoder
- Pretrained ResNet18 backbone
- Skip connections from:
  - `layer1`
  - `layer2`
  - `layer3`
  - `layer4`

### Input Processing
- Normalized using ImageNet mean and standard deviation

### Motivation
Leverages transfer learning to extract stronger visual features despite the small dataset size.

---

## 2. TinyUNet-Scratch (`unet_scratch.ipynb`)

An identical U-Net decoder, but with an encoder trained entirely from scratch.

### Encoder
Repeated blocks of:

```text
Conv → BatchNorm → ReLU
Conv → BatchNorm → ReLU
