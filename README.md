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

## 1. TinyUNet-ResNet18 (`unet_resnet.ipynb`)

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


## Results

| Model             | Val mIoU ↑ | Val Loss ↓ | Trainable Parameters |
| ----------------- | ---------- | ---------- | -------------------- |
| TinyUNet-Scratch  | `0.5347`     | `0.5893`     | `31,037,763`               |
| TinyUNet-ResNet18 | `0.7118`     | `0.4452`     | `24,950,403`               |

## Key Findings

### Transfer Learning Improves Performance

The ResNet18-based model is expected to converge faster and achieve a higher validation mIoU thanks to ImageNet pretraining. The advantage is particularly important for the thin **Border** class, where learning meaningful features from only 230 training images is challenging.

## Repository Structure

```text
.
├── unet_resnet.ipynb   # TinyUNet + pretrained ResNet18 backbone
├── unet_scratch.ipynb         # TinyUNet trained from scratch
└── README.md
```

---

## Reproducibility

Both models use the exact same train/validation split:

```python
random.seed(42)

random.shuffle(all_files)

train_files = all_files[:200]
val_files   = all_files[200:]
```

Run the notebooks independently:

```text
segmentation_fixed.ipynb
unet_scratch.ipynb
```

At the end of training, each notebook automatically reports:

* Validation mIoU
* Validation Loss
* Number of Trainable Parameters

allowing a direct comparison between the pretrained and scratch models.

