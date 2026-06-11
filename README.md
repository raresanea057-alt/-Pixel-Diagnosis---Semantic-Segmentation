# -Pixel-Diagnosis---Semantic-Segmentation
Semantic segmentation task from the Selection Camp Simulation 2 on (https://judge.nitro-ai.org), part of the Romanian national selection process for international AI olympiads.

Task: Given a 256×256 RGB photo of an animal at a veterinary clinic, predict for each pixel whether it belongs to:

ClassIDDescriptionBackground0Anything that is not the animalPet1The animal's bodyBorder2A thin contour around the animal

Metric: Mean IoU across all 3 classes
Dataset: 230 train images + 50 test images (256×256 PNG, masks in {0, 1, 2})


Approach

Two architectures compared — same training setup, same train/val split (random.seed(42), 200 train / 30 val):

TinyUNet-ResNet18 (segmentation_fixed.ipynb)

Standard UNet decoder attached to a pretrained ResNet18 backbone (ImageNet weights). Skip connections from layer1–layer4. Input normalized with ImageNet mean/std.

TinyUNet-Scratch (unet_scratch.ipynb)

Identical UNet decoder, but encoder built from scratch using Conv→BN→ReLU×2 blocks. No pretrained weights. Trained from random initialization.

Both models use:


AdamW, lr=1e-3, 10 epochs, batch_size=8
CrossEntropyLoss
Validation mIoU tracked per epoch


