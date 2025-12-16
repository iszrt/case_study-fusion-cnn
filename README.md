# CIFAR-10 Image Classification using CNN Feature Fusion

## Overview
This project is a **Mini Case Study for Graphics and Visual Computing**, focusing on image classification using deep learning.

We propose a **feature fusion architecture** that combines two pretrained CNN backbones—**ResNet18** and **MobileNetV2**—to improve classification performance on the CIFAR-10 dataset. The goal is to demonstrate how fusing complementary feature representations can lead to more robust visual understanding compared to single-model baselines.

## Dataset
* **Dataset:** CIFAR-10
* **Classes:** 10 object categories
* **Images:** 60,000 RGB images (32×32), resized to 224×224 for pretrained models
* **Split:** Training, validation, and test sets
* **Source:** `torchvision.datasets`

## Methodology

### 1. Baseline Models
To establish a performance benchmark, the following models are trained independently:
* **ResNet18:** Pretrained on ImageNet.
* **MobileNetV2:** Pretrained on ImageNet.

### 2. Fusion Architecture
The proposed method fuses the capabilities of both backbones:
1.  **Layer Removal:** The final classification layers of ResNet18 and MobileNetV2 are removed.
2.  **Feature Extraction:**
    * **ResNet18** → Outputs 512-dimensional features.
    * **MobileNetV2** → Outputs 1280-dimensional features.
3.  **Concatenation:** Features are concatenated and passed through a small MLP classifier.
4.  **Training:** The fused model is trained end-to-end using transfer learning.

**Why this works:** This fusion leverages ResNet’s strong hierarchical representations combined with MobileNet’s lightweight and efficient feature extraction.

## Training Details
* **Framework:** PyTorch
* **Optimizer:** AdamW
* **Loss Function:** Cross-Entropy Loss with label smoothing
* **Learning Rate Scheduler:** Cosine Annealing
* **Techniques Used:**
    * Data augmentation (random crop, flip, color jitter)
    * Mixed-precision training
    * "Freeze-then-finetune" strategy for pretrained backbones

## Results
The fusion model is evaluated against the baseline models using:
* Accuracy on the test set.
* Training and validation loss/accuracy curves.
* Confusion matrices.
* Sample correct and incorrect predictions.

**Findings:** Results show that the feature fusion model achieves comparable or improved accuracy and demonstrates better generalization across several CIFAR-10 classes compared to single-backbone models.

## Visual Outputs
The project notebook includes the following visualizations:
* Training and validation curves.
* Confusion matrices.
* Correct and incorrect classification examples.
* A visual diagram of the fusion architecture.

## Repository Contents

```text
├── notebook.ipynb        # End-to-end training and evaluation
├── README.md             # Project description
└── data/                 # CIFAR-10 dataset (auto-downloaded)
