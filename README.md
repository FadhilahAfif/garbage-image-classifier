# Garbage Image Classification

[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19+-ff6f00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.x-d00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

> A deep learning model that classifies waste images into 12 categories using Transfer Learning (MobileNetV2), deployable across mobile, web, and server platforms.

[Overview](#overview) • [Dataset](#dataset) • [Model Architecture](#model-architecture) • [Results](#results) • [Project Structure](#project-structure) • [Getting Started](#getting-started) • [Inference](#inference)

---

## Overview

This project builds an image classification model capable of identifying 12 types of garbage from photos. The model uses **MobileNetV2** pretrained on ImageNet as a feature extractor, with a custom classification head trained on the garbage dataset. The final model is exported in three formats — SavedModel, TF-Lite, and TensorFlow.js — enabling deployment across server, mobile, and browser environments.

**Key highlights:**

- Transfer Learning with MobileNetV2 (fine-tuned)
- Two-phase training: feature extraction followed by fine-tuning
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Test accuracy: **95.81%** across 12 classes
- Multi-format export: SavedModel, TF-Lite (quantized), TFJS

---

## Dataset

**Source:** [Garbage Classification — Kaggle](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)

| Property | Value |
|---|---|
| Total images | 15,515 |
| Number of classes | 12 |
| Image resolution | Non-uniform (varies per image) |
| Split ratio | 70% train / 15% val / 15% test |

**Classes:**

| | | | |
|---|---|---|---|
| battery | biological | brown-glass | cardboard |
| clothes | green-glass | metal | paper |
| plastic | shoes | trash | white-glass |

> [!NOTE]
> Images in this dataset have non-uniform resolutions and are resized to 224×224 during preprocessing. No manual resolution normalization is applied prior to splitting.

---

## Model Architecture

```
Input (224 × 224 × 3)
        │
MobileNetV2 (pretrained on ImageNet)
  Phase 1 → fully frozen
  Phase 2 → top 30 layers unfrozen (fine-tuning)
        │
Global Average Pooling
        │
Dense(256) → BatchNormalization → Dropout(0.4)
        │
Dense(128) → BatchNormalization → Dropout(0.3)
        │
Dense(12, activation='softmax')
```

| Parameter | Value |
|---|---|
| Base model | MobileNetV2 |
| Input size | 224 × 224 |
| Total params | 2,621,900 |
| Trainable params (Phase 1) | 363,148 |
| Optimizer (Phase 1) | Adam (lr=1e-3) |
| Optimizer (Phase 2) | Adam (lr=1e-5) |
| Loss function | Categorical Crossentropy |

**Augmentation** (training set only):

- Random rotation (±30°)
- Width & height shift (±20%)
- Shear and zoom (±15–20%)
- Horizontal flip
- Nearest-fill for empty pixels

---

## Results

| Metric | Value |
|---|---|
| Validation Accuracy | 95.26% |
| Test Accuracy | **95.81%** |
| Test Loss | — |

**Per-class performance (test set):**

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| battery | 0.9929 | 0.9790 | 0.9859 |
| biological | 0.9932 | 0.9732 | 0.9831 |
| brown-glass | 0.9438 | 0.9130 | 0.9282 |
| cardboard | 0.9847 | 0.9556 | 0.9699 |
| clothes | 0.9863 | 0.9900 | 0.9881 |
| green-glass | 0.9355 | 0.9158 | 0.9255 |
| metal | 0.8295 | 0.9224 | 0.8735 |
| paper | 0.9490 | 0.9430 | 0.9460 |
| plastic | 0.8613 | 0.9008 | 0.8806 |
| shoes | 0.9636 | 0.9765 | 0.9700 |
| trash | 1.0000 | 0.9811 | 0.9905 |
| white-glass | 0.8889 | 0.8205 | 0.8533 |
| **weighted avg** | **0.9589** | **0.9581** | **0.9582** |

> [!TIP]
> Classes with lower F1-scores (`metal`, `plastic`, `white-glass`) share similar visual characteristics. Performance can be improved by adding more training data or applying class-specific augmentation for these categories.

---

## Project Structure

```
submission/
├── saved_model/            # TensorFlow SavedModel format
│   ├── saved_model.pb
│   └── variables/
├── tflite/                 # TF-Lite format (optimized for mobile)
│   ├── model.tflite
│   └── label.txt
├── tfjs_model/             # TensorFlow.js format (for browser)
│   ├── model.json
│   └── group1-shard1of1.bin
├── notebook.ipynb          # Full training notebook
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Google Colab (recommended) with T4 GPU, or a local GPU environment
- Kaggle account with `kaggle.json` API token

### Installation

```bash
pip install -r requirements.txt
```

### Download Dataset

```bash
# Place kaggle.json in ~/.kaggle/ first
kaggle datasets download -d mostafaabla/garbage-classification
unzip garbage-classification.zip -d garbage_raw
```

### Run the Notebook

Open `notebook.ipynb` in Google Colab and run all cells sequentially. The notebook covers:

1. Library imports
2. Dataset download and exploration (EDA)
3. Data splitting (70/15/15)
4. Augmentation and data generators
5. Model building (MobileNetV2 + custom head)
6. Two-phase training with callbacks
7. Evaluation (classification report + confusion matrix)
8. Model export (SavedModel, TF-Lite, TFJS)
9. Inference demonstration

> [!IMPORTANT]
> Enable GPU runtime in Colab before running: **Runtime → Change runtime type → T4 GPU**. Training on CPU may take several hours.

---

## Inference

### Using TF-Lite (Python)

```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Load interpreter
interpreter = tf.lite.Interpreter(model_path="tflite/model.tflite")
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load class labels
with open("tflite/label.txt") as f:
    class_names = f.read().splitlines()

# Preprocess image
img = Image.open("your_image.jpg").convert("RGB").resize((224, 224))
img_array = np.array(img, dtype=np.float32) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Run inference
interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])[0]

pred_class = class_names[np.argmax(output)]
confidence = np.max(output) * 100

print(f"Predicted class : {pred_class}")
print(f"Confidence      : {confidence:.2f}%")
```

### Using SavedModel (Python)

```python
import tensorflow as tf

model = tf.saved_model.load("saved_model")
# Preprocess image to (1, 224, 224, 3) float32 tensor normalized to [0, 1]
# then call model(image_tensor)
```

---

## Resources

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Transfer Learning Guide](https://keras.io/guides/transfer_learning/)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [Garbage Classification Dataset](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)
- [Dicoding Machine Learning Course](https://www.dicoding.com/academies/185)
