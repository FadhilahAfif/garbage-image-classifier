# garbage-image-classifier

[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19+-ff6f00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.x-d00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

> Model deep learning untuk mengklasifikasikan gambar sampah ke dalam 12 kategori menggunakan Transfer Learning (MobileNetV2), yang dapat di-deploy di platform mobile, web, maupun server.

[Tentang Proyek](#tentang-proyek) • [Dataset](#dataset) • [Arsitektur Model](#arsitektur-model) • [Hasil](#hasil) • [Struktur Proyek](#struktur-proyek) • [Cara Penggunaan](#cara-penggunaan) • [Inferensi](#inferensi)

---

## Tentang Proyek

Proyek ini membangun model klasifikasi gambar yang mampu mengidentifikasi 12 jenis sampah dari foto. Model menggunakan **MobileNetV2** yang telah dilatih pada ImageNet sebagai feature extractor, dengan classification head khusus yang dilatih pada dataset sampah. Model akhir diekspor dalam tiga format — SavedModel, TF-Lite, dan TensorFlow.js — sehingga dapat di-deploy di lingkungan server, mobile, maupun browser.

**Fitur utama:**

- Transfer Learning dengan MobileNetV2 (fine-tuned)
- Pelatihan dua fase: feature extraction dilanjutkan fine-tuning
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Akurasi test: **95,81%** pada 12 kelas
- Ekspor multi-format: SavedModel, TF-Lite (quantized), TFJS

---

## Dataset

**Sumber:** [Garbage Classification — Kaggle](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)

| Properti | Nilai |
|---|---|
| Total gambar | 15.515 |
| Jumlah kelas | 12 |
| Resolusi gambar | Tidak seragam (bervariasi per gambar) |
| Rasio pembagian | 70% train / 15% validasi / 15% test |

**Kelas yang diklasifikasikan:**

| | | | |
|---|---|---|---|
| battery | biological | brown-glass | cardboard |
| clothes | green-glass | metal | paper |
| plastic | shoes | trash | white-glass |

> [!NOTE]
> Gambar dalam dataset ini memiliki resolusi yang tidak seragam dan di-resize ke 224×224 saat preprocessing. Tidak ada normalisasi resolusi manual yang dilakukan sebelum proses splitting.

---

## Arsitektur Model

```
Input (224 × 224 × 3)
        │
MobileNetV2 (pretrained ImageNet)
  Fase 1 → seluruh layer dibekukan
  Fase 2 → 30 layer terakhir dibuka (fine-tuning)
        │
Global Average Pooling
        │
Dense(256) → BatchNormalization → Dropout(0.4)
        │
Dense(128) → BatchNormalization → Dropout(0.3)
        │
Dense(12, activation='softmax')
```

| Parameter | Nilai |
|---|---|
| Base model | MobileNetV2 |
| Ukuran input | 224 × 224 |
| Total parameter | 2.621.900 |
| Parameter dilatih (Fase 1) | 363.148 |
| Optimizer (Fase 1) | Adam (lr=1e-3) |
| Optimizer (Fase 2) | Adam (lr=1e-5) |
| Loss function | Categorical Crossentropy |

**Augmentasi data** (hanya pada training set):

- Rotasi acak (±30°)
- Pergeseran horizontal & vertikal (±20%)
- Shear dan zoom (±15–20%)
- Horizontal flip
- Nearest-fill untuk piksel kosong

---

## Hasil

| Metrik | Nilai |
|---|---|
| Akurasi Validasi | 95,26% |
| Akurasi Test | **95,81%** |

**Performa per kelas (test set):**

| Kelas | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| battery | 0,9929 | 0,9790 | 0,9859 | 143 |
| biological | 0,9932 | 0,9732 | 0,9831 | 149 |
| brown-glass | 0,9438 | 0,9130 | 0,9282 | 92 |
| cardboard | 0,9847 | 0,9556 | 0,9699 | 135 |
| clothes | 0,9863 | 0,9900 | 0,9881 | 800 |
| green-glass | 0,9355 | 0,9158 | 0,9255 | 95 |
| metal | 0,8295 | 0,9224 | 0,8735 | 116 |
| paper | 0,9490 | 0,9430 | 0,9460 | 158 |
| plastic | 0,8613 | 0,9008 | 0,8806 | 131 |
| shoes | 0,9636 | 0,9765 | 0,9700 | 298 |
| trash | 1,0000 | 0,9811 | 0,9905 | 106 |
| white-glass | 0,8889 | 0,8205 | 0,8533 | 117 |
| **weighted avg** | **0,9589** | **0,9581** | **0,9582** | **2340** |

> [!TIP]
> Kelas dengan F1-score lebih rendah (`metal`, `plastic`, `white-glass`) memiliki karakteristik visual yang serupa satu sama lain. Performa dapat ditingkatkan dengan menambah data training atau menerapkan augmentasi khusus untuk kategori-kategori tersebut.

---

## Struktur Proyek

```
garbage-image-classifier/
├── saved_model/            # Format TensorFlow SavedModel
│   ├── saved_model.pb
│   └── variables/
├── tflite/                 # Format TF-Lite (dioptimalkan untuk mobile)
│   ├── model.tflite
│   └── label.txt
├── tfjs_model/             # Format TensorFlow.js (untuk browser)
│   ├── model.json
│   └── group1-shard1of1.bin
├── notebook.ipynb          # Notebook pelatihan lengkap
├── requirements.txt
└── README.md
```

---

## Cara Penggunaan

### Prasyarat

- Python 3.10+
- Google Colab (direkomendasikan) dengan GPU T4, atau lingkungan GPU lokal
- Akun Kaggle dengan API token `kaggle.json`

### Instalasi

```bash
pip install -r requirements.txt
```

### Download Dataset

```bash
# Letakkan kaggle.json di ~/.kaggle/ terlebih dahulu
kaggle datasets download -d mostafaabla/garbage-classification
unzip garbage-classification.zip -d garbage_raw
```

### Menjalankan Notebook

Buka `notebook.ipynb` di Google Colab dan jalankan semua sel secara berurutan. Notebook mencakup:

1. Import library
2. Download dan eksplorasi dataset (EDA)
3. Pembagian data (70/15/15)
4. Augmentasi dan data generator
5. Pembangunan model (MobileNetV2 + custom head)
6. Pelatihan dua fase dengan callbacks
7. Evaluasi (classification report + confusion matrix)
8. Ekspor model (SavedModel, TF-Lite, TFJS)
9. Demonstrasi inferensi

> [!IMPORTANT]
> Aktifkan GPU runtime di Colab sebelum menjalankan notebook: **Runtime → Change runtime type → T4 GPU**. Pelatihan tanpa GPU dapat memakan waktu berjam-jam.

---

## Inferensi

### Menggunakan TF-Lite (Python)

```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Load interpreter
interpreter = tf.lite.Interpreter(model_path="tflite/model.tflite")
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load label kelas
with open("tflite/label.txt") as f:
    class_names = f.read().splitlines()

# Preprocess gambar
img = Image.open("gambar_sampah.jpg").convert("RGB").resize((224, 224))
img_array = np.array(img, dtype=np.float32) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Jalankan inferensi
interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])[0]

pred_class = class_names[np.argmax(output)]
confidence = np.max(output) * 100

print(f"Kelas prediksi : {pred_class}")
print(f"Kepercayaan    : {confidence:.2f}%")
```

### Menggunakan SavedModel (Python)

```python
import tensorflow as tf

model = tf.saved_model.load("saved_model")
# Preprocess gambar ke tensor (1, 224, 224, 3) float32 ternormalisasi ke [0, 1]
# lalu panggil model(image_tensor)
```

---

## Referensi

- [Dokumentasi TensorFlow](https://www.tensorflow.org/api_docs)
- [Panduan Transfer Learning Keras](https://keras.io/guides/transfer_learning/)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [Dataset Garbage Classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)
- [Kelas Machine Learning Dicoding](https://www.dicoding.com/academies/185)
