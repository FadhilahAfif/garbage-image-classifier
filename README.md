# garbage-image-classifier

[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-ff6f00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Sequential_CNN-d00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

> Model deep learning untuk mengklasifikasikan gambar sampah ke dalam 12 kategori menggunakan CNN Sequential (dari nol), yang dapat di-deploy di platform mobile, web, maupun server.

[Tentang Proyek](#tentang-proyek) • [Dataset](#dataset) • [Arsitektur Model](#arsitektur-model) • [Struktur Proyek](#struktur-proyek) • [Cara Penggunaan](#cara-penggunaan) • [Inferensi](#inferensi)

---

## Tentang Proyek

Proyek ini membangun model klasifikasi gambar yang mampu mengidentifikasi 12 jenis sampah dari foto. Model dibangun **dari nol** menggunakan **Sequential API** dengan arsitektur CNN berlapis (Conv2D + BatchNormalization + MaxPooling). Model akhir diekspor dalam tiga format — SavedModel, TF-Lite, dan TensorFlow.js — sehingga dapat di-deploy di lingkungan server, mobile, maupun browser.

**Fitur utama:**

- CNN Sequential dari nol (5 blok konvolusi)
- Data augmentasi agresif (rotasi, brightness, color jitter, dll.)
- Class weight untuk menangani ketidakseimbangan data
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Target akurasi: ≥ 85% (train & test)
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
> Gambar dalam dataset ini memiliki resolusi yang tidak seragam dan di-resize ke 224×224 saat preprocessing. Splitting dilakukan sebelum augmentasi untuk menghindari data leakage.

---

## Arsitektur Model

```
Input (224 × 224 × 3)
        │
Block 1 — 2×Conv2D(64) + BN + MaxPool + Dropout(0.2)
        │
Block 2 — 2×Conv2D(128) + BN + MaxPool + Dropout(0.2)
        │
Block 3 — 2×Conv2D(256) + BN + MaxPool + Dropout(0.3)
        │
Block 4 — 2×Conv2D(512) + BN + MaxPool + Dropout(0.3)
        │
Block 5 — 2×Conv2D(512) + BN + MaxPool + Dropout(0.4)
        │
GlobalAveragePooling2D
        │
Dense(512) → BatchNormalization → Dropout(0.5)
        │
Dense(12, activation='softmax')
```

| Parameter | Nilai |
|---|---|
| Arsitektur | CNN Sequential (5 blok konvolusi) |
| Ukuran input | 224 × 224 |
| Kernel initializer | He Normal |
| Optimizer | Adam (lr=5e-4) |
| Loss function | CategoricalCrossentropy (label_smoothing=0.1) |
| Max epochs | 50 (dengan EarlyStopping patience=15) |
| Batch size | 32 |

**Augmentasi data** (hanya pada training set):

- Rotasi acak (±40°)
- Pergeseran horizontal & vertikal (±20%)
- Shear (±15%) dan zoom (±20%)
- Horizontal flip
- Variasi kecerahan (0.8–1.2)
- Color jitter (channel shift ±30)
- Nearest-fill untuk piksel kosong

---

## Struktur Proyek

```
garbage-image-classifier/
├── saved_model/            # Format TensorFlow SavedModel
│   ├── fingerprint.pb
│   ├── saved_model.pb
│   └── variables/
├── tflite/                 # Format TF-Lite (dioptimalkan untuk mobile)
│   ├── model.tflite
│   └── label.txt
├── tfjs_model/             # Format TensorFlow.js (untuk browser)
│   └── model.json
├── notebook.ipynb          # Notebook pelatihan lengkap
├── requirements.txt
└── README.md
```

---

## Cara Penggunaan

### Prasyarat

- Python 3.10+
- Google Colab (direkomendasikan) dengan GPU, atau lingkungan GPU lokal
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
2. Data loading (download & ekstrak dataset dari Kaggle)
3. Eksplorasi data (EDA) — distribusi kelas & sampel gambar
4. Data splitting (70/15/15)
5. Data augmentation & preprocessing
6. Membangun model (CNN Sequential)
7. Class weight untuk ketidakseimbangan data
8. Konfigurasi callbacks
9. Training model
10. Visualisasi training (akurasi & loss)
11. Evaluasi model (classification report & confusion matrix)
12. Menyimpan model (SavedModel, TF-Lite, TFJS)
13. Demonstrasi inferensi (TF-Lite)

> [!IMPORTANT]
> Aktifkan GPU runtime di Colab sebelum menjalankan notebook: **Runtime → Change runtime type → GPU**. Pelatihan CNN dari nol tanpa GPU dapat memakan waktu sangat lama.

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
- [Panduan Keras Sequential Model](https://keras.io/guides/sequential_model/)
- [Dataset Garbage Classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)
- [Kelas Machine Learning Dicoding](https://www.dicoding.com/academies/185)
