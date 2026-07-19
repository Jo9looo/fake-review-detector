# Fake Review Detector Dashboard

Sistem deteksi ulasan produk e-commerce palsu (*fake review*) berbasis Machine Learning yang menggabungkan ekstraksi fitur leksikal **TF-IDF** dengan **5 Fitur Stilometrik-Sentimen**. Proyek ini dikembangkan sebagai asesmen Capstone Project Mata Kuliah Pembelajaran Mesin di Universitas Dian Nuswantoro (UDINUS).

Aplikasi ini dilengkapi antarmuka web interaktif menggunakan Streamlit, visualisasi data eksploratif Plotly, serta interpretasi keputusan model lokal menggunakan **LIME** (*Local Interpretable Model-agnostic Explanations*).

---

## Ringkasan Performa Model

Pengujian dilakukan menggunakan **Mexwell Fake Reviews Dataset** (~40.094 ulasan) dengan pembagian data 80:20 (stratified split). Berdasarkan evaluasi 5 algoritma klasifikasi, **Linear Support Vector Machine (Linear SVM)** memperoleh hasil performa tertinggi:

| Algoritma | Eksperimen 1: TF-IDF Only (F1-Score) | Eksperimen 2: TF-IDF + Fitur Stilometrik (F1-Score) | Peningkatan |
| :--- | :---: | :---: | :---: |
| **SVM (Linear)** | 87.79% | **89.02%** | **+1.23%** |
| **Logistic Regression** | 87.32% | 88.04% | +0.72% |
| **Naive Bayes** | 84.64% | 85.23% | +0.59% |
| **Random Forest** | 74.30% | 77.99% | +3.69% |
| **LightGBM** | 75.08% | 78.14% | +3.06% |

Signifikansi peningkatan performa antara model baseline dan pemanfaatan fitur stilometrik divalidasi menggunakan **McNemar's Test** dengan hasil signifikan ($p < 0.05$).

---

## Fitur Utama Aplikasi

1. **Demo Deteksi Ulasan Real-Time**
   - Mendukung masukan teks ulasan dalam Bahasa Indonesia dan Bahasa Inggris (terjemahan otomatis via `deep-translator`).
   - Menampilkan skor probabilitas keyakinan model dan klasifikasi sub-tipe kecurangan (Deceptive, Bot-Generated, Spam, Paid/Promotional).
   - Visualisasi kontribusi kata *Explainable AI* (XAI) berbasis bobot koefisien SVM (merah = cenderung palsu, hijau = cenderung asli).

2. **Dashboard EDA Interaktif**
   - Visualisasi distribusi kelas data seimbang (50.3% Genuine vs 49.7% Fake).
   - Analisis frekuensi frasa dominan (Unigram & Bigram).
   - Matriks korelasi ketiadaan multikolinearitas antar fitur linguistik.

3. **Evaluasi & Performa Model**
   - Grafik perbandingan performa 5 model klasifikasi secara interaktif.
   - Matriks kebingungan (*Confusion Matrix*) dan skor 5-Fold Cross Validation.
   - Studi Ablasi Fitur (*Feature Ablation Study*) dan tingkat kepentingan fitur (*Feature Importance*).

4. **Interpretasi Model LIME**
   - Pembongkaran keputusan model secara lokal menggunakan pustaka LIME untuk memberikan transparansi penjelasan prediksi per kata.

---

## Fitur Linguistik (Feature Engineering)

Selain matriks frekuensi kata TF-IDF, sistem mengekstrak 5 fitur stilometrik tambahan dari setiap ulasan:

1. `text_length`: Total karakter dalam ulasan.
2. `exclamation_count`: Jumlah penggunaan tanda seru `!`.
3. `uppercase_ratio`: Rasio kata kapital penuh (*ALL-CAPS*).
4. `sentiment_score`: Skor polaritas emosi VADER (-1 s.d. +1).
5. `lexical_diversity`: Rasio keanekaragaman kata unik (*Type-Token Ratio / TTR*).

---

## Struktur Folder Proyek

```text
Mechine_learning_Final/
├── streamlit_app.py        # Aplikasi utama Streamlit Dashboard
├── train.py                # Skrip pelatihan model otomatis
├── extract_weights.py      # PengeskrIP bobot SVM ke format JSON
├── preprocessing.py        # Modul pembersihan teks & ekstraksi fitur
├── requirements.txt        # Dependensi pustaka Python
├── .gitignore              # Konfigurasi pengabaian file Git
├── outputs/
│   ├── figures/            # Output visualisasi grafik (PNG & HTML)
│   └── results/            # Parameter model (.json, .joblib, .csv)
└── docs/                   # Dokumentasi & draf laporan proyek
```

---

## Cara Menjalankan Proyek Secara Lokal

### 1. Kloning Repository & Instal Dependensi
```bash
git clone https://github.com/Jo9looo/fake-review-detector.git
cd fake-review-detector
pip install -r requirements.txt
```

### 2. Jalankan Pelatihan Model (Opsional)
Jika ingin melatih ulang model dan mengekstrak bobot parameter baru:
```bash
python train.py
```

### 3. Jalankan Aplikasi Web Streamlit
```bash
streamlit run streamlit_app.py
```
Aplikasi akan terbuka di browser lokal pada alamat `http://localhost:8501`.

---

## Pustaka Utama

* Data Processing: `pandas`, `numpy`
* Machine Learning: `scikit-learn`, `lightgbm`
* Model Interpretation: `lime`
* Natural Language Processing: `nltk`, `vaderSentiment`, `deep-translator`
* Visualization: `plotly`, `matplotlib`, `seaborn`
* Web Framework: `streamlit`

---

## Identitas Pengembang

* **Nama**: Muhammad Iqbal Satria Hutama
* **NIM**: A11.2024.15637
* **Program Studi**: Teknik Informatika - Universitas Dian Nuswantoro
