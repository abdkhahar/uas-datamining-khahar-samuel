# 🎓 Student Academic Risk: Early Warning System & Lifestyle Profiling

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployment-red)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost%20%7C%20KPrototypes-green)

Proyek akhir mata kuliah Data Mining ini bertujuan untuk mengembangkan sistem deteksi dini bagi guru Bimbingan Konseling (BK) guna mengidentifikasi siswa yang berisiko mengalami kegagalan akademik. Sistem ini menggunakan pendekatan segmentasi gaya hidup dan latar belakang sosial, bukan sekadar riwayat nilai matematis.

Proyek ini dibangun menggunakan kerangka kerja **CRISP-DM (Cross-Industry Standard Process for Data Mining)**.

## 🚀 Fitur Utama
- **Lifestyle Clustering (K-Prototypes):** Mengelompokkan siswa berdasarkan perilaku sosial (konsumsi alkohol, absensi, waktu luang) dan latar belakang keluarga.
- **Risk Classification (XGBoost):** Memprediksi probabilitas risiko kegagalan siswa (Gagal vs Lulus) menggunakan fitur asli beserta hasil segmentasi klaster.
- **Explainable AI (SHAP):** Transparansi model menggunakan SHAP Waterfall Plot untuk menjelaskan *mengapa* seorang siswa diprediksi gagal.
- **Interactive Web App:** Antarmuka pengguna berbasis Streamlit untuk prediksi *real-time* dan visualisasi data.

## 📊 Dataset
Dataset yang digunakan adalah **Student Performance Dataset (Portuguese Language Course)** dari UCI Machine Learning Repository.
- **Jumlah Data:** 649 baris, 33 atribut.
- **Catatan Preprocessing:** Fitur `G1` dan `G2` dihapus untuk mencegah *data leakage*, dan fitur `G3` dikonversi menjadi klasifikasi biner (Aman vs Berisiko).

## 🛠️ Instalasi & Cara Menjalankan Lokal

Pastikan Anda memiliki Python 3.10 atau lebih baru. Ikuti langkah berikut untuk menjalankan aplikasi di komputer lokal:

1. **Clone Repositori**
   ```bash
   git clone <URL_GITHUB_ANDA>
   cd UAS_DataMining_NamaKelompok