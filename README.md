# Workflow CI - MLflow Project

## 📌 Deskripsi
Proyek ini adalah CI/CD pipeline untuk training model machine learning otomatis.

## 🚀 Cara Kerja
1. Push code ke GitHub → otomatis training model
2. Hasil model bisa di-download dari tab "Actions"

## 📁 Struktur
- `.github/workflows/train.yml` - GitHub Actions workflow
- `MLProject/` - MLflow Project
  - `modelling.py` - Script training
  - `conda.yaml` - Dependencies
  - `MLProject` - MLflow config

## 📸 Screenshot
(Isi setelah workflow berhasil)