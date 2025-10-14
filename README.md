# DSP Attrition Model

Project ini menggunakan MLflow untuk tracking experiment pada prediksi attrition.

## Struktur Folder

- `.gitignore` - File untuk mengabaikan file/folder tertentu di git
- `README.md` - Dokumentasi project
- `modeling.py` - Script utama untuk modeling
- `requirements.txt` - Daftar dependencies
- `data/` - Folder untuk menyimpan data
    - `data_clean.csv` - Data yang sudah di-preprocessing

## Cara Menjalankan

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Jalankan script modeling:
   ```bash
   python modeling.py
   ```
3. Tracking experiment dapat dilakukan dengan MLflow UI:
   ```bash
   mlflow ui
   ```
