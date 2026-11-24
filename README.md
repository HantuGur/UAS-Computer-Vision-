# Pothole Depth Category Classification Using YOLO26 and RT-DETR with Depth-Derived Labels

UAS Computer Vision, Kelompok 5, Teknik Informatika, Universitas Tarumanagara.

Proyek ini menguji apakah detektor objek real-time bisa menentukan **kategori kedalaman lubang jalan** hanya dari **citra warna (RGB)**, tanpa sensor kedalaman saat inferensi. Peta kedalaman (depth map) hanya dipakai satu kali, yaitu saat membuat label.

## Anggota

| Nama | NIM |
|------|-----|
| Chandra Wijaya | 535250144 |
| Ramdan Thalib | 535250121 |
| Aaron Valerio Lieputra | 535250103 |
| Dimas Ananda Pasha | 535250152 |

## Ringkasan

- **Dataset:** PothRGBD (citra warna + depth map selaras + poligon segmentasi).
- **Label kategori:** dihitung otomatis dari depth map. Bidang jalan di sekeliling lubang di-fit dengan least squares, lalu kedalaman lubang diukur terhadap bidang itu (persentil ke-95).
  - `shallow`: kurang dari 25 mm
  - `medium`: 25 sampai 50 mm
  - `deep`: lebih dari 50 mm
- **Data akhir:** 977 citra dengan 1067 lubang, setelah membuang sampel yang ambigu atau tidak valid.
- **Model yang dibandingkan:** YOLO26s, YOLO26m, dan RT-DETR-L, dilatih dengan protokol yang sama.
- **Hasil utama (test set):**

| Model | mAP@0.5 | mAP@0.5:0.95 | Catatan |
|-------|---------|--------------|---------|
| RT-DETR-L | 0.664 | 0.469 | akurasi tertinggi |
| YOLO26s | 0.627 | 0.418 | sekitar 5 kali lebih cepat (77 FPS) |

  Hampir semua kesalahan klasifikasi terjadi antar kategori yang bersebelahan, dan kelas `deep` paling sulit dikenali.

Detail lengkap ada di laporan paper kelompok.

## Isi repository

```
.
├── pothole_depth_kaggle_v2.ipynb   # notebook utama (Kaggle, GPU)
└── README.md
```

Notebook mencakup:
1. Instalasi library dan pencarian dataset
2. Pengecekan struktur dataset
3. Perhitungan kedalaman dan pembuatan label kategori
4. Pembagian data train/val/test (70/15/15, stratifikasi, seed 42)
5. Training tiga model dengan pengaturan sama
6. Evaluasi di test set
7. Uji ketahanan pada kondisi buatan (gelap, silau, hujan, kabut, buram)
8. Training ulang dengan augmentasi kuat
9. Analisis citra yang berisi beberapa lubang dengan kategori berbeda

## Dataset

Dataset yang kami gunakan (hasil olahan) ada di Google Drive:

https://drive.google.com/drive/folders/1YIbeqSjmpY-hqHErlsQfhO3Z1966bHvO?usp=sharing

Dataset asli: **PothRGBD** (Yurdakul dan Taşdemir). Dataset tidak dimasukkan ke repository ini karena ukurannya besar.

## Cara menjalankan

Notebook dibuat untuk **Kaggle**.

1. Buka notebook di Kaggle (Upload notebook dari repo ini).
2. Di panel kanan, **Add Input** lalu pasang dataset `PothRGBD`.
3. Di **Settings > Accelerator**, pilih **GPU** (T4 x2 atau P100).
4. Jalankan semua sel dari atas ke bawah.

Catatan:
- Training tiga model memakan waktu berjam-jam. Training ulang dengan augmentasi kuat memakan sekitar 3 sampai 4 jam tambahan. Set `RUN_AUG_TRAINING = False` untuk melewatinya.
- Library utama: `ultralytics`, `torch`, `opencv-python`, `numpy`, `pandas`, `matplotlib`.
- Seed dikunci (`SEED = 42`) supaya pembagian data bisa direproduksi.

## Output

Notebook menghasilkan file hasil (metrik, grafik, confusion matrix, bobot `best.pt`) di `/kaggle/working`, termasuk `pothole_results.zip` dan `pothole_robustness.zip`.
