import cv2
import numpy as np

def psnr(img1, img2):
    # Menghitung mean squared error (MSE)
    mse = np.mean((img1 - img2) ** 2)
    
    # Jika MSE adalah nol, kembalikan PSNR tak terbatas
    if mse == 0:
        return float('inf')
    
    # Hitung PSNR menggunakan formula
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value

# Input nama file gambar
filename = input("Masukkan nama file gambar (dengan ekstensi): ")

# Baca gambar menggunakan OpenCV
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("File tidak ditemukan atau bukan gambar grayscale.")
else:
    # Membaca gambar referensi yang dianggap benar (misalnya, gambar asli tanpa watermark)
    ref_img = cv2.imread("yuldane.jpg", cv2.IMREAD_GRAYSCALE)
    
    # Pastikan kedua gambar memiliki ukuran yang sama
    if img.shape == ref_img.shape:
        # Hitung PSNR antara kedua gambar
        psnr_value = psnr(ref_img, img)
        print("Nilai PSNR:", psnr_value)
    else:
        print("Ukuran gambar tidak sama dengan gambar referensi.")
