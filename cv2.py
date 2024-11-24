import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Fungsi untuk menampilkan gambar
def show_images(titles, images, cmap=None):
    if cmap is None:
        cmap = [None] * len(images)
    n = len(images)
    plt.figure(figsize=(15, 5))
    for i in range(n):
        if images[i] is None:
            raise ValueError(f"Gambar dengan indeks {i} tidak valid.")
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i], cmap=cmap[i])
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

# Periksa keberadaan file gambar
file_path = 'image.jpg'  # Ubah ke jalur absolut jika perlu
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File '{file_path}' tidak ditemukan. Pastikan file ada di direktori yang benar.")

# Baca citra berwarna
image_color = cv2.imread(file_path)
if image_color is None:
    raise FileNotFoundError(f"File '{file_path}' tidak dapat dibaca. Periksa format dan izin file.")
image_color = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)

# Konversi ke grayscale
image_gray = cv2.cvtColor(image_color, cv2.COLOR_RGB2GRAY)

# Filter Low-Pass (Gaussian Blur)
low_pass_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)
low_pass_color = cv2.GaussianBlur(image_color, (5, 5), 0)

# Filter High-Pass (Laplacian)
high_pass_gray = cv2.Laplacian(image_gray, cv2.CV_64F)
high_pass_color = cv2.Laplacian(image_color, cv2.CV_64F)
high_pass_color = np.clip(np.abs(high_pass_color), 0, 255).astype(np.uint8)

# Filter High-Boost
boost_factor = 1.5
high_boost_gray = cv2.addWeighted(image_gray, boost_factor, low_pass_gray, 1 - boost_factor, 0)
high_boost_color = cv2.addWeighted(image_color.astype(np.float32), boost_factor, 
                                   low_pass_color.astype(np.float32), 1 - boost_factor, 0)
high_boost_color = np.clip(high_boost_color, 0, 255).astype(np.uint8)

# Menampilkan hasil untuk Grayscale
show_images(
    ['Original (Grayscale)', 'Low-Pass (Grayscale)', 'High-Pass (Grayscale)', 'High-Boost (Grayscale)'],
    [image_gray, low_pass_gray, high_pass_gray, high_boost_gray],
    cmap=['gray', 'gray', 'gray', 'gray']
)

# Menampilkan hasil untuk Berwarna
show_images(
    ['Original (Color)', 'Low-Pass (Color)', 'High-Pass (Color)', 'High-Boost (Color)'],
    [image_color, low_pass_color, high_pass_color, high_boost_color]
)
