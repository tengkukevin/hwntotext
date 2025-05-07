import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
import cv2

# Load model MNIST GoogLeNet
MODEL_PATH = "googlenet_mnist.h5"
model = load_model(MODEL_PATH, compile=False)

# Konfigurasi halaman
st.title("MNIST GoogLeNet")

# Tambahkan canvas untuk menggambar angka
canvas_result = st_canvas(
    fill_color="black",  # Warna background
    stroke_width=20,     # Ketebalan garis
    stroke_color="white",# Warna coretan (angka)
    background_color="black",  # Latar belakang hitam
    width=320,
    height=320,
    drawing_mode="freedraw",
    key="canvas",
)

# Tombol untuk prediksi
if st.button("Cek Angka"):
    if canvas_result.image_data is not None:
        # Ambil gambar dari canvas
        img = np.array(canvas_result.image_data)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)  # Konversi ke grayscale

        # FIX 2: Resize dengan INTER_NEAREST untuk menghindari distorsi
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_NEAREST)

        # FIX 3: Gunakan GaussianBlur untuk menghaluskan gambar
        # img = cv2.GaussianBlur(img, (3, 3), 0)

        # Normalisasi (pastikan nilainya antara 0-1)
        img = img / 255.0

        # FIX 4: Ubah grayscale (1 channel) ke RGB (3 channel)
        img_rgb = np.stack((img,)*3, axis=-1)  # (32, 32, 1) -> (32, 32, 3)

        # Format input sesuai model
        img_input = img_rgb.reshape(1, 32, 32, 3)

        # Tampilkan gambar yang telah diproses sebelum prediksi
        st.image(img_rgb, caption="Gambar yang Diproses", use_column_width=False, width=128)

        # Prediksi angka
        prediction = model.predict(img_input)
        predicted_label = np.argmax(prediction)

        st.write(f"### Angka yang Diprediksi: {predicted_label}")
    else:
        st.write("Gambar angka terlebih dahulu!")
