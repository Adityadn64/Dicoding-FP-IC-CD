from flask import Flask, request, jsonify, render_template
from flask_ngrok import run_with_ngrok
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import io

app = Flask(__name__)

# Path ke model di dalam repositori
MODEL_PATH = "saved_model"

# Periksa apakah folder model ada
if not os.path.exists(MODEL_PATH): raise ValueError(f"❌ Model tidak ditemukan di path: {MODEL_PATH}")

# Load model menggunakan `tf.saved_model.load()`
model = tf.saved_model.load(MODEL_PATH)
infer = model.signatures["serving_default"]
print("✅ Model berhasil dimuat!")

# Load label dari file
label_file_path = "tflite/label.txt"
if not os.path.exists(label_file_path): raise ValueError("❌ Label file tidak ditemukan!")

with open(label_file_path, "r") as f: include_label = f.read().splitlines()

# Fungsi Preprocessing Gambar
def preprocess_image(image):
    """Preprocessing gambar untuk input model."""
    image = Image.open(image).convert("RGB")  # Konversi ke RGB
    image = image.resize((224, 224))  # Resize ke ukuran model
    image_array = np.array(image, dtype=np.float32) / 255.0  # Normalisasi [0,1]
    image_array = np.expand_dims(image_array, axis=0)  # Tambah dimensi batch
    return tf.convert_to_tensor(image_array)  # Konversi ke tensor

@app.route('/')
def home(): return render_template('inference_with_tfjs.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Periksa apakah ada file yang dikirim
        if "file" not in request.files: return jsonify({"error": "Tidak ada file yang diunggah"}), 400
        
        # Ambil file gambar dari request
        image_file = request.files["file"]
        image = preprocess_image(image_file)

        # Lakukan prediksi dengan model
        predictions = infer(image)

        # Ambil output tensor sebagai numpy array
        output_array = predictions["output_0"].numpy()

        # Ambil kelas dengan confidence tertinggi
        predicted_index = int(np.argmax(output_array[0]))
        confidence = float(np.max(output_array[0]) * 100)

        return jsonify({
            "prediction": include_label[predicted_index],
            "confidence": confidence
        })

    except Exception as e: return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    run_with_ngrok(app)
    app.run(debug=True)
