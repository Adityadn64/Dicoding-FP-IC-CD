from flask import Flask, request, jsonify, render_template
from flask_ngrok import run_with_ngrok
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Matikan GPU untuk menghindari error CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)

# Path ke model di dalam repositori
MODEL_PATH = "saved_model"

# Periksa apakah folder model ada
if not os.path.exists(MODEL_PATH):
    raise ValueError(f"❌ Model tidak ditemukan di path: {MODEL_PATH}")

# Load model menggunakan tf.saved_model.load()
model = tf.saved_model.load(MODEL_PATH)
infer = model.signatures["serving_default"]

print("✅ Model berhasil dimuat!")

# Load Label
label_file_path = "tflite/label.txt"
if os.path.exists(label_file_path):
    with open(label_file_path, "r") as f:
        include_label = f.read().splitlines()
else:
    include_label = ["Class-0", "Class-1", "Class-2", "Class-3"]

def preprocess_image(image_path):
    """Preprocessing gambar untuk input model."""
    image = Image.open(image_path).convert("RGBA")  # Konversi ke RGBA (4 channel)
    image = image.resize((100, 100))  # Resize ke ukuran input model
    image_array = np.array(image) / 255.0  # Normalisasi ke [0,1]
    image_array = np.expand_dims(image_array, axis=0)  # Tambah dimensi batch
    return tf.convert_to_tensor(image_array, dtype=tf.float32)

@app.route('/')
def home():
    return render_template('inference_with_tfjs.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Periksa apakah ada file gambar yang diupload
        if "file" not in request.files:
            return jsonify({"error": "Tidak ada file yang diunggah"}), 400

        # Ambil file gambar dari request
        image_file = request.files["file"]
        image_path = "temp.jpg"
        image_file.save(image_path)

        # Preproses gambar
        image_data = preprocess_image(image_path)

        # Lakukan prediksi dengan model
        predictions = infer(tf.constant(image_data))["output_0"].numpy()

        # Ambil kelas dengan confidence tertinggi
        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]) * 100)

        return jsonify({
            "prediction": include_label[predicted_index],
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    run_with_ngrok(app)  # Pastikan hanya ini yang dipanggil!
    app.run()  # Hapus debug=True
