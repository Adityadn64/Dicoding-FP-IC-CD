from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Path ke model di dalam repositori
MODEL_PATH = "saved_model"  # Sesuaikan dengan lokasi model di dalam repo

# Periksa apakah folder model ada
if not os.path.exists(MODEL_PATH): raise ValueError(f"❌ Model tidak ditemukan di path: {MODEL_PATH}")

# Load model menggunakan TFSMLayer
model = tf.keras.layers.TFSMLayer(MODEL_PATH, call_endpoint="serving_default")
print("✅ Model berhasil dimuat!")

def preprocess_image(image_path):
    """Preprocessing gambar untuk input model."""
    image = Image.open(image_path).convert("RGB")  # Konversi ke RGB
    image = image.resize((224, 224))  # Resize ke ukuran input model
    image_array = np.array(image) / 255.0  # Normalisasi ke [0,1]
    image_array = np.expand_dims(image_array, axis=0)  # Tambah dimensi batch
    return image_array

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Periksa apakah ada file gambar yang diupload
        if "file" not in request.files: return jsonify({"error": "Tidak ada file yang diunggah"}), 400
        
        # Ambil file gambar dari request
        image_file = request.files["file"]
        image_path = "temp.jpg"  # Simpan sementara
        image_file.save(image_path)

        # Preproses gambar
        image_data = preprocess_image(image_path)

        # Lakukan prediksi dengan model
        predictions = model(image_data)

        # Ambil output sebagai numpy array
        output_array = predictions["output_0"].numpy()

        # Ambil kelas dengan confidence tertinggi
        predicted_index = int(np.argmax(output_array[0]))
        confidence = float(np.max(output_array[0]) * 100)

        return jsonify({
            "prediction": predicted_index,
            "confidence": confidence
        })

    except Exception as e: return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
