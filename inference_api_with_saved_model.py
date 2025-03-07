import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, request, jsonify, render_template
from pyngrok import ngrok
import numpy as np
from PIL import Image
import tensorflow as tf

template_path = os.path.abspath("./")  # Menggunakan root folder
print(f"Template path: {template_path}")  # Debugging

app = Flask(__name__, template_folder=template_path)

# Autentikasi ngrok
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")
if not NGROK_AUTH_TOKEN:
    NGROK_AUTH_TOKEN = "279KH6O4fFzLrl8C36VTxyns37S_cbJVurv4f6ysYVNNDPdA"
    print("NGROK_AUTH_TOKEN is not set. Please provide a valid ngrok token.")

ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Jalankan Ngrok
public_url = ngrok.connect(5000).public_url
print(f"🚀 Ngrok tunnel berjalan di: {public_url}")

# Pastikan Model Ada
MODEL_PATH = "saved_model"
if not os.path.exists(MODEL_PATH): raise ValueError(f"Model tidak ditemukan di path: {MODEL_PATH}")

try:
    saved_model = tf.saved_model.load(MODEL_PATH)
    print("Model berhasil dimuat!")
except Exception as e:  raise ValueError(f"Gagal memuat model: {e}")

include_label = [
    "Fresh Apples"
    "Fresh Banana"
    "Rotten Apples"
    "Rotten Banana"
]

@app.route("/")
def home(): return render_template('inference_with_tfjs.html')

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files: return jsonify({"error": "Tidak ada file yang diunggah"}), 400

    image_file = request.files["file"]
    image_path = "temp.jpg"
    image_file.save(image_path)

    image = Image.open(image_path).convert("RGB")
    image = image.resize((100, 100))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    image_data = tf.convert_to_tensor(image_array, dtype=tf.float32)

    predictions = saved_model(image_data)
    output_tensor = predictions['output_0'] if isinstance(predictions, dict) else predictions
    output_array = output_tensor.numpy()
    
    predicted_index = int(np.argmax(output_array[0]))
    confidence = float(np.max(output_array[0]) * 100)

    is_label = predicted_index < len(include_label)
    label = include_label[predicted_index] if is_label else f"Label Index: {predicted_index}"

    data = {
        "predicted_label": label,
        "confidence": confidence,
        "is_label": is_label
    }

    print(data)

    return jsonify(data)

# Handler Error 500 Dengan Detail
@app.errorhandler(500)
def handle_500_error(e): return jsonify({"error": "Terjadi kesalahan di server", "message": str(e)}), 500

# Jalankan Server dengan Debug Mode
if __name__ == "__main__": app.run()
