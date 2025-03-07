from flask import Flask, request, jsonify, render_template
from pyngrok import ngrok
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Autentikasi ngrok
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")
if not NGROK_AUTH_TOKEN: NGROK_AUTH_TOKEN = "279KH6O4fFzLrl8C36VTxyns37S_cbJVurv4f6ysYVNNDPdA"
    # raise ValueError("NGROK_AUTH_TOKEN is not set. Please provide a valid ngrok token.")

ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Jalankan ngrok secara manual
public_url = ngrok.connect(5000).public_url
print(f"🚀 Ngrok tunnel berjalan di: {public_url}")

# Load Model
MODEL_PATH = "saved_model"
if not os.path.exists(MODEL_PATH): raise ValueError(f"❌ Model tidak ditemukan di path: {MODEL_PATH}")

model = tf.saved_model.load(MODEL_PATH)
print("✅ Model berhasil dimuat!")

@app.route("/")
def home(): render_template('inference_with_tfjs.html')

@app.route("/predict", methods=["POST"])
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGBA")
    image = image.resize((100, 100))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return tf.convert_to_tensor(image_array, dtype=tf.float32)

def predict():
    if "file" not in request.files: return jsonify({"error": "Tidak ada file yang diunggah"}), 400

    image_file = request.files["file"]
    image_path = "temp.jpg"
    image_file.save(image_path)

    # Preproses gambar
    image_data = preprocess_image(image_path)

    # Prediksi
    predictions = saved_model(image_data)
    output_tensor = predictions['output_0'] if isinstance(predictions, dict) else predictions
    output_array = output_tensor.numpy()
    
    predicted_index = int(np.argmax(output_array[0]))
    confidence = float(np.max(output_array[0]) * 100)

    is_label = predicted_index <= len(include_label)
    label = include_label[predicted_index] if is_label else f"Label Index: {predicted_index}"

    return jsonify({
        "predicted_label": label,
        "confidence": confidence,
        "is_label": is_label
    })

if __name__ == "__main__": app.run()
