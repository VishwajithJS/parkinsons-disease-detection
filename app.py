from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf
import numpy as np
import cv2
import os
import csv
from datetime import datetime

app = Flask(__name__)

# ── Load model once at startup ────────────────────────────────────────────
model = load_model("model/parkinsons_fixed.keras")

# ── Folders ───────────────────────────────────────────────────────────────
UPLOAD_FOLDER  = "uploads"
GRADCAM_FOLDER = "static/gradcam"

os.makedirs(UPLOAD_FOLDER,  exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

# ── FIX 1: Threshold corrected to 0.5 (standard binary classification) ───
# 0.95 was causing almost everything to be classified as Parkinson's
THRESHOLD = 0.5

# ── Allowed file types ────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "gif"}

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )

# ── CSV log setup ─────────────────────────────────────────────────────────
LOG_FILE = "prediction_logs.csv"

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Timestamp",
            "Filename",
            "Raw Prediction",
            "Result",
            "Confidence (%)"
        ])


# ── FIX 2: Corrected Grad-CAM implementation ──────────────────────────────
# Old: loss = tf.reduce_mean(predictions)  ← wrong, doesn't target a class
# New: loss = predictions[:, 0]            ← targets the Parkinson's class
def generate_gradcam(img_path, save_path):
    try:
        # Get the ResNet50 sub-model
        resnet = model.get_layer("resnet50")

        # Target the last convolutional layer
        last_conv_layer = resnet.get_layer("conv5_block3_out")

        # Build gradient model
        grad_model = tf.keras.models.Model(
            inputs=resnet.input,
            outputs=[last_conv_layer.output, resnet.output]
        )

        # Preprocess image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)

            # FIX 2: Target the Parkinson's output neuron specifically
            loss = predictions[:, 0]

        # Compute gradients of the class score w.r.t conv layer output
        grads = tape.gradient(loss, conv_outputs)

        # Pool gradients over spatial dimensions
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the conv output channels by the pooled gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # ReLU — keep only positive activations
        heatmap = tf.maximum(heatmap, 0)

        # FIX 3: Safe normalization — avoid divide by zero
        max_val = tf.reduce_max(heatmap)
        if max_val == 0:
            heatmap = tf.zeros_like(heatmap)
        else:
            heatmap = heatmap / (max_val + 1e-8)

        heatmap = heatmap.numpy()

        # Read original image
        original = cv2.imread(img_path)
        if original is None:
            raise ValueError(f"Could not read image at {img_path}")

        # Resize heatmap to original image size
        heatmap_resized = cv2.resize(
            heatmap,
            (original.shape[1], original.shape[0])
        )

        # Convert to uint8 and apply colormap
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Superimpose heatmap on original image
        superimposed = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)

        cv2.imwrite(save_path, superimposed)
        return True

    except Exception as e:
        print(f"Grad-CAM generation failed: {e}")
        return False


# ── Routes ────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# FIX 4: Serve Grad-CAM images from correct folder
@app.route("/static/gradcam/<filename>")
def gradcam_file(filename):
    return send_from_directory(GRADCAM_FOLDER, filename)


@app.route("/predict", methods=["POST"])
def predict():

    # ── FIX 5: Proper file validation with error messages ─────────────────
    if "file" not in request.files:
        return render_template("index.html", error="No file uploaded.")

    file = request.files["file"]

    if file.filename == "":
        return render_template("index.html", error="No file selected.")

    if not allowed_file(file.filename):
        return render_template(
            "index.html",
            error="Invalid file type. Please upload a PNG, JPG, or BMP image."
        )

    try:
        # ── Save uploaded file ────────────────────────────────────────────
        # FIX 6: Sanitize filename to prevent path traversal
        from werkzeug.utils import secure_filename
        filename  = secure_filename(file.filename)
        filepath  = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # ── Preprocess image for model ────────────────────────────────────
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # ── Run prediction ────────────────────────────────────────────────
        raw_prediction = float(model.predict(img_array)[0][0])
        print(f"RAW PREDICTION: {raw_prediction:.4f}")

        # ── FIX 1: Corrected classification logic ─────────────────────────
        # Old logic had a gap: 0.5–0.94 showed "Healthy" which is misleading
        # Now: >= 0.5 = Parkinson's, < 0.5 = Healthy (standard binary split)
        if raw_prediction >= THRESHOLD:
            result     = "Parkinson's Detected"
            confidence = round(raw_prediction * 100, 2)
            color      = "#ff4d4d"
        else:
            result     = "Healthy"
            confidence = round((1 - raw_prediction) * 100, 2)
            color      = "#00ff99"

        # ── Generate Grad-CAM ─────────────────────────────────────────────
        # FIX 4: Save to static/gradcam/ with unique filename per upload
        gradcam_filename = f"gradcam_{filename}"
        gradcam_path     = os.path.join(GRADCAM_FOLDER, gradcam_filename)
        gradcam_ok       = generate_gradcam(filepath, gradcam_path)

        # ── Log prediction ────────────────────────────────────────────────
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                filename,
                round(raw_prediction, 4),
                result,
                confidence
            ])

        print(f"FINAL RESULT: {result} | Confidence: {confidence}%")

        return render_template(
            "index.html",
            prediction      = result,
            confidence      = confidence,
            color           = color,
            # FIX 4: Correct path for Grad-CAM image
            gradcam         = gradcam_filename if gradcam_ok else None,
            uploaded_image  = filename
        )

    except Exception as e:
        print(f"Prediction error: {e}")
        return render_template(
            "index.html",
            error=f"Something went wrong during prediction: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)