from tensorflow.keras.models import load_model

model = load_model("model/parkinsons_fixed.keras")

model.summary()