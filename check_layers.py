from tensorflow.keras.models import load_model

model = load_model("model/parkinsons_fixed.keras")

for layer in model.layers:
    print(layer.name, layer.output.shape)