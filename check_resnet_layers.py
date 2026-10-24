from tensorflow.keras.models import load_model

model = load_model("model/parkinsons_fixed.keras")

resnet = model.get_layer("resnet50")

for layer in resnet.layers[-20:]:
    print(layer.name)