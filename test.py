import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model(
    "model/parkinsons_resnet50_model.h5",
    compile=False
)

print("MODEL LOADED SUCCESSFULLY")