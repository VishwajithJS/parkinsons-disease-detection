import tensorflow as tf
import numpy as np
import cv2

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

model = load_model("model/parkinsons_fixed.keras")

resnet = model.get_layer("resnet50")

last_conv_layer = resnet.get_layer("conv5_block3_out")

grad_model = tf.keras.models.Model(
    inputs=resnet.input,
    outputs=[
        last_conv_layer.output,
        resnet.output
    ]
)

img_path = r"C:\Users\jsvis\OneDrive\Desktop\AI-Based Early Parkinsons Disease Detection System\TEST_DATA\healthy\Healthy.jpg"

img = image.load_img(
    img_path,
    target_size=(224,224)
)

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

with tf.GradientTape() as tape:

    conv_outputs, predictions = grad_model(img_array)

    loss = tf.reduce_mean(predictions)

grads = tape.gradient(loss, conv_outputs)

pooled_grads = tf.reduce_mean(
    grads,
    axis=(0,1,2)
)

conv_outputs = conv_outputs[0]

heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

heatmap = tf.squeeze(heatmap)

heatmap = tf.maximum(heatmap, 0)

heatmap /= tf.reduce_max(heatmap)

heatmap = heatmap.numpy()

original = cv2.imread(img_path)

heatmap = cv2.resize(
    heatmap,
    (original.shape[1], original.shape[0])
)

heatmap = np.uint8(255 * heatmap)

heatmap = cv2.applyColorMap(
    heatmap,
    cv2.COLORMAP_JET
)

superimposed = cv2.addWeighted(
    original,
    0.6,
    heatmap,
    0.4,
    0
)

cv2.imwrite(
    "gradcam_output.jpg",
    superimposed
)

print("GradCAM saved successfully!")