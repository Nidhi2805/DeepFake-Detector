import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

IMAGE_SIZE = (224, 224)

model_path = "deepfake_detector_finetuned.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found. Train and save the model first.")

print(f"Loading trained model from '{model_path}'...")
model = tf.keras.models.load_model(model_path)

def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"⚠ Test image '{image_path}' not found. Please provide a valid path.")
        return

    print(f"\nTesting the model on: {image_path}")

    img = image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  
    prediction = model.predict(img_array)[0][0]
    corrected_prediction = 1 - prediction  
    if corrected_prediction > 0.5:
        print(f"Prediction: FAKE (Confidence: {corrected_prediction:.2f})")
    else:
        print(f"Prediction: REAL (Confidence: {corrected_prediction:.2f})")

test_image_path = "./saree.jpg"  
predict_image(test_image_path)