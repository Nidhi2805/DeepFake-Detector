import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

dataset_path = "./imagedataset/Dataset/Test"
real_images_path = os.path.join(dataset_path, "Real")
fake_images_path = os.path.join(dataset_path, "Fake")

if not os.path.exists(real_images_path) or not os.path.exists(fake_images_path):
    raise FileNotFoundError(f"Dataset folders 'Real' and 'Fake' not found in {dataset_path}. Check your dataset path.")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_INITIAL = 5  
EPOCHS_FINE_TUNE = 5  
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=30,
    horizontal_flip=True,
    zoom_range=0.3,
    shear_range=0.3,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
    dataset_path,  
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    dataset_path, 
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(1, activation="sigmoid")(x) 
model = Model(inputs=base_model.input, outputs=x)
model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS_INITIAL)

base_model.trainable = True

model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

history_fine_tune = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS_FINE_TUNE)

model.save("deepfake_detector_finetuned.h5")
plt.plot(history.history["accuracy"], label="Train Accuracy (Frozen)")
plt.plot(history_fine_tune.history["accuracy"], label="Train Accuracy (Fine-Tune)")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

def predict_image(image_path):
    from tensorflow.keras.preprocessing import image

    model = tf.keras.models.load_model("deepfake_detector_finetuned.h5")

    img = image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        print(f"Prediction: FAKE (Confidence: {prediction:.2f})")
    else:
        print(f"Prediction: REAL (Confidence: {1 - prediction:.2f})")

predict_image("./ai_image")