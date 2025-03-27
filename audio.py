import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

REAL_AUDIO_DIR = "./AUDIO/REAL"
FAKE_AUDIO_DIR = "./AUDIO/FAKE"
MODEL_SAVE_PATH = "deepfake_audio_detector.h5"


def extract_mfcc(audio_path, n_mfcc=13, max_pad_len=216):
    y, sr = librosa.load(audio_path, sr=16000)  
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  
    pad_width = max_pad_len - mfcc.shape[1]  
    if pad_width > 0:
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]  
    return mfcc

X, y = [], []

for filename in os.listdir(REAL_AUDIO_DIR):
    if filename.endswith(".wav"):
        mfcc_features = extract_mfcc(os.path.join(REAL_AUDIO_DIR, filename))
        X.append(mfcc_features)
        y.append(0)  

for filename in os.listdir(FAKE_AUDIO_DIR):
    if filename.endswith(".wav"):
        mfcc_features = extract_mfcc(os.path.join(FAKE_AUDIO_DIR, filename))
        X.append(mfcc_features)
        y.append(1)  

X = np.array(X)
y = np.array(y)

X = X[..., np.newaxis]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(13, 216, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid') 
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\n🔵 Training Model...")
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

model.save(MODEL_SAVE_PATH)
print(f"\n✅ Model saved as '{MODEL_SAVE_PATH}'")