import os
import numpy as np
import pandas as pd
from glob import glob
import tensorflow as tf

DATA_INDEX = "data/raw/dataset_index.csv"
SAMPLES = 100  

def load_example(fname):
    df = pd.read_csv(fname)
    arr = df[['ax','ay','az','gx','gy','gz']].values
    if arr.shape[0] < SAMPLES:
        pad = np.zeros((SAMPLES - arr.shape[0], arr.shape[1]))
        arr = np.vstack([arr, pad])
    else:
        arr = arr[:SAMPLES,:]
    return arr


idx = pd.read_csv(DATA_INDEX)
X = []
y = []
labels = sorted(idx['label'].unique())
label_to_i = {l:i for i,l in enumerate(labels)}
for i, row in idx.iterrows():
    arr = load_example(row['filename'])
    X.append(arr)
    y.append(label_to_i[row['label']])
X = np.array(X)  
y = np.array(y)

X_mean = X.mean(axis=(0,1), keepdims=True)
X_std = X.std(axis=(0,1), keepdims=True) + 1e-9
X = (X - X_mean) / X_std


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(SAMPLES,6)),
    tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPool1D(2),
    tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=20, batch_size=8, validation_split=0.15)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  
tflite_model = converter.convert()
with open("ml/model.tflite", "wb") as f:
    f.write(tflite_model)

print("Saved ml/model.tflite")
