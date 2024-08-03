import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.utils import normalize, to_categorical
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.callbacks import EarlyStopping

# Directory setup
Directory = Path('Dataset/')
labels_dict = {"no": 0, "yes": 1}

# Load and preprocess data
image_data = []
labels = []
for i in Directory.glob("*"):
    sub_directory = i.glob("*")
    label = str(i).split("\\")[-1]
    for j in sub_directory:
        img = image.load_img(j, target_size=(64, 64))
        img_array = image.img_to_array(img)
        image_data.append(img_array)
        labels.append(labels_dict[label])

# Convert to numpy arrays and normalize
labels = np.array(labels)
image_data = np.array(image_data, dtype='float32') / 255.0

# Split data
x_train, x_test, y_train, y_test = train_test_split(image_data, labels, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(x_train)

# Model architecture
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), kernel_initializer='he_uniform', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with data augmentation
history = model.fit(datagen.flow(x_train, y_train, batch_size=16),
                    epochs=50,
                    validation_data=(x_test, y_test),
                    callbacks=[early_stopping],
                    shuffle=True)

# Save the model
model.save("BrainTumorImproved.h5")

# Plot training history
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss')

plt.show()
