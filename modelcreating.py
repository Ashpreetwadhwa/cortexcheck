import dagshub
import mlflow
import mlflow.keras
import numpy as np

from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from keras.utils import normalize, to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from pathlib import Path

# Initialize DagsHub
dagshub.init(repo_owner='Ashpreetwadhwa', repo_name='cortexcheck', mlflow=True)


# Define model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # Two output neurons for categorical classification
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Load and preprocess data (example code, adjust as needed)
Directory = Path('Dataset/')
labels_dict = {"no": 0, "yes": 1}
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

labels = np.array(labels)
image_data = np.array(image_data, dtype='uint8') / 255

x_train, x_test, y_train, y_test = train_test_split(image_data, labels, test_size=0.2,shuffle=True,stratify=labels)
x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

#

# Train the model
model.fit(x_train, y_train, batch_size=16, epochs=10, validation_data=(x_test, y_test), shuffle=True)
# Save the model using MLflow
with mlflow.start_run():
    history = model.fit(x_train, y_train, batch_size=16, epochs=10, validation_data=(x_test, y_test), shuffle=False)

    # Log parameters
    mlflow.log_param('batch_size', 16)
    mlflow.log_param('epochs', 10)

    # Log metrics
    mlflow.log_metric('train_accuracy', history.history['accuracy'][-1])
    mlflow.log_metric('train_loss', history.history['loss'][-1])
    mlflow.log_metric('val_accuracy', history.history['val_accuracy'][-1])
    mlflow.log_metric('val_loss', history.history['val_loss'][-1])
model.save("model.h5")
