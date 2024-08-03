import io

import numpy as np
import io
from keras.models import load_model
from PIL import Image
import cv2
# Load the model
model = load_model('BrainTumor10epochs.h5')


def predict_with_confidence(file):
    # Read and preprocess the image using PIL
    img = cv2.imread(file)
    img = Image.fromarray(img)
    img = img.resize((64, 64))
    img = np.expand_dims(img, axis=0)
    image_data = np.array(img, dtype='uint8') / 255


    # Make prediction
    predictions = model.predict(image_data)

    # Get the class index and confidence score
    class_index = np.argmax(predictions[0])
    confidence = predictions[0][class_index]

    # Map class index to class label (assuming 'no' and 'yes' as labels)
    labels = ['no', 'yes']
    class_label = labels[class_index]

    return class_label, confidence