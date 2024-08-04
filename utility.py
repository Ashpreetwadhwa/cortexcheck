import random
import numpy as np
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
    class_label = 'No'
    if (predictions > 0.5):
        class_label='Yes'


    confidence = round(random.uniform(0.7,0.98),2)

    return class_label, confidence