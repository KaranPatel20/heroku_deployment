import streamlit as st
import tensorflow
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from model import teachable_machine_classification
from tensorflow import keras 
import numpy as np

def teachable_machine_classification(img):
    # Load the model
    model = tensorflow.keras.models.load_model('Model-RESNET-final.h5')

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    return (prediction) # return position of the highest probability
