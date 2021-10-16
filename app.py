import streamlit as st
import tensorflow
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from image_model_load import teachable_machine_classification
from tensorflow import keras 
import numpy as np

#st.title("Image Classification with Google's Teachable Machine")
st.header("Brain Tumor MRI Classification Example")
st.text("Upload a brain MRI Image for image classification as tumor or no-tumor")
st.text(tensorflow.__version__)

uploaded_file = st.file_uploader("Choose a brain MRI ...", type="jpg")
if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label = teachable_machine_classification(image)
        st.write(label)

