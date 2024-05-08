import streamlit as st
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import joblib
import matplotlib.pyplot as plt

# Load the model
model_filename = "svm_model.pkl"
svm_model = joblib.load(model_filename)
# Title of the web app
st.title("Panda Image Classifier")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpeg", "jpg", "png"])
if uploaded_file is not None:
    # Display the uploaded image
    img = imread(uploaded_file)
    
    # Preprocess the image
    img_resized = resize(img, (15, 15))
    img_flatten = img_resized.flatten()
    img_array = np.asarray(img_flatten).reshape(1, -1)
    
    # Make a prediction
    result = svm_model.predict(img_array)
    prediction = "Panda" if result[0] == 0 else "Not Panda"
    
    # Show the prediction
    st.write(f"Prediction: {prediction}")

    # Display the image below the prediction
    st.image(img, caption='Uploaded Image', use_column_width=False, width=500)