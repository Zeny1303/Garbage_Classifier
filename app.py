import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import json
import os
import subprocess

# Constants from the notebook
IMAGE_SIZE = (320, 320)
CATEGORIES = {
    0: 'paper', 1: 'cardboard', 2: 'plastic', 3: 'metal', 4: 'trash', 5: 'battery',
    6: 'shoes', 7: 'clothes', 8: 'green-glass', 9: 'brown-glass', 10: 'white-glass',
    11: 'biological'
}

# Generator label map from the notebook's test_generator.class_indices
GEN_LABEL_MAP = {
    0: 'battery', 1: 'biological', 2: 'brown-glass', 3: 'cardboard', 4: 'clothes',
    5: 'green-glass', 6: 'metal', 7: 'paper', 8: 'plastic', 9: 'shoes', 10: 'trash',
    11: 'white-glass'
}

# Load the model with safe_mode=False to allow Lambda layer deserialization
@st.cache_resource
def load_garbage_model():
    try:
        # Enable unsafe deserialization (use with caution, ensure model source is trusted)
        model = load_model('garbage_model.h5', custom_objects=None, compile=True, safe_mode=False)
        return model
    except FileNotFoundError:
        st.error("Model file 'garbage_model.h5' not found. Please ensure it's in the working directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_garbage_model()

# Function to preprocess image
def preprocess_image(img):
    img = img.resize(IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize as per Xception preprocessing (assuming from notebook)
    img_array = tf.keras.applications.xception.preprocess_input(img_array)
    return img_array

# Function to predict
def predict_garbage(img):
    if model is None:
        return "Model not loaded", 0.0
    try:
        preprocessed = preprocess_image(img)
        prediction = model.predict(preprocessed)
        pred_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction) * 100
        label = GEN_LABEL_MAP.get(pred_class, "Unknown")
        return label, confidence
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return "Prediction failed", 0.0

# Function to convert ipynb to HTML using nbconvert
@st.cache_data
def convert_ipynb_to_html(ipynb_content):
    try:
        # Save the JSON content to a temporary .ipynb file
        temp_ipynb_path = "temp/temp_notebook.ipynb"
        os.makedirs("temp", exist_ok=True)
        with open(temp_ipynb_path, "w") as f:
            json.dump(ipynb_content, f)
        
        # Use nbconvert to convert to HTML
        temp_html_path = "temp/temp_notebook.html"
        subprocess.run(["jupyter", "nbconvert", "--to", "html", temp_ipynb_path, "--output", temp_html_path], check=True)
        
        # Read the HTML
        with open(temp_html_path, "r") as f:
            html_content = f.read()
        
        # Clean up temp files
        os.remove(temp_ipynb_path)
        os.remove(temp_html_path)
        
        return html_content
    except Exception as e:
        st.error(f"Error converting notebook to HTML: {str(e)}")
        return "<p>Failed to render notebook.</p>"

# Notebook JSON (replace with full JSON from garbage-classification-transfer-learning.ipynb)
# Placeholder (you must provide the full JSON)
notebook_json = {
    "metadata": {
        "kernelspec": {
            "language": "python",
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.7.6",
            "mimetype": "text/x-python",
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py"
        },
        "kaggle": {
            "accelerator": "gpu",
            "dataSources": [
                {"sourceId": 9905, "sourceType": "datasetVersion", "datasetId": 6300},
                {"sourceId": 1874598, "sourceType": "datasetVersion", "datasetId": 1115942}
            ],
            "dockerImageVersionId": 30043,
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
            "isGpuEnabled": True
        }
    },
    "nbformat_minor": 4,
    "nbformat": 4,
    "cells": [
        {"cell_type": "markdown", "source": "# Garbage Classification using keras and transfer learning", "metadata": {}},
        # ... (replace with full cells array from the .ipynb file)
    ]
}

# Streamlit App
st.title("Garbage Classification App")

tab1, tab2 = st.tabs(["Model Inference", "Notebook Showcase"])

with tab1:
    st.header("Upload an Image for Classification")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Classify"):
            with st.spinner("Classifying..."):
                label, confidence = predict_garbage(img)
                st.success(f"Predicted Class: {label} (Confidence: {confidence:.2f}%)")

with tab2:
    st.header("Jupyter Notebook Showcase")
    st.write("Below is the rendered Jupyter Notebook for the Garbage Classification project.")
    
    # Convert and display as HTML
    html_content = convert_ipynb_to_html(notebook_json)
    st.components.v1.html(html_content, height=800, scrolling=True)

    # Result
    st.success(f"🧠 Predicted Class: *{predicted_class.capitalize()}*")
    st.balloons()

