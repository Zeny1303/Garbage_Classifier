import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image
import json
from streamlit_lottie import st_lottie

# ---------------------- Page Configuration ----------------------
st.set_page_config(
    page_title="Garbage Classifier",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------- Session State Theme Management ----------------------
if "current_theme" not in st.session_state:
    st.session_state.current_theme = "light"

themes = {
    "light": {
        "background_color": "white",
        "text_color": "#000000",
        "button_face": "🌞",
    },
    "dark": {
        "background_color": "#0e1117",
        "text_color": "#ffffff",
        "button_face": "🌙",
    }
}

# Theme Toggle Function
def change_theme():
    st.session_state.current_theme = "dark" if st.session_state.current_theme == "light" else "light"

current = themes[st.session_state.current_theme]

# Inject CSS for theme
st.markdown(f"""
    <style>
        .stApp {{
            background-color: {current['background_color']};
            color: {current['text_color']};
        }}
        .stButton>button {{
            background-color: transparent;
            color: {current['text_color']};
            border: 1px solid {current['text_color']};
            padding: 0.25em 1em;
            border-radius: 0.5em;
        }}
    </style>
""", unsafe_allow_html=True)

# ---------------------- Sidebar Settings ----------------------
st.sidebar.title("🛠️ Settings")
st.sidebar.button(current["button_face"], on_click=change_theme)

# ---------------------- Load the Model ----------------------
model = load_model("garbage_classifier_model.h5")
class_names = ['non_recyclable', 'organic', 'recyclable']

# ---------------------- Load Lottie Animation ----------------------
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

lottie_recycle = load_lottiefile("recycle_animation.json")

# ---------------------- App Header ----------------------
st.title("♻️ Smart Garbage Classification")
st.write("Upload an image to classify it as: **Recyclable**, **Organic**, or **Non-Recyclable**.")

# ---------------------- Lottie Animation ----------------------
from streamlit_lottie import st_lottie
st_lottie(lottie_recycle, height=200)

# ---------------------- Image Upload and Prediction ----------------------
uploaded_file = st.file_uploader("📸 Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="📷 Uploaded Image", use_container_width=True)

    # Preprocessing
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    # Prediction
    with st.spinner("🔍 Classifying..."):
        prediction = model.predict(x)
        predicted_class = class_names[np.argmax(prediction)]

    # Result
    st.success(f"🧠 Predicted Class: *{predicted_class.capitalize()}*")
    st.balloons()

