import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# ======================
# LOAD MODEL
# ======================
model = load_model("pneumonia_model.h5")

st.title("🩺 Medical Image Classifier (Pneumonia Detection)")

# ======================
# IMAGE UPLOAD
# ======================
uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # ======================
    # PREPROCESS (CRITICAL FIX)
    # ======================
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # ======================
    # PREDICTION
    # ======================
    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        st.error("🛑 PNEUMONIA DETECTED")
    else:
        st.success("✅ NORMAL")

    st.write("Confidence Score:", float(prediction))