import os
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

@st.cache_resource
def load_seg_model(model_path="checkpoints/unet_best.h5"):
    # Fallback to final model if best checkpoint is missing
    if not os.path.exists(model_path):
        fallback = "unet_final.h5"
        if os.path.exists(fallback):
            model_path = fallback
        else:
            st.error(f"Model file not found: {model_path} or {fallback}.")
            st.stop()
    return load_model(model_path)

@st.cache_data
def preprocess_img(img, img_size=(256, 256)):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, img_size)
    img_norm = img_resized.astype(np.float32) / 255.0
    return img_norm[..., np.newaxis]

@st.cache_data
def segment(image, model):
    inp = preprocess_img(image)
    pred = model.predict(np.expand_dims(inp, axis=0))[0, ..., 0]
    mask = (pred > 0.5).astype(np.uint8) * 255
    return mask
