# app.py
import streamlit as st
import cv2
import numpy as np
from inference import load_seg_model, segment

st.title("Brain Tumor Segmentation")

# Sidebar controls
uploaded_model = st.sidebar.file_uploader("Upload segmentation model (.h5)", type=["h5"])
if uploaded_model:
    model_path = "uploaded_model.h5"
    with open(model_path, "wb") as f:
        f.write(uploaded_model.read())
    st.sidebar.success("Custom model uploaded and saved.")
else:
    model_path = None

threshold = st.sidebar.slider("Mask threshold", 0.0, 1.0, 0.5, 0.05)
show_prob = st.sidebar.checkbox("Show probability map for debugging", False)

# Load model (falls back internally if needed)
model = load_seg_model(model_path) if model_path else load_seg_model()

uploaded = st.file_uploader("Upload MRI image", type=["png", "jpg", "jpeg"])
if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.subheader("Original Image")
    st.image(img, channels='BGR', use_container_width=True)

    # Predict probability mask
    prob_mask = segment(img, model)
    prob_resized = cv2.resize(prob_mask.astype(np.float32), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

    if show_prob:
        st.subheader("Probability Map")
        # Normalize for display
        disp = (prob_resized * 255).astype(np.uint8)
        st.image(disp, use_container_width=True)

        # Apply threshold
    mask_binary = (prob_resized >= threshold).astype(np.uint8) * 255
    st.subheader("Segmented Tumor Mask")
    st.image(mask_binary, use_container_width=True)

    # Post-process mask: closing and contour detection
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask_closed = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours and fill the largest one
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Select largest contour by area
        c = max(contours, key=cv2.contourArea)
        mask_contour = np.zeros_like(mask_binary)
        cv2.drawContours(mask_contour, [c], -1, 255, thickness=-1)
    else:
        mask_contour = mask_binary
        c = None

    st.subheader("Post-processed Mask")
    st.image(mask_contour, use_container_width=True)

    # Overlay contour on original if found
    overlay = img.copy()
    if c is not None:
        # Outline in green
        cv2.drawContours(overlay, [c], -1, (0,255,0), thickness=2)
    blended = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    st.subheader("Overlay with Contour")
    st.image(blended, channels='BGR', use_container_width=True)

    # Download button for post-processed mask
    st.download_button(
        "Download Processed Mask",
        data=cv2.imencode('.png', mask_contour)[1].tobytes(),
        file_name='mask_processed.png'
    )