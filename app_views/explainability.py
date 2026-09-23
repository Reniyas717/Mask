import streamlit as st
import os
import glob
import random
import json
import numpy as np
import cv2
from PIL import Image

from src.inference import load_inference_model
from src.explainability import get_gradcam_heatmap, overlay_gradcam, get_lime_explanation
from src.models import IMG_SIZE

@st.cache_resource
def get_model(model_name):
    model_path = os.path.join("models", f"{model_name}.h5")
    class_idx_path = os.path.join("models", "class_index.json")
    if not os.path.exists(model_path) or not os.path.exists(class_idx_path):
        return None, None, None
    model, idx_to_class = load_inference_model(model_path, class_idx_path)
    with open(class_idx_path, "r") as f:
        class_to_idx = json.load(f)
    return model, idx_to_class, class_to_idx

@st.cache_data(show_spinner=False)
def generate_gradcam(_model, img_array, pred_idx, img_path):
    heatmap = get_gradcam_heatmap(_model, img_array, pred_idx)
    overlay = overlay_gradcam(img_path, heatmap)
    return overlay

@st.cache_data(show_spinner=False)
def generate_lime(_model, img_path, pred_idx):
    return get_lime_explanation(_model, img_path, pred_idx, img_size=IMG_SIZE)

def render():
    st.title("Model Explainability (XAI)")
    st.markdown("Understand *why* the model made its decision using Grad-CAM and LIME.")
    
    models_dir = "models"
    available_models = []
    if os.path.exists(models_dir):
        available_models = [f.split('.')[0] for f in os.listdir(models_dir) if f.endswith('.h5')]
        
    if not available_models:
        st.warning("No models found.")
        return
        
    selected_model = st.selectbox("Select Model", available_models)
    model, idx_to_class, class_to_idx = get_model(selected_model)
    
    if model is None:
        st.error("Model loading failed.")
        return
        
    # Pick a random test image (filter out small blurry crops)
    if st.button("Load Random Test Image") or 'xai_img_path' not in st.session_state:
        test_images = glob.glob("data/test_samples/*/*.jpg")
        valid_images = []
        # Try to find a high-res image
        random.shuffle(test_images)
        for img_p in test_images[:50]: # check up to 50 to save time
            img_chk = cv2.imread(img_p)
            if img_chk is not None and img_chk.shape[0] >= 80 and img_chk.shape[1] >= 80:
                valid_images.append(img_p)
                break
                
        if valid_images:
            st.session_state.xai_img_path = valid_images[0]
        elif test_images:
            st.session_state.xai_img_path = random.choice(test_images)
        else:
            st.session_state.xai_img_path = None
            
    img_path = st.session_state.xai_img_path
    if img_path is None:
        st.warning("No test images found. Please run preprocessing.")
        return
        
    true_class = os.path.basename(os.path.dirname(img_path))
    
    st.markdown(f"**True Label:** `{true_class}`")
    
    # Process
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, IMG_SIZE)
    img_array = np.expand_dims(img_resized, axis=0)
    
    preds = model.predict(img_array)[0]
    pred_idx = np.argmax(preds)
    pred_class = idx_to_class[pred_idx]
    confidence = preds[pred_idx]
    
    if true_class == pred_class:
        st.success(f"**Predicted:** `{pred_class}` ({confidence*100:.1f}%) ✅")
    else:
        st.error(f"**Predicted:** `{pred_class}` ({confidence*100:.1f}%) ❌")
        
    st.markdown("""
    ---
    ### How to interpret these results:
    * **Original Image**: The preprocessed input as seen by the neural network.
    * **Grad-CAM**: Highlights (in bright yellow/red) the specific regions of the image that the model paid the most attention to when making its prediction.
    * **LIME**: Highlights super-pixels that positively contributed to the prediction in green, and areas that argued against it in red.
    ---
    """)
        
    col1, col2, col3 = st.columns(3)
    
    # Upscale size for high-quality display in UI (prevents browser blocky stretching)
    display_size = (500, 500)
    
    with col1:
        st.markdown("**Original Image**")
        orig_img = cv2.imread(img_path)
        orig_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        orig_hq = cv2.resize(orig_rgb, display_size, interpolation=cv2.INTER_CUBIC)
        st.image(orig_hq, width='stretch')
        
    with col2:
        st.markdown("**Grad-CAM**")
        with st.spinner("Generating Grad-CAM..."):
            try:
                overlay = generate_gradcam(model, img_array, pred_idx, img_path)
                overlay_hq = cv2.resize(overlay, display_size, interpolation=cv2.INTER_CUBIC)
                st.image(overlay_hq, width='stretch')
            except Exception as e:
                st.error(f"Grad-CAM error: {e}")
                
    with col3:
        st.markdown("**LIME**")
        with st.spinner("Generating LIME (takes a few seconds)..."):
            try:
                lime_img = generate_lime(model, img_path, pred_idx)
                lime_hq = cv2.resize(lime_img, display_size, interpolation=cv2.INTER_CUBIC)
                st.image(lime_hq, width='stretch')
            except Exception as e:
                st.error(f"LIME error: {e}")
