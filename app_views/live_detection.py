import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import time

from src.inference import process_frame, load_inference_model

@st.cache_resource
def get_model(model_name):
    model_path = os.path.join("models", f"{model_name}.h5")
    class_idx_path = os.path.join("models", "class_index.json")
    if not os.path.exists(model_path) or not os.path.exists(class_idx_path):
        return None, None
    return load_inference_model(model_path, class_idx_path)


def _run_inference_on_image(image_pil, model, idx_to_class):
    """Convert PIL image to BGR, run process_frame, return annotated RGB image."""
    img_array = np.array(image_pil.convert("RGB"))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    annotated_bgr = process_frame(img_bgr, model, idx_to_class)
    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)


def render():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:wght@700&display=swap');

    .ld-header h1 {
        font-family: 'Playfair Display', serif;
        font-size: 2.2rem;
        color: #1A1A1A;
        margin: 0 0 0.25rem 0;
    }
    .ld-header p {
        font-family: 'Inter', sans-serif;
        color: #718096;
        font-size: 0.95rem;
        margin: 0 0 1.5rem 0;
    }
    .panel-title {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #A0AEC0;
        margin-bottom: 0.75rem;
    }
    .result-card {
        border-radius: 14px;
        border: 1px solid #E2E8F0;
        background: #FAFBFC;
        padding: 1.2rem 1.4rem;
        margin-top: 1rem;
        font-family: 'Inter', sans-serif;
    }
    .result-card .rc-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        color: #A0AEC0;
        margin-bottom: 0.3rem;
    }
    .result-card .rc-verdict {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .result-card .rc-conf {
        font-size: 0.82rem;
        color: #718096;
    }
    .legend-strip {
        display: flex;
        gap: 1.5rem;
        font-family: 'Inter', sans-serif;
        font-size: 0.82rem;
        color: #718096;
        margin-top: 1.2rem;
        padding: 0.6rem 1rem;
        background: #F8F9FA;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
    }
    .legend-dot {
        display: inline-block;
        width: 9px;
        height: 9px;
        border-radius: 50%;
        margin-right: 5px;
        vertical-align: middle;
    }
    .cam-info-box {
        background: linear-gradient(135deg, #EBF4FF 0%, #F0FFF4 100%);
        border: 1px solid #BEE3F8;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
        color: #2D3748;
        margin-bottom: 1rem;
    }
    .stTabs [role="tablist"] { border-bottom: 2px solid #E2E8F0; gap: 2rem; }
    .stTabs [role="tab"] { font-family: 'Inter', sans-serif; font-weight: 500; color: #718096; border: none !important; padding-bottom: 0.75rem; }
    .stTabs [aria-selected="true"] { color: #4A7C59 !important; border-bottom: 2px solid #4A7C59 !important; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="ld-header">
        <h1>Detection Studio</h1>
        <p>Upload an image, use your local webcam, or capture via browser — all powered by CNN inference.</p>
    </div>
    """, unsafe_allow_html=True)

    # Model selector
    models_dir = "models"
    available_models = []
    if os.path.exists(models_dir):
        available_models = [f.split('.')[0] for f in os.listdir(models_dir) if f.endswith('.h5')]

    if not available_models:
        st.warning("No trained models found in the models/ folder.")
        return

    col_sel, col_badge, _ = st.columns([2, 3, 2])
    with col_sel:
        selected_model_name = st.selectbox("Engine", available_models, label_visibility="collapsed")

    model, idx_to_class = get_model(selected_model_name)
    if model is None:
        st.error(f"Could not load **{selected_model_name}**.")
        return

    with col_badge:
        st.markdown(f"""
        <div style="display:inline-flex;align-items:center;gap:0.5rem;background:#F0FFF4;
                    border:1px solid #C6F6D5;border-radius:8px;padding:0.42rem 0.9rem;
                    font-family:Inter;font-size:0.85rem;color:#276749;margin-top:0.2rem;">
            <span style="font-size:0.55rem;">&#9679;</span>
            Model: <strong>{selected_model_name.replace('_',' ').title()}</strong>
        </div>
        """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Image Upload", "Local Camera", "Browser Camera"])

    # ------------------------------------------------------------------ #
    # TAB 1 — Image Upload                                                #
    # ------------------------------------------------------------------ #
    with tab1:
        up_col, res_col = st.columns([1, 1], gap="large")
        with up_col:
            st.markdown('<div class="panel-title">Input Image</div>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Drop an image or click to browse",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed"
            )
        with res_col:
            st.markdown('<div class="panel-title">Inference Result</div>', unsafe_allow_html=True)
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                with st.spinner("Running model..."):
                    result_rgb = _run_inference_on_image(image, model, idx_to_class)
                st.image(result_rgb, use_container_width=True)
            else:
                st.markdown("""
                <div style="height:220px;display:flex;align-items:center;justify-content:center;
                            background:#F8F9FA;border:1px dashed #CBD5E0;border-radius:10px;
                            font-family:Inter;color:#CBD5E0;font-size:0.9rem;">
                    Upload an image to see results
                </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="legend-strip">
            <span><span class="legend-dot" style="background:#27AE60;"></span>Correctly Worn</span>
            <span><span class="legend-dot" style="background:#E74C3C;"></span>No Mask</span>
            <span><span class="legend-dot" style="background:#E67E22;"></span>Incorrectly Worn</span>
        </div>""", unsafe_allow_html=True)

    # ------------------------------------------------------------------ #
    # TAB 2 — Local Camera (OpenCV, works only when running locally)      #
    # ------------------------------------------------------------------ #
    with tab2:
        st.markdown("""
        <div class="cam-info-box">
            This tab only works when running the app <strong>locally on your machine</strong>.
            It uses OpenCV to access your USB/built-in webcam directly.
        </div>
        """, unsafe_allow_html=True)

        if "cam_running" not in st.session_state:
            st.session_state.cam_running = False

        btn_label = "Stop Camera" if st.session_state.cam_running else "Start Camera"
        btn_col, _ = st.columns([1, 5])
        with btn_col:
            if st.button(btn_label, type="primary" if not st.session_state.cam_running else "secondary"):
                st.session_state.cam_running = not st.session_state.cam_running
                st.rerun()

        frame_placeholder = st.empty()

        if st.session_state.cam_running:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not open webcam. Make sure it is connected and not in use by another app.")
                st.session_state.cam_running = False
            else:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                stop_placeholder = st.empty()
                with stop_placeholder.container():
                    st.info("Camera is running. Click **Stop Camera** above to stop.")

                while st.session_state.cam_running:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    annotated = process_frame(frame, model, idx_to_class)
                    frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, use_container_width=True)
                    time.sleep(0.04)

                cap.release()
                stop_placeholder.empty()
        else:
            frame_placeholder.markdown("""
            <div style="height:360px;display:flex;align-items:center;justify-content:center;
                        background:#0F172A;border-radius:12px;
                        font-family:Inter;color:#475569;font-size:0.95rem;">
                Press Start Camera to begin local detection
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="legend-strip">
            <span><span class="legend-dot" style="background:#27AE60;"></span>Correctly Worn</span>
            <span><span class="legend-dot" style="background:#E74C3C;"></span>No Mask</span>
            <span><span class="legend-dot" style="background:#E67E22;"></span>Incorrectly Worn</span>
        </div>""", unsafe_allow_html=True)

    # ------------------------------------------------------------------ #
    # TAB 3 — Browser Camera (st.camera_input, works on cloud)           #
    # ------------------------------------------------------------------ #
    with tab3:
        st.markdown("""
        <div class="cam-info-box">
            This tab works <strong>both locally and on the cloud</strong>.
            Click <em>Take Photo</em> to capture a frame from your browser camera, then see instant mask detection results below.
        </div>
        """, unsafe_allow_html=True)

        cam_col, res_col = st.columns([1, 1], gap="large")
        with cam_col:
            st.markdown('<div class="panel-title">Camera Capture</div>', unsafe_allow_html=True)
            camera_image = st.camera_input("Capture a frame", label_visibility="collapsed")

        with res_col:
            st.markdown('<div class="panel-title">Inference Result</div>', unsafe_allow_html=True)
            if camera_image is not None:
                image = Image.open(camera_image)
                with st.spinner("Analysing..."):
                    result_rgb = _run_inference_on_image(image, model, idx_to_class)
                st.image(result_rgb, use_container_width=True)
            else:
                st.markdown("""
                <div style="height:300px;display:flex;flex-direction:column;align-items:center;
                            justify-content:center;background:#F8F9FA;border:1px dashed #CBD5E0;
                            border-radius:10px;font-family:Inter;color:#CBD5E0;font-size:0.9rem;gap:0.5rem;">
                    <span style="font-size:2rem;">&#128247;</span>
                    Capture a photo to run detection
                </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="legend-strip">
            <span><span class="legend-dot" style="background:#27AE60;"></span>Correctly Worn</span>
            <span><span class="legend-dot" style="background:#E74C3C;"></span>No Mask</span>
            <span><span class="legend-dot" style="background:#E67E22;"></span>Incorrectly Worn</span>
        </div>""", unsafe_allow_html=True)
