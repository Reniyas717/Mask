import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import time
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

from src.inference import process_frame, load_inference_model

@st.cache_resource
def get_model(model_name):
    model_path = os.path.join("models", f"{model_name}.h5")
    class_idx_path = os.path.join("models", "class_index.json")
    if not os.path.exists(model_path) or not os.path.exists(class_idx_path):
        return None, None
    return load_inference_model(model_path, class_idx_path)


def render():
    st.markdown("""
    <style>
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
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #A0AEC0;
        margin-bottom: 0.75rem;
    }
    .legend-strip {
        display: flex;
        gap: 1.5rem;
        font-family: 'Inter', sans-serif;
        font-size: 0.82rem;
        color: #718096;
        margin-top: 1rem;
        padding: 0.6rem 1rem;
        background: #F8F9FA;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
    }
    .legend-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 5px;
    }
    .stTabs [role="tablist"] { border-bottom: 2px solid #E2E8F0; gap: 2rem; }
    .stTabs [role="tab"] { font-family: 'Inter', sans-serif; font-weight: 500; color: #718096; border: none !important; padding-bottom: 0.75rem; }
    .stTabs [aria-selected="true"] { color: #4A7C59 !important; border-bottom: 2px solid #4A7C59 !important; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="ld-header">
        <h1>Live Detection</h1>
        <p>Real-time mask compliance classification powered by CNN inference.</p>
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

    col_sel, _ = st.columns([2, 5])
    with col_sel:
        selected_model_name = st.selectbox("Engine", available_models, label_visibility="collapsed")

    model, idx_to_class = get_model(selected_model_name)
    if model is None:
        st.error(f"Could not load **{selected_model_name}**.")
        return

    st.markdown(f"""
    <div style="display:inline-flex;align-items:center;gap:0.5rem;background:#F8F9FA;
                border:1px solid #E2E8F0;border-radius:8px;padding:0.4rem 0.9rem;
                font-family:Inter;font-size:0.85rem;color:#4A5568;margin-bottom:1.25rem;">
        <span style="color:#4A7C59;font-size:0.6rem;">&#9679;</span>
        Engine: <strong style="color:#1A1A1A;">{selected_model_name.replace('_',' ').title()}</strong>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Image Upload", "Local Camera (OpenCV)", "Web Browser Camera (Cloud)"])

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
                img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                with st.spinner("Running model..."):
                    processed_img = process_frame(img_bgr, model, idx_to_class)
                st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), width='stretch')
            else:
                st.markdown("""
                <div style="height:220px;display:flex;align-items:center;justify-content:center;
                            background:#F8F9FA;border:1px dashed #CBD5E0;border-radius:10px;
                            font-family:Inter;color:#CBD5E0;font-size:0.9rem;">
                    Upload an image to see results
                </div>""", unsafe_allow_html=True)

    # ------------------------------------------------------------------ #
    # TAB 2 — Live Webcam (server-side OpenCV, no WebRTC)                #
    # ------------------------------------------------------------------ #
    with tab2:
        # Start / Stop button
        if "cam_running" not in st.session_state:
            st.session_state.cam_running = False

        btn_label = "Stop Camera" if st.session_state.cam_running else "Start Camera"
        btn_col, _ = st.columns([1, 5])
        with btn_col:
            if st.button(btn_label, type="primary" if not st.session_state.cam_running else "secondary"):
                st.session_state.cam_running = not st.session_state.cam_running
                st.rerun()

        # Live frame placeholder
        frame_placeholder = st.empty()

        if st.session_state.cam_running:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not open webcam. Make sure it is connected and not in use by another app.")
                st.session_state.cam_running = False
            else:
                # Set resolution for performance
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                stop_placeholder = st.empty()
                with stop_placeholder.container():
                    st.info("Camera is running. Click **Stop Camera** above to stop.")

                while st.session_state.cam_running:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Run inference
                    annotated = process_frame(frame, model, idx_to_class)
                    frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, width='stretch')

                    # Small delay to prevent pegging CPU
                    time.sleep(0.04)  # ~25 FPS ceiling

                cap.release()
                stop_placeholder.empty()
        else:
            frame_placeholder.markdown("""
            <div style="height:360px;display:flex;align-items:center;justify-content:center;
                        background:#0F172A;border-radius:12px;
                        font-family:Inter;color:#475569;font-size:0.95rem;">
                Press Start Camera to begin local detection
            </div>""", unsafe_allow_html=True)
            
    # ------------------------------------------------------------------ #
    # TAB 3 — Web Browser Camera (Cloud-Ready via WebRTC)                #
    # ------------------------------------------------------------------ #
    with tab3:
        st.info("Use this tab when the app is deployed to the cloud. It streams video directly from your browser to the cloud server using WebRTC.")
        
        class MaskDetectionProcessor(VideoTransformerBase):
            def __init__(self):
                self.model = model
                self.idx_to_class = idx_to_class
                
            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                img_bgr = frame.to_ndarray(format="bgr24")
                # Run inference
                annotated_img = process_frame(img_bgr, self.model, self.idx_to_class)
                return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

        webrtc_streamer(
            key="mask-detection",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=MaskDetectionProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        st.markdown("""
        <div class="legend-strip">
            <span><span class="legend-dot" style="background:#27AE60;"></span>Correctly Worn</span>
            <span><span class="legend-dot" style="background:#E74C3C;"></span>No Mask</span>
            <span><span class="legend-dot" style="background:#E67E22;"></span>Incorrectly Worn</span>
        </div>""", unsafe_allow_html=True)
