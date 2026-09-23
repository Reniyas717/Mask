import streamlit as st
import pandas as pd
import os

def render():
    # Inject Custom CSS for Premium Typography & Theme
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap');

    /* Typography assignments */
    h1, h2, h3, .st-emotion-cache-10trblm {
        font-family: 'Playfair Display', serif !important;
        color: #1A1A1A !important;
    }
    p, span, div, li {
        font-family: 'Inter', sans-serif;
        color: #4A5568;
    }
    
    .metric-value {
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
    }

    /* Hero Section */
    .hero-container {
        padding: 3rem 0;
        text-align: center;
        background: linear-gradient(to bottom, #FFFFFF, #F8F9FA);
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 3rem;
        border-radius: 12px;
    }
    .hero-title {
        font-size: 3.2rem;
        margin-bottom: 0.5rem;
        color: #1A1A1A;
        font-family: 'Playfair Display', serif;
        font-weight: 700;
    }
    .hero-subtitle {
        font-size: 1.25rem;
        color: #4A7C59; /* Sage Green Primary */
        font-weight: 500;
        margin-bottom: 2rem;
        font-family: 'Inter', sans-serif;
    }

    /* Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.5rem;
        margin-bottom: 3rem;
    }
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        border-top: 4px solid #4A7C59; /* Sage Green Top Border */
    }
    .metric-card.accent {
        border-top: 4px solid #E07A5F; /* Coral Accent */
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    .metric-title {
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #718096;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .metric-value {
        font-size: 2rem;
        color: #1A1A1A;
        margin: 0;
        line-height: 1.2;
    }
    .metric-desc {
        font-size: 0.875rem;
        color: #A0AEC0;
        margin-top: 0.25rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Hero Section
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">Facial Compliance Engine</h1>
        <p class="hero-subtitle">Enterprise-Grade Mask Detection & Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load Metrics Data
    best_f1 = "0.749"
    best_model = "Baseline CNN"
    fps = "105.9"
    metrics_path = "outputs/metrics/model_comparison.csv"
    if os.path.exists(metrics_path):
        try:
            df = pd.read_csv(metrics_path)
            best_idx = df['Macro F1'].idxmax()
            best_f1 = f"{df.loc[best_idx, 'Macro F1']:.3f}"
            best_model = df.loc[best_idx, 'Model'].replace('_', ' ').title()
            fps = f"{df.loc[best_idx, 'FPS']:.1f}"
        except Exception:
            pass

    # Custom HTML Metric Cards Grid
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-title">Optimal Engine</div>
            <div class="metric-value">{best_model}</div>
            <div class="metric-desc">Highest overall F1 Score</div>
        </div>
        <div class="metric-card accent">
            <div class="metric-title">Macro F1 Score</div>
            <div class="metric-value">{best_f1}</div>
            <div class="metric-desc">Test split performance</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Inference Speed</div>
            <div class="metric-value">{fps} FPS</div>
            <div class="metric-desc">Real-time processing capability</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Compliance Classes</div>
            <div class="metric-value">03</div>
            <div class="metric-desc">Correct / Incorrect / Missing</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature Overview
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("""
        ### Intelligent Monitoring
        Our scalable computer vision pipeline replaces manual safety checks with automated, high-precision detection. Designed for high-throughput environments, the engine classifies incoming visual feeds into three strict compliance states in real-time.
        
        **Core Features:**
        - High-speed CNN architectures designed for Edge and Server deployment.
        - Robust tolerance to varying lighting and diverse facial features.
        - Streamlined operational dashboard for monitoring and evaluation.
        """)
        
    with col2:
        st.markdown("""
        ### Trust & Explainability
        Transparency is critical in automated compliance systems. This application integrates state-of-the-art explainable AI techniques (Grad-CAM and LIME) to expose exactly *why* the model makes a classification.
        
        By navigating to the **Explainability** module, operators can audit the network's focal points in real-time, ensuring that decisions are driven by mask placement rather than spurious background artifacts.
        """)
