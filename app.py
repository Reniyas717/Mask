import os
import sys
# Force python to load local 'views' folder instead of any globally installed package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from streamlit_option_menu import option_menu

# MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Mask Compliance Monitor", layout="wide", page_icon=None, initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

/* Premium image styling */
.stImage {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    selected = option_menu(
        "Main Menu", 
        ["Overview", "Model Performance", "Live Detection", "Explainability"],
        icons=['house', 'clipboard-data', 'camera-video', 'search'],
        menu_icon="cast", default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent", "border": "none"},
            "icon": {"color": "#4A5568", "font-size": "1.1rem"}, 
            "nav-link": {"font-family": "Inter", "color": "#1A1A1A", "font-size": "1rem", "text-align": "left", "margin":"0.5rem 0", "--hover-color": "#E2E8F0", "border-radius": "8px"},
            "nav-link-selected": {"background-color": "#4A7C59", "color": "white", "font-weight": "600"},
        }
    )

if selected == "Overview":
    from app_views import overview
    overview.render()
elif selected == "Model Performance":
    from app_views import model_performance
    model_performance.render()
elif selected == "Live Detection":
    from app_views import live_detection
    live_detection.render()
elif selected == "Explainability":
    from app_views import explainability
    explainability.render()
