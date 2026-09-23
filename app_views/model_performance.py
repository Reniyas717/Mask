import streamlit as st
import pandas as pd
import os
from PIL import Image

def render():
    st.title("Model Performance")
    
    metrics_path = "outputs/metrics/model_comparison.csv"
    fig_dir = "outputs/figures"
    
    if not os.path.exists(metrics_path):
        st.warning("Metrics file not found. Please evaluate models first.")
        return
        
    df = pd.read_csv(metrics_path)
    
    st.markdown("### Performance Comparison Table")
    # Use streamlit-shadcn-ui table equivalent or standard dataframe
    st.dataframe(df.style.highlight_max(subset=['Accuracy', 'Macro F1', 'FPS'], color='darkgreen')
                         .highlight_min(subset=['Latency (ms)', 'Size (MB)', 'Parameters'], color='darkgreen'), 
                 width='stretch')
                 
    st.divider()
    
    st.markdown("### Detailed Analysis per Model")
    
    models = df['Model'].tolist()
    tabs = st.tabs([m.upper() for m in models])
    
    for i, m_name in enumerate(models):
        with tabs[i]:
            st.markdown(f"#### {m_name.upper()} Diagnostics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Confusion Matrix**")
                cm_path = os.path.join(fig_dir, f"{m_name}_cm.png")
                if os.path.exists(cm_path):
                    st.image(Image.open(cm_path), width='stretch')
                else:
                    st.info("Confusion matrix not generated yet.")
                    
            with col2:
                st.markdown("**ROC Curves**")
                roc_path = os.path.join(fig_dir, f"{m_name}_roc.png")
                if os.path.exists(roc_path):
                    st.image(Image.open(roc_path), width='stretch')
                else:
                    st.info("ROC curve not generated yet.")
                    
            st.markdown("**Training History**")
            hcol1, hcol2 = st.columns(2)
            acc_path = os.path.join(fig_dir, f"{m_name}_accuracy.png")
            # fallback for colab naming difference
            acc_path_alt = os.path.join(fig_dir, f"{m_name}_acc.png")
            loss_path = os.path.join(fig_dir, f"{m_name}_loss.png")
            
            with hcol1:
                if os.path.exists(acc_path):
                    st.image(Image.open(acc_path), width='stretch')
                elif os.path.exists(acc_path_alt):
                    st.image(Image.open(acc_path_alt), width='stretch')
            with hcol2:
                if os.path.exists(loss_path):
                    st.image(Image.open(loss_path), width='stretch')
