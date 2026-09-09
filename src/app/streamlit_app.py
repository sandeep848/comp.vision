import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import torch
import torchvision.transforms as T
import sys
from pathlib import Path
import plotly.graph_objects as go
from streamlit_image_comparison import image_comparison

# Setup paths
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.configs import config
from src.models.model import build_model
from src.degradations.transforms import RandomJPEGCompression, RandomDownscaleRestore, RandomGaussianNoise, RandomMotionBlur, RandomSharpen

# Page config
st.set_page_config(page_title="Deepfake Vibe Check", page_icon="🔮", layout="wide")

# Vibe Coded CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@300;400;600&family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Fira Code', monospace;
        letter-spacing: -0.5px;
    }
    
    .glow-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00FFCC 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 0px 20px rgba(0,255,204,0.4);
        margin-bottom: 0rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #8c8f9e;
        margin-bottom: 2rem;
        font-family: 'Fira Code', monospace;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        font-family: 'Fira Code', monospace;
    }
    
    .fake-text { color: #ff4b4b; text-shadow: 0 0 10px rgba(255,75,75,0.4); }
    .real-text { color: #00FFCC; text-shadow: 0 0 10px rgba(0,255,204,0.4); }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="glow-header">🔮 Deepfake Vibe Check</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">>> Robustness benchmark // v2.0 // Neural engine</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🎛️ CONTROL PANEL")
    uploaded_file = st.file_uploader("UPLOAD TARGET [.jpg, .png]", type=["jpg", "jpeg", "png"])
    
    st.divider()
    st.markdown("### 🧠 MODEL ARCHITECTURE")
    model_choice = st.selectbox("ARCH", ["EfficientNet-B4 (Robust)", "ResNet-50 (Standard)", "Swin-T (Experimental)"])
    
    st.divider()
    st.markdown("### 🎚️ DEGRADATION ENGINE")
    
    jpeg_q = st.slider("JPEG Artifacts", 10, 100, 70, 5, help="Lower = more artifacts")
    resize_s = st.slider("Downscale", 0.1, 1.0, 1.0, 0.1, help="Lower = lower resolution")
    noise_lvl = st.slider("Gaussian Noise", 0.0, 10.0, 0.0, 0.5, help="Higher = more noise")
    blur_k = st.slider("Motion Blur", 0, 7, 0, 2, help="Must be an odd number (0 for off)")
    sharp_f = st.slider("Sharpen", 1.0, 3.0, 1.0, 0.2, help="Higher = excessive sharpening")


from src.evaluation.gradcam import generate_gradcam, get_target_layer, overlay_heatmap
from src.evaluation.evaluate import load_model_from_checkpoint
from src.degradations.transforms import get_transforms

@st.cache_resource
def load_models(model_name_choice="EfficientNet-B0"):
    # Attempt to load real checkpoints
    if "EfficientNet" in model_name_choice:
        base_name = "efficientnet_b0"
    elif "ResNet" in model_name_choice:
        base_name = "resnet50"
    else:
        base_name = config.MODEL_NAME

    clean_ckpt = config.OUTPUT_ROOT / f"{base_name}_clean" / "best_model.pt"
    robust_ckpt = config.OUTPUT_ROOT / f"{base_name}_degradation" / "best_model.pt"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not clean_ckpt.exists() and not robust_ckpt.exists():
        st.error(f"🚨 Checkpoint Load Failure: Neither clean nor robust checkpoints found for {base_name}. Please train the models first.")
        st.stop()
        
    m1, t1, m2, t2 = None, 0.5, None, 0.5
    
    if robust_ckpt.exists():
        m1, t1, _ = load_model_from_checkpoint(robust_ckpt, device)
        m1.eval()
    else:
        st.warning(f"Robust checkpoint missing for {base_name}. Using Standard for comparison.")
        
    if clean_ckpt.exists():
        m2, t2, _ = load_model_from_checkpoint(clean_ckpt, device)
        m2.eval()
    else:
        st.warning(f"Standard checkpoint missing for {base_name}.")
        
    if m1 is None:
        m1, t1 = m2, t2
    if m2 is None:
        m2, t2 = m1, t1
        
    return m1, t1, m2, t2

def run_model(model, img):
    _, eval_transform = get_transforms()
    input_tensor = eval_transform(img).unsqueeze(0)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits, _ = model(input_tensor.to(device))
        fake_prob = torch.sigmoid(logits.squeeze(1)).item()
    return fake_prob

def generate_real_heatmap(model, img, target_class="predicted", threshold=0.5):
    _, eval_transform = get_transforms()
    input_tensor = eval_transform(img).unsqueeze(0)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits, _ = model(input_tensor.to(device))
        prob = torch.sigmoid(logits.squeeze(1)).item()
        
    resolved_target = "fake" if prob >= threshold else "real"
    if target_class != "predicted":
        resolved_target = target_class
        
    target_layer = get_target_layer(model, branch="rgb")
    cam, _ = generate_gradcam(model, input_tensor, target_layer, target_class=resolved_target)
    overlay, _ = overlay_heatmap(img, cam)
    return overlay


def apply_all_degradations(img):
    # Apply based on sliders
    if resize_s < 1.0:
        img = RandomDownscaleRestore(scales=[resize_s], probability=1.0)(img)
    if blur_k > 0:
        k = blur_k if blur_k % 2 != 0 else blur_k + 1
        img = RandomMotionBlur(sizes=(k, k), probability=1.0)(img)
    if noise_lvl > 0:
        img = RandomGaussianNoise(std_range=(noise_lvl, noise_lvl), probability=1.0)(img)
    if sharp_f > 1.0:
        img = RandomSharpen(factor_range=(sharp_f, sharp_f), probability=1.0)(img)
    if jpeg_q < 100:
        img = RandomJPEGCompression(quality_range=(jpeg_q, jpeg_q), probability=1.0)(img)
    return img




if uploaded_file:
    orig_img = Image.open(uploaded_file).convert("RGB")
    deg_img = apply_all_degradations(orig_img)

    t1, t2, t3 = st.tabs(["👁️ VISION", "📈 TELEMETRY", "🔍 EXPLAINER"])
    
    model1, thresh1, model2, thresh2 = load_models(model_choice)
    
    # Real probability calculation via forward pass
    if "Robust" in model_choice:
        active_model = model1
        active_thresh = thresh1
    else:
        active_model = model2
        active_thresh = thresh2
        
    prob_clean = run_model(active_model, orig_img)
    prob_deg = run_model(active_model, deg_img)
    is_fake = prob_deg > active_thresh
    
    # Estimate degradation penalty for the UI telemetry
    degradation_penalty = max(0, prob_clean - prob_deg)


    with t1:
        st.markdown("### // IMAGE COMPARISON")
        image_comparison(
            img1=orig_img,
            img2=deg_img,
            label1="SOURCE",
            label2="CORRUPTED",
            width=800,
            starting_position=50,
            show_labels=True,
            make_responsive=True,
            in_memory=True,
        )

    with t2:
        st.markdown("### // SYSTEM READOUT")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.write("DETECTION CONFIDENCE")
            color_class = "fake-text" if is_fake else "real-text"
            label = "SYNTHETIC [FAKE]" if is_fake else "AUTHENTIC [REAL]"
            st.markdown(f'<div class="metric-value {color_class}">{prob_deg*100:.1f}%</div>', unsafe_allow_html=True)
            st.write(f"Classified as: **{label}**")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.write("CONFIDENCE DELTA")
            deg_score = min(100.0, degradation_penalty * 100)
            st.markdown(f'<div class="metric-value" style="color: #ff9900;">{deg_score:.1f}</div>', unsafe_allow_html=True)
            st.write("Confidence Drop (0-100)")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### // CONFIDENCE TRAJECTORY")
        
        # Generate chart by sweeping JPEG quality
        qs = np.linspace(100, 10, 10)
        trajectories = []
        for q in qs:
            sweep_img = RandomJPEGCompression(quality_range=(int(q), int(q)), probability=1.0)(orig_img)
            trajectories.append(run_model(active_model, sweep_img))

                
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=qs, y=trajectories, mode='lines+markers', 
                                 line=dict(color='#00FFCC', width=4),
                                 marker=dict(size=8, color='#fff')))
                                 
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Fira Code", color="#e0e0e0"),
            xaxis=dict(autorange="reversed", title="JPEG Quality", showgrid=True, gridcolor='#333'),
            yaxis=dict(title="Confidence", showgrid=True, gridcolor='#333'),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with t3:
        st.markdown("### // ACTIVATION MAPS (GRAD-CAM)")
        st.write("Visualizing regions that strongly activated the model's decision network.")
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("ON SOURCE")
            st.image(generate_real_heatmap(active_model, orig_img, threshold=active_thresh), use_container_width=True)
        with c2:
            st.write("ON CORRUPTED")
            st.image(generate_real_heatmap(active_model, deg_img, threshold=active_thresh), use_container_width=True)

else:
    st.markdown("""
    <div style='margin-top: 50px; text-align: center;'>
        <div style='font-size: 4rem; opacity: 0.5;'>👁️</div>
        <h2 style='color: #666; font-family: "Fira Code", monospace;'>AWAITING INPUT</h2>
        <p style='color: #555;'>Upload a target subject in the control panel to initiate scan.</p>
    </div>
    """, unsafe_allow_html=True)
