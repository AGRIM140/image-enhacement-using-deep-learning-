import streamlit as st
from PIL import Image
import yaml
import torch
from pathlib import Path
import tempfile
import os
import sys
import numpy as np

# Add project root to path for Streamlit Cloud
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import project modules
from tools.model_runner import ModelRunner

# Page configuration
st.set_page_config(
    page_title="GAN Image Enhancer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .model-card {
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #667eea;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_runner' not in st.session_state:
    st.session_state.model_runner = None
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'current_checkpoint' not in st.session_state:
    st.session_state.current_checkpoint = None

# Header
st.markdown('<h1 class="main-header">🎨 GAN Image Enhancer</h1>', unsafe_allow_html=True)
st.markdown("### Enhance your images using AI-powered GAN models")

# Sidebar for model selection
with st.sidebar:
    st.header("⚙️ Model Configuration")
    
    # Model selection
    model_type = st.selectbox(
        "Select Model",
        ["srgan", "noise2noise", "deblurgan"],
        help="Choose the model type: SRGAN (super-resolution), Noise2Noise (denoising), or DeblurGAN (deblurring)"
    )
    
    # Model descriptions
    model_descriptions = {
        "srgan": "**SRGAN**: Super-Resolution GAN - Upscales low-resolution images to high resolution",
        "noise2noise": "**Noise2Noise**: Removes noise from noisy images",
        "deblurgan": "**DeblurGAN**: Removes blur and sharpens images"
    }
    st.markdown(model_descriptions[model_type])
    
    st.divider()
    
    # Checkpoint selection
    st.subheader("📦 Model Checkpoint")
    checkpoint_dir = Path(f"checkpoints/{model_type}_model")
    
    # Auto-detect checkpoints
    checkpoint_options = ["Use default (if available)"]
    checkpoint_paths = {"Use default (if available)": None}
    
    if checkpoint_dir.exists():
        best_checkpoint = checkpoint_dir / "best.pth"
        latest_checkpoint = checkpoint_dir / "latest.pth"
        
        if best_checkpoint.exists():
            checkpoint_options.append("Best Model (best.pth)")
            checkpoint_paths["Best Model (best.pth)"] = str(best_checkpoint)
        
        if latest_checkpoint.exists():
            checkpoint_options.append("Latest Model (latest.pth)")
            checkpoint_paths["Latest Model (latest.pth)"] = str(latest_checkpoint)
    
    selected_checkpoint = st.selectbox(
        "Select Checkpoint",
        checkpoint_options,
        help="Choose which model checkpoint to use"
    )
    
    checkpoint_path = checkpoint_paths[selected_checkpoint]
    
    if checkpoint_path:
        st.success(f"✅ Checkpoint found: {Path(checkpoint_path).name}")
    else:
        st.warning("⚠️ No checkpoint selected - model will use random weights")
    
    st.divider()
    
    # Model info
    st.subheader("ℹ️ Model Information")
    if checkpoint_path and Path(checkpoint_path).exists():
        file_size = Path(checkpoint_path).stat().st_size / (1024 * 1024)
        st.metric("Checkpoint Size", f"{file_size:.2f} MB")
        
        # Try to load config
        config_file = checkpoint_dir / "config.yaml"
        if config_file.exists():
            try:
                cfg = yaml.safe_load(open(config_file))
                if 'training' in cfg:
                    st.caption(f"Epochs: {cfg['training'].get('num_epochs', 'N/A')}")
            except:
                pass

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image to enhance"
    )
    
    if uploaded_file is not None:
        # Display input image
        input_image = Image.open(uploaded_file).convert("RGB")
        st.image(input_image, caption="Input Image", use_container_width=True)
        
        # Image info
        st.caption(f"Size: {input_image.size[0]} x {input_image.size[1]} pixels")
        st.caption(f"Format: {uploaded_file.type}")

with col2:
    st.header("📥 Enhanced Output")
    
    if uploaded_file is not None:
        if st.button("🚀 Enhance Image", type="primary", use_container_width=True):
            with st.spinner("Processing image..."):
                try:
                    # Load config
                    config_path = f"configs/{model_type}.yaml"
                    if Path(config_path).exists():
                        cfg = yaml.safe_load(open(config_path))
                    else:
                        st.error(f"Config file not found: {config_path}")
                        st.stop()
                    
                    # Initialize model runner
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    
                    # Check if model needs to be reloaded
                    if (st.session_state.current_model != model_type or 
                        st.session_state.current_checkpoint != checkpoint_path):
                        st.session_state.model_runner = ModelRunner(model_type, cfg, device)
                        if checkpoint_path and Path(checkpoint_path).exists():
                            st.session_state.model_runner.load_checkpoint(checkpoint_path)
                            st.session_state.current_checkpoint = checkpoint_path
                        st.session_state.current_model = model_type
                    
                    # Process image
                    tmp_dir = Path(tempfile.mkdtemp())
                    in_path = tmp_dir / uploaded_file.name
                    out_path = tmp_dir / f"enhanced_{uploaded_file.name}"
                    
                    input_image.save(in_path)
                    st.session_state.model_runner.enhance_and_save(str(in_path), str(out_path))
                    
                    # Load and display output
                    output_image = Image.open(out_path)
                    st.image(output_image, caption="Enhanced Image", use_container_width=True)
                    
                    # Show success and image statistics
                    st.success("✅ Image enhanced successfully!")
                    
                    # Image statistics
                    with st.expander("📊 Image Statistics"):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Input Size", f"{input_image.size[0]} x {input_image.size[1]}")
                        with col_b:
                            st.metric("Output Size", f"{output_image.size[0]} x {output_image.size[1]}")
                    
                    # Download button
                    with open(out_path, "rb") as file:
                        st.download_button(
                            label="📥 Download Enhanced Image",
                            data=file,
                            file_name=f"enhanced_{uploaded_file.name}",
                            mime="image/png",
                            use_container_width=True
                        )
                    
                except Exception as e:
                    st.error(f"❌ Error processing image: {str(e)}")
                    st.exception(e)
    else:
        st.info("👆 Upload an image to see the enhanced result here")

# Footer with model comparison
st.divider()
st.header("📈 Model Performance")

# Try to load metrics if available
metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

models_info = [
    ("SRGAN", "srgan", "Super-Resolution"),
    ("Noise2Noise", "noise2noise", "Denoising"),
    ("DeblurGAN", "deblurgan", "Deblurring")
]

for i, (name, model_key, desc) in enumerate(models_info):
    with [metrics_col1, metrics_col2, metrics_col3][i]:
        st.subheader(name)
        st.caption(desc)
        
        checkpoint_path = Path(f"checkpoints/{model_key}_model/best.pth")
        if checkpoint_path.exists():
            st.success("✅ Trained")
            file_size = checkpoint_path.stat().st_size / (1024 * 1024)
            st.caption(f"Size: {file_size:.2f} MB")
        else:
            st.warning("⏳ Training...")

# Instructions
with st.expander("ℹ️ How to Use"):
    st.markdown("""
    1. **Select a Model**: Choose from the sidebar (SRGAN, Noise2Noise, or DeblurGAN)
    2. **Select Checkpoint**: Choose which trained model to use (Best or Latest)
    3. **Upload Image**: Click "Browse files" to upload your image
    4. **Enhance**: Click the "Enhance Image" button
    5. **Download**: Download your enhanced image
    
    **Tips:**
    - Use "Best Model" checkpoint for best quality
    - SRGAN works best with low-resolution images
    - Noise2Noise works best with noisy images
    - DeblurGAN works best with blurred images
    """)

# About section
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>Powered by PyTorch GAN Models | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
