# Frontend for CAM-SAM Demo
# Streamlit-based frontend for visualizing CAM and SAM results

import os
import io
import base64
import requests
from datetime import datetime

import streamlit as st
from PIL import Image

# =============================================================================
# Configuration
# =============================================================================
BACKEND_URL = "http://localhost:8000"
IMAGES_DIR = "images"

# =============================================================================
# Page Config
# =============================================================================
st.set_page_config(
    page_title="CAM-SAM Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Custom CSS
# =============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #333;
        margin: 1rem 0;
    }
    .stImage {
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .history-card {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Utility Functions
# =============================================================================

def base64_to_image(base64_str: str) -> Image.Image:
    """Convert Base64 string to PIL Image."""
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data))

def get_backend_status() -> bool:
    """Check if backend is running."""
    try:
        response = requests.get(f"{BACKEND_URL}/", timeout=5)
        return response.status_code == 200
    except:
        return False

def process_image(image_bytes: bytes) -> dict:
    """Send image to backend for processing."""
    try:
        files = {"file": ("image.png", image_bytes, "image/png")}
        response = requests.post(f"{BACKEND_URL}/process-image", files=files, timeout=120)
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": response.text}
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_history_from_folder() -> list:
    """Get list of saved images from local folder."""
    images = []
    if os.path.exists(IMAGES_DIR):
        for filename in sorted(os.listdir(IMAGES_DIR), reverse=True):
            if filename.endswith((".png", ".jpg", ".jpeg")):
                filepath = os.path.join(IMAGES_DIR, filename)
                images.append({
                    "filename": filename,
                    "path": filepath
                })
    return images

# =============================================================================
# Sidebar Navigation
# =============================================================================

with st.sidebar:
    st.image("https://antigravity.google/assets/image/blog/blog-feature-introducing-google-antigravity.png", width=150)
    st.markdown("---")
    
    page = st.radio(
        "📌 Navigation",
        ["🎯 Demo", "📚 History"],
        index=0,
        key="navigation"
    )
    
    st.markdown("---")
    
    # Backend status indicator
    backend_status = get_backend_status()
    if backend_status:
        st.success("✅ Backend Connected")
    else:
        st.error("❌ Backend Offline")
        st.info("Start backend with:\n```\npython backend.py\n```")
    
    st.markdown("---")
    st.markdown("""
    ### About
    This demo visualizes:
    - **CAM**: Class Activation Maps
    - **SAM**: Segment Anything Model
    
    Compare baseline vs latest model results.
    """)

# =============================================================================
# Demo Page
# =============================================================================

if page == "🎯 Demo":
    st.markdown('<h1 class="main-header">🔍 CAM-SAM Visualization Demo</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload an image to compare baseline and latest model results</p>', unsafe_allow_html=True)
    
    # File uploader
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            key="image_uploader"
        )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        st.markdown("---")
        st.markdown('<p class="result-header">📷 Uploaded Image</p>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Original Image", width='stretch')
        
        # Process button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            process_btn = st.button("🚀 Process Image", type="primary", width='stretch')
        
        if process_btn:
            if not backend_status:
                st.error("❌ Backend is not running. Please start the backend first.")
            else:
                with st.spinner("🔄 Processing image... This may take a moment."):
                    # Convert image to bytes
                    img_bytes = io.BytesIO()
                    image.save(img_bytes, format="PNG")
                    img_bytes = img_bytes.getvalue()
                    
                    # Send to backend
                    result = process_image(img_bytes)
                
                if result["success"]:
                    data = result["data"]
                    
                    st.success("✅ Processing complete!")
                    st.markdown("---")
                    
                    # ==========================================================
                    # Display Results in 2x3 Grid
                    # ==========================================================
                    
                    st.markdown('<p class="result-header">📊 Results Comparison</p>', unsafe_allow_html=True)
                    
                    # Row 1: Baseline
                    st.markdown("### 🔹 Baseline Model")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.image(
                            base64_to_image(data["original"]),
                            caption="Original",
                            width='stretch'
                        )
                    with col2:
                        st.image(
                            base64_to_image(data["baseline_cam"]),
                            caption="Baseline CAM",
                            width='stretch'
                        )
                    with col3:
                        st.image(
                            base64_to_image(data["baseline_sam"]),
                            caption="Baseline SAM Enhanced",
                            width='stretch'
                        )
                    
                    st.markdown("---")
                    
                    # Row 2: Latest
                    st.markdown("### 🔸 Latest Model")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.image(
                            base64_to_image(data["original"]),
                            caption="Original",
                            width='stretch'
                        )
                    with col2:
                        st.image(
                            base64_to_image(data["latest_cam"]),
                            caption="Latest CAM",
                            width='stretch'
                        )
                    with col3:
                        st.image(
                            base64_to_image(data["latest_sam"]),
                            caption="Latest SAM Enhanced",
                            width='stretch'
                        )
                    
                    st.markdown("---")
                    
                    # Grid image
                    st.markdown('<p class="result-header">📋 Combined Grid</p>', unsafe_allow_html=True)
                    st.image(
                        base64_to_image(data["grid"]),
                        caption="Combined Comparison Grid",
                        width='stretch'
                    )
                    
                    # Download button
                    grid_bytes = io.BytesIO()
                    base64_to_image(data["grid"]).save(grid_bytes, format="PNG")
                    
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col2:
                        st.download_button(
                            label="📥 Download Grid Image",
                            data=grid_bytes.getvalue(),
                            file_name=f"cam_sam_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png",
                            width='stretch'
                        )
                    
                    st.info(f"💾 Grid image also saved to: `{data.get('grid_path', 'images/')}`")
                    
                else:
                    st.error(f"❌ Processing failed: {result['error']}")

# =============================================================================
# History Page
# =============================================================================

elif page == "📚 History":
    st.markdown('<h1 class="main-header">📚 Processing History</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">View previously processed images</p>', unsafe_allow_html=True)
    
    # Get history
    history = get_history_from_folder()
    
    if len(history) == 0:
        st.info("📭 No images found in history. Process some images first!")
    else:
        st.markdown(f"**Found {len(history)} image(s)**")
        st.markdown("---")
        
        # Display in gallery format (3 columns)
        cols = st.columns(3)
        
        for idx, item in enumerate(history):
            col = cols[idx % 3]
            
            with col:
                try:
                    img = Image.open(item["path"])
                    st.image(img, caption=item["filename"], width='stretch')
                    
                    # Download button for each
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format="PNG")
                    
                    st.download_button(
                        label="📥 Download",
                        data=img_bytes.getvalue(),
                        file_name=item["filename"],
                        mime="image/png",
                        key=f"download_{idx}",
                        width='stretch'
                    )
                    st.markdown("---")
                    
                except Exception as e:
                    st.error(f"Error loading {item['filename']}: {e}")
        
        # Clear history button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🗑️ Clear History", type="secondary", width='stretch'):
                st.warning("⚠️ This will delete all saved images!")
                if st.button("Confirm Delete", type="primary"):
                    import shutil
                    if os.path.exists(IMAGES_DIR):
                        shutil.rmtree(IMAGES_DIR)
                        os.makedirs(IMAGES_DIR, exist_ok=True)
                    st.success("History cleared!")
                    st.rerun()

# =============================================================================
# Footer
# =============================================================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888; font-size: 0.9rem;">
        CAM-SAM Demo | Built with Streamlit ❤️
    </div>
    """,
    unsafe_allow_html=True
)
