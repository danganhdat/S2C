import streamlit as st
import requests
import base64
from PIL import Image
import io

# Page config
st.set_page_config(
    page_title="CAM-SAM Visualization",
    page_icon="🔍",
    layout="wide"
)

# GLOBAL CSS STYLES
# st.markdown(
#     """
#     <style>
#     /* Main title */
#     .title {
#         text-align: center;
#         font-size: 42px !important;
#         font-weight: 700 !important;
#         margin-bottom: -10px;
#     }

#     /* Subtitle */
#     .subtitle {
#         text-align: center;
#         color: #888;
#         font-size: 18px !important;
#         margin-bottom: 30px;
#     }

#     /* Cards */
#     .result-card {
#         background: #ffffff10;
#         padding: 20px;
#         border-radius: 12px;
#         backdrop-filter: blur(8px);
#         border: 1px solid rgba(255,255,255,0.1);
#         transition: 0.3s;
#     }
#     .result-card:hover {
#         transform: translateY(-3px);
#         border-color: #7f8cff;
#     }

#     /* Footer */
#     .footer {
#         text-align: center;
#         color: #666;
#         padding-top: 20px;
#         font-size: 14px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# Title
st.markdown("<div class='title'>🔍 CAM–SAM Visualization Tool</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>From-SAM-to-CAMs    •    Class Activation Map    •    Segment Anything Model</div>", unsafe_allow_html=True)

# Backend API URL
API_URL = "http://localhost:8000/process"

def decode_base64_image(base64_str):
    """Decode base64 string to PIL Image"""
    img_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_bytes))

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'results' not in st.session_state:
    st.session_state.results = None

def get_image_base64(file):
    if file is None:
        return ""
    file_bytes = file.getvalue()
    encoded = base64.b64encode(file_bytes).decode()
    return f"data:image/png;base64,{encoded}"

st.markdown("""
<style>
:root {
    --box-height: 240px;
}
            
.upload-box {
    border: 2px dashed #6c63ff;
    border-radius: 15px;
    height: var(--box-height);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    padding: 20px;
    cursor: pointer;
    transition: 0.25s;
    margin-top: calc(var(--box-height) * -1); 
    position: relative;
    z-index: 1;
}

.upload-box:hover {
    background: rgba(108, 99, 255, 0.08);
    border-color: #8d86ff;
}

/* File Uploader */
[data-testid="stFileUploader"] {
width: 100%;
    height: var(--box-height);
    position: relative;
    z-index: 99;
    opacity: 0; 
}

[data-testid="stFileUploader"] section {
    min-height: var(--box-height); 
    padding: 0;
}
            
[data-testid="stFileUploader"] section > div {
    display: none;
}
            
.preview-img {
    max-height: 150px;
    max-width: 90%;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin-bottom: 10px;
    object-fit: contain;
}

.file-name {
    font-size: 14px;
    color: #6c63ff;
    font-weight: bold;
    word-break: break-all;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="upload-box">
    <div style="font-size: 50px;">📁</div>
    <div style="font-size: 20px;"><b>Drag & Drop your image here</b></div>
    <div style="font-size: 14px; color: #b5b5b5;">or Click to select a file</div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload",
    type=['png', 'jpg', 'jpeg'],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    img_src = get_image_base64(uploaded_file)
    file_name = uploaded_file.name
    
    # Sửa lỗi: Đẩy sát lề trái (xóa indentation)
    box_content = f"""
<img src="{img_src}" class="preview-img">
<div class="file-name">✅ {file_name}</div>
"""
else:
    # Sửa lỗi: Đẩy sát lề trái (xóa indentation)
    box_content = """
<div style="font-size: 50px;">📁</div>
<div style="font-size: 20px;"><b>Drag & Drop your image here</b></div>
<div style="font-size: 14px; color: #b5b5b5;">or Click to select a file</div>
"""

st.markdown(f"""
<div class="upload-box">
{box_content}
</div>
""", unsafe_allow_html=True)

# Process button
if uploaded_file is not None:
    img_src = get_image_base64(uploaded_file)
    file_name = uploaded_file.name
    
    html_content = f"""
        <img src="{img_src}" class="preview-img">
        <div class="file-name">✅ {file_name}</div>
        <div style="font-size: 12px; color: #b5b5b5; margin-top:5px;">Click to change image</div>
    """
    if st.button("🚀 Process Image", type="primary"):
        with st.spinner("Processing image... This may take a few seconds"):
            try:
                # Send request to backend
                files = {"file": uploaded_file.getvalue()}
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    st.session_state.results = response.json()
                    st.session_state.processed = True
                    st.success("✅ Image processed successfully!")
                else:
                    st.error(f"❌ Error: {response.status_code} - {response.text}")
                    st.session_state.processed = False
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend API. Please ensure FastAPI server is running on http://localhost:8000")
                st.session_state.processed = False
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.session_state.processed = False

# Display results
if st.session_state.processed and st.session_state.results:
    results = st.session_state.results
    
    # Show predicted class if available
    if 'predicted_class' in results:
        st.info(f"🎯 Predicted Class: {results['predicted_class']}")
    
    if 'message' in results:
        st.info(f"ℹ️ {results['message']}")
    
    # Create three columns for images
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📷 Original Image")
        if 'original' in results:
            original_img = decode_base64_image(results['original'])
            st.image(original_img, use_container_width =True)
        else:
            st.warning("Original image not available")
    
    with col2:
        st.subheader("🔥 CAM Visualization")
        if 'cam' in results:
            cam_img = decode_base64_image(results['cam'])
            st.image(cam_img, use_container_width =True)
            st.caption("Class Activation Map shows which regions the model focuses on")
        else:
            st.warning("CAM image not available")
    
    with col3:
        st.subheader("🎨 SAM Segmentation")
        if 'sam' in results:
            sam_img = decode_base64_image(results['sam'])
            st.image(sam_img, use_container_width =True)
            st.caption("Segment Anything Model segmentation")
        else:
            st.warning("SAM image not available")
    
    # Download buttons
    st.markdown("---")
    st.subheader("💾 Download Results")
    
    col_d1, col_d2, col_d3 = st.columns(3)
    
    with col_d1:
        if 'original' in results:
            st.download_button(
                label="Download Original",
                data=base64.b64decode(results['original']),
                file_name="original.png",
                mime="image/png"
            )
    
    with col_d2:
        if 'cam' in results:
            st.download_button(
                label="Download CAM",
                data=base64.b64decode(results['cam']),
                file_name="cam_result.png",
                mime="image/png"
            )
    
    with col_d3:
        if 'sam' in results:
            st.download_button(
                label="Download SAM",
                data=base64.b64decode(results['sam']),
                file_name="sam_result.png",
                mime="image/png"
            )

else:
    # Show placeholder when no image is processed
    st.info("👆 Upload an image and click 'Process Image' to get started")
    
    # Show example/instructions
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. **Upload an image** using the file uploader above
        2. **Click 'Process Image'** to run CAM and SAM analysis
        3. **View results** in three columns:
           - Original Image
           - CAM Visualization (shows important regions for classification)
           - SAM Segmentation (automatic segmentation)
        4. **Download results** using the download buttons
        
        **Note:** Make sure the FastAPI backend is running on port 8000
        ```bash
        python main.py
        ```
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>CAM-SAM Visualization System | "
    "Built with Streamlit & FastAPI</div>",
    unsafe_allow_html=True
)
