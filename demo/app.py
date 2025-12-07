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

# Title
st.title("🔍 Class Activation Mapping (CAM) & Segment Anything Model (SAM)")
st.markdown("Upload an image to visualize CAM and SAM segmentation")

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

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=['png', 'jpg', 'jpeg'],
    help="Upload an image to process with CAM and SAM"
)

# Process button
if uploaded_file is not None:
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
            st.image(original_img, use_column_width =True)
        else:
            st.warning("Original image not available")
    
    with col2:
        st.subheader("🔥 CAM Visualization")
        if 'cam' in results:
            cam_img = decode_base64_image(results['cam'])
            st.image(cam_img, use_column_width =True)
            st.caption("Class Activation Map shows which regions the model focuses on")
        else:
            st.warning("CAM image not available")
    
    with col3:
        st.subheader("🎨 SAM Segmentation")
        if 'sam' in results:
            sam_img = decode_base64_image(results['sam'])
            st.image(sam_img, use_column_width =True)
            st.caption("Segment Anything Model segmentation (Coming soon)")
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
