import streamlit as st
import requests
import base64
from PIL import Image
import io

st.set_page_config(
    page_title="CAM-SAM Visualization",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Class Activation Mapping (CAM) & Segment Anything Model (SAM)")
st.markdown("Upload an image to visualize CAM and SAM segmentation")

API_URL = "http://localhost:8000/process"

def decode_base64_image(base64_str):
    try:
        img_bytes = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        st.error(f"Error decoding image: {e}")
        return None

if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'results' not in st.session_state:
    st.session_state.results = None

uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=['png', 'jpg', 'jpeg'],
    help="Upload an image to process with CAM and SAM"
)

if uploaded_file is not None:
    if st.button("🚀 Process Image", type="primary"):
        with st.spinner("Processing image... Please wait..."):
            try:
                files = {"file": uploaded_file.getvalue()}
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    st.session_state.results = response.json()
                    st.session_state.processed = True
                    st.success("Image processed successfully!")
                else:
                    st.error(f"Error from Backend: {response.status_code} - {response.text}")
                    st.session_state.processed = False
                    
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend. Make sure 'main.py' is running!")
                st.session_state.processed = False
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.processed = False

if st.session_state.processed and st.session_state.results:
    results = st.session_state.results
    
    if 'predicted_class' in results:
        st.info(f"🎯 Predicted Class Index: {results['predicted_class']}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📷 Original")
        if 'original' in results:
            img = decode_base64_image(results['original'])
            if img: 
                st.image(img, use_container_width=True)
    
    with col2:
        st.subheader("🔥 CAM (ResNet38d)")
        if 'cam' in results:
            img = decode_base64_image(results['cam'])
            if img: 
                st.image(img, use_container_width=True)
        else:
            st.warning("CAM data missing")
    
    with col3:
        st.subheader("🎨 SAM (Segmentation)")
        if 'sam' in results:
            img = decode_base64_image(results['sam'])
            if img: 
                st.image(img, use_container_width=True)
        else:
            st.warning("SAM not implemented in backend yet")
    
    st.markdown("---")
