# Backend API for CAM-SAM Demo
# FastAPI-based backend for processing images with CAM and SAM

import os
import sys
import io
import base64
from datetime import datetime
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add parent directory to path for imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# For peak detection
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

# SAM imports
from segment_anything import SamPredictor

# =============================================================================
# Configuration
# =============================================================================
IMAGES_DIR = "images"
os.makedirs(IMAGES_DIR, exist_ok=True)

# =============================================================================
# FastAPI App Setup
# =============================================================================
app = FastAPI(
    title="CAM-SAM Demo API",
    description="API for Class Activation Map and Segment Anything Model visualization",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Global Model Variables (Lazy Loading)
# =============================================================================
baseline_model = None
latest_model = None
sam_model = None
sam_predictor = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# =============================================================================
# Model Loading Functions
# =============================================================================

def load_baseline_model():
    """
    Load the baseline ResNet38 model.
    TODO: Replace with actual model loading code.
    """
    global baseline_model
    if baseline_model is None:
        print("Loading baseline model...")
        # TODO: Load your baseline model here
        # Example:
        from networks import resnet38d
        baseline_model = resnet38d.Net_CAM(C=20, D=256)
        baseline_model.load_state_dict(torch.load(os.path.join(ROOT, "pretrained/001net_main.pth"), map_location=device))
        baseline_model.eval()
        # baseline_model = "baseline_placeholder"
        print("Baseline model loaded.")
    return baseline_model

def load_latest_model():
    """
    Load the latest/improved ResNet38 model.
    TODO: Replace with actual model loading code.
    """
    global latest_model
    if latest_model is None:
        print("Loading latest model...")
        # TODO: Load your latest model here
        # Example:
        from networks import resnet38d
        latest_model = resnet38d.Net_CAM(C=20, D=256)
        latest_model.load_state_dict(torch.load(os.path.join(ROOT, "pretrained/019net_main.pth"), map_location=device))
        latest_model.eval()
        # latest_model = "latest_placeholder"
        print("Latest model loaded.")
    return latest_model

def load_sam_model():
    """
    Load the Segment Anything Model.
    TODO: Replace with actual SAM loading code.
    """
    global sam_model
    if sam_model is None:
        print("Loading SAM model...")
        # TODO: Load SAM here
        # Example:
        from segment_anything import sam_model_registry
        sam_model = sam_model_registry["vit_b"](checkpoint=os.path.join(ROOT, "pretrained/sam_vit_b_01ec64.pth"))
        sam_model = sam_model.to(device)
        sam_model.eval()
        # sam_model = "sam_placeholder"
        print("SAM model loaded.")
    return sam_model

# =============================================================================
# Image Processing Placeholder Functions
# =============================================================================

def preprocess_image(image: Image.Image, target_size: int = 448) -> Image.Image:
    """
    Preprocess image for model inference.
    TODO: Adjust preprocessing as needed.
    """
    # Resize while maintaining aspect ratio
    w, h = image.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    image_resized = image.resize((new_w, new_h), Image.BILINEAR)
    return image_resized

def get_cam(model, image: Image.Image, scales=[0.5, 1.0, 1.5, 2.0]) -> tuple:
    """
    Generate Multi-Scale Class Activation Map from the model.
    
    Args:
        model: The loaded ResNet38 model
        image: PIL Image
        scales: List of scales for multi-scale inference
        
    Returns:
        tuple: (cam, class_name, class_idx, score)
            - cam: numpy array (H, W) normalized to [0, 1] for top predicted class
            - class_name: name of the predicted class
            - class_idx: index of the predicted class
            - score: confidence score
    """
    # VOC class names
    VOC_CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    w, h = image.size
    
    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    model = model.to(device)
    model.eval()
    
    ms_cam_list = []
    
    for s in scales:
        target_size = (int(448 * s), int(448 * s))
        img_s = F.interpolate(img_tensor, size=target_size, mode='bilinear', align_corners=True)
        
        # Also process flipped version
        img_s_flip = torch.flip(img_s, dims=[-1])
        img_batch = torch.cat([img_s, img_s_flip], dim=0)
        
        with torch.no_grad():
            out = model(img_batch)
            cam = F.relu(out['cam'])
            
        # Resize CAM to standard size
        cam = F.interpolate(cam, size=(448, 448), mode='bilinear', align_corners=False)
        
        # Flip back the flipped CAM
        cam_flip_back = torch.flip(cam[1:2], dims=[-1])
        
        ms_cam_list.append(cam[0:1])
        ms_cam_list.append(cam_flip_back)
    
    # Combine multi-scale CAMs
    combined_cam = torch.sum(torch.stack(ms_cam_list), dim=0)  # (1, 20, H, W)
    
    # Normalize
    cam_max = torch.max(combined_cam.view(1, 20, -1), dim=-1)[0].view(1, 20, 1, 1)
    norm_ms_cam = combined_cam / (cam_max + 1e-5)
    
    # Get prediction to select top class
    with torch.no_grad():
        img_10 = F.interpolate(img_tensor, size=(448, 448), mode='bilinear', align_corners=True)
        preds = model(img_10)['pred'].view(-1)
        probs = torch.sigmoid(preds)
        top_score, top_class = torch.max(probs, dim=0)
        top_class = top_class.item()
        top_score = top_score.item()
    
    # Extract CAM for top class and resize to original image size
    final_cam = norm_ms_cam[0, top_class].cpu().numpy()
    
    import cv2
    final_cam = cv2.resize(final_cam, (w, h), interpolation=cv2.INTER_LINEAR)
    
    class_name = VOC_CLASSES[top_class]
    
    return final_cam, class_name, top_class, top_score

def get_peaks(cam: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Find peaks in CAM for SAM prompting.
    Based on model_s2c.py peak detection logic.
    
    Args:
        cam: numpy array (H, W) normalized to [0, 1]
        threshold: minimum value for valid peaks
        
    Returns:
        peaks: numpy array of shape (N, 2) with (x, y) coordinates
    """
    h, w = cam.shape
    
    # Global Maximum
    argmax_indices = np.argmax(cam)
    gy, gx = np.unravel_index(argmax_indices, cam.shape)
    peak_max = np.array([[gx, gy]])  # x, y format
    
    # Local Peaks using maximum filter
    cam_filtered = ndi.maximum_filter(cam, size=3, mode='constant')
    peaks_temp = peak_local_max(cam_filtered, min_distance=20)
    
    # Filter by threshold
    if len(peaks_temp) > 0:
        val_at_peaks = cam[peaks_temp[:, 0], peaks_temp[:, 1]]
        peaks_valid = peaks_temp[val_at_peaks > threshold]
        
        if len(peaks_valid) > 0:
            # Convert (row, col) to (x, y)
            peaks_valid_xy = np.flip(peaks_valid, axis=1)
            all_peaks = np.concatenate((peak_max, peaks_valid_xy), axis=0)
        else:
            all_peaks = peak_max
    else:
        all_peaks = peak_max
    
    return all_peaks

def get_sam_mask(sam_model_instance, image: Image.Image, points: np.ndarray) -> np.ndarray:
    """
    Generate segmentation mask using SAM with point prompts.
    Based on demo/main.py SAM inference logic.
    
    Args:
        sam_model_instance: Loaded SAM model
        image: PIL Image
        points: numpy array of shape (N, 2) with (x, y) coordinates
        
    Returns:
        mask: numpy array (H, W) binary mask
    """
    global sam_predictor
    
    # Initialize predictor if needed
    if sam_predictor is None:
        sam_predictor = SamPredictor(sam_model_instance)
    
    # Convert PIL image to numpy RGB
    image_rgb = np.array(image.convert("RGB"))
    
    # Set image for predictor
    sam_predictor.set_image(image_rgb)
    
    # Prepare point labels (all positive prompts)
    point_labels = np.ones(len(points), dtype=int)
    
    # Run SAM prediction
    masks, scores, logits = sam_predictor.predict(
        point_coords=points,
        point_labels=point_labels,
        multimask_output=True
    )
    
    # Select best mask (index 2 is typically best for multi-mask output)
    idx_max_sam = 2
    final_mask = masks[idx_max_sam]
    
    return final_mask.astype(np.uint8)

def overlay_cam_on_image(image: Image.Image, cam: np.ndarray, alpha: float = 0.4) -> Image.Image:
    """
    Overlay CAM heatmap on original image.
    
    Args:
        image: PIL Image
        cam: numpy array (H, W) normalized to [0, 1]
        alpha: transparency for heatmap overlay
        
    Returns:
        overlay: PIL Image with heatmap overlay
    """
    import cv2
    
    img_array = np.array(image.convert("RGB"))
    
    # Resize CAM to match image size
    cam_resized = cv2.resize(cam, (img_array.shape[1], img_array.shape[0]))
    
    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = cv2.addWeighted(img_array, 1 - alpha, heatmap, alpha, 0)
    
    return Image.fromarray(overlay)

def overlay_mask_on_image(image: Image.Image, mask: np.ndarray, color: tuple = (255, 0, 0), alpha: float = 0.5) -> Image.Image:
    """
    Overlay segmentation mask on original image.
    
    Args:
        image: PIL Image
        mask: numpy array (H, W) binary mask
        color: RGB color for mask overlay
        alpha: transparency for mask overlay
        
    Returns:
        overlay: PIL Image with mask overlay
    """
    import cv2
    
    img_array = np.array(image.convert("RGB"))
    
    # Resize mask to match image size
    mask_resized = cv2.resize(mask.astype(np.uint8), (img_array.shape[1], img_array.shape[0]))
    
    # Create colored mask
    colored_mask = np.zeros_like(img_array)
    colored_mask[mask_resized > 0] = color
    
    # Overlay
    overlay = img_array.copy()
    overlay[mask_resized > 0] = cv2.addWeighted(
        img_array[mask_resized > 0], 1 - alpha,
        colored_mask[mask_resized > 0], alpha, 0
    )
    
    return Image.fromarray(overlay)

# =============================================================================
# Utility Functions
# =============================================================================

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to Base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def create_grid_image(images: list, rows: int = 2, cols: int = 3, padding: int = 10) -> Image.Image:
    """
    Create a grid image from a list of PIL Images.
    
    Args:
        images: List of PIL Images (should have rows * cols images)
        rows: Number of rows
        cols: Number of columns
        padding: Padding between images
        
    Returns:
        grid: PIL Image containing the grid
    """
    # Get max dimensions
    max_w = max(img.width for img in images)
    max_h = max(img.height for img in images)
    
    # Resize all images to same size
    resized_images = [img.resize((max_w, max_h), Image.BILINEAR) for img in images]
    
    # Create grid canvas
    grid_w = cols * max_w + (cols + 1) * padding
    grid_h = rows * max_h + (rows + 1) * padding
    grid = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
    
    # Paste images
    for idx, img in enumerate(resized_images):
        row = idx // cols
        col = idx % cols
        x = padding + col * (max_w + padding)
        y = padding + row * (max_h + padding)
        grid.paste(img, (x, y))
    
    return grid

def add_labels_to_grid(grid: Image.Image, labels: list, rows: int = 2, cols: int = 3, padding: int = 10) -> Image.Image:
    """
    Add text labels to grid image.
    
    Args:
        grid: Grid PIL Image
        labels: List of label strings
        rows, cols, padding: Grid layout parameters
        
    Returns:
        labeled_grid: PIL Image with labels
    """
    from PIL import ImageDraw, ImageFont
    
    draw = ImageDraw.Draw(grid)
    
    # Try to use a nice font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Calculate cell dimensions
    cell_w = (grid.width - (cols + 1) * padding) // cols
    cell_h = (grid.height - (rows + 1) * padding) // rows
    
    for idx, label in enumerate(labels):
        row = idx // cols
        col = idx % cols
        x = padding + col * (cell_w + padding) + 5
        y = padding + row * (cell_h + padding) + 5
        
        # Draw text with background
        draw.rectangle([x-2, y-2, x + len(label) * 10, y + 22], fill=(0, 0, 0, 180))
        draw.text((x, y), label, fill=(255, 255, 255), font=font)
    
    return grid

# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "CAM-SAM Demo API is running", "status": "healthy"}

@app.post("/process-image")
async def process_image(file: UploadFile = File(...)) -> Dict:
    """
    Process an uploaded image through CAM and SAM pipelines.
    
    Returns:
        - Base64 strings for each individual result image
        - Base64 string for the concatenated grid image
        - Path to the saved grid image
    """
    try:
        # Read uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess
        processed_image = preprocess_image(image)
        
        # Load models (lazy loading)
        baseline = load_baseline_model()
        latest = load_latest_model()
        sam = load_sam_model()
        
        # =====================================================================
        # Generate CAMs and SAM masks
        # =====================================================================
        
        # Baseline CAM
        baseline_cam, baseline_class_name, baseline_class_idx, baseline_score = get_cam(baseline, processed_image)
        baseline_cam_overlay = overlay_cam_on_image(processed_image, baseline_cam)
        
        # Baseline SAM (using CAM peaks as prompts)
        baseline_peaks = get_peaks(baseline_cam, threshold=0.5)
        baseline_sam_mask = get_sam_mask(sam, processed_image, baseline_peaks)
        baseline_sam_overlay = overlay_mask_on_image(processed_image, baseline_sam_mask, color=(0, 255, 0))
        
        # Latest CAM
        latest_cam, latest_class_name, latest_class_idx, latest_score = get_cam(latest, processed_image)
        latest_cam_overlay = overlay_cam_on_image(processed_image, latest_cam)
        
        # Latest SAM (using CAM peaks as prompts)
        latest_peaks = get_peaks(latest_cam, threshold=0.5)
        latest_sam_mask = get_sam_mask(sam, processed_image, latest_peaks)
        latest_sam_overlay = overlay_mask_on_image(processed_image, latest_sam_mask, color=(0, 0, 255))
        
        # =====================================================================
        # Create grid image
        # =====================================================================
        
        # Grid layout:
        # Row 1: Original | Baseline CAM | Baseline SAM
        # Row 2: Original | Latest CAM   | Latest SAM
        
        grid_images = [
            processed_image.copy(),      # Row 1, Col 1: Original
            baseline_cam_overlay,        # Row 1, Col 2: Baseline CAM
            baseline_sam_overlay,        # Row 1, Col 3: Baseline SAM
            processed_image.copy(),      # Row 2, Col 1: Original
            latest_cam_overlay,          # Row 2, Col 2: Latest CAM
            latest_sam_overlay,          # Row 2, Col 3: Latest SAM
        ]
        
        grid_labels = [
            "Original", 
            f"Baseline: {baseline_class_name}", 
            "Baseline SAM",
            "Original", 
            f"Latest: {latest_class_name}", 
            "Latest SAM"
        ]
        
        grid = create_grid_image(grid_images, rows=2, cols=3)
        grid = add_labels_to_grid(grid, grid_labels, rows=2, cols=3)
        
        # =====================================================================
        # Save grid image
        # =====================================================================
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"result_{timestamp}.png"
        filepath = os.path.join(IMAGES_DIR, filename)
        grid.save(filepath)
        print(f"Saved grid image to: {filepath}")
        
        # =====================================================================
        # Prepare response
        # =====================================================================
        
        response = {
            "original": image_to_base64(processed_image),
            "baseline_cam": image_to_base64(baseline_cam_overlay),
            "baseline_sam": image_to_base64(baseline_sam_overlay),
            "latest_cam": image_to_base64(latest_cam_overlay),
            "latest_sam": image_to_base64(latest_sam_overlay),
            "grid": image_to_base64(grid),
            "grid_path": filepath,
            "baseline_prediction": {
                "class_name": baseline_class_name,
                "class_idx": baseline_class_idx,
                "score": baseline_score
            },
            "latest_prediction": {
                "class_name": latest_class_name,
                "class_idx": latest_class_idx,
                "score": latest_score
            },
            "message": "Image processed successfully"
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_history():
    """
    Get list of all saved grid images.
    
    Returns:
        List of image filenames and their paths
    """
    try:
        images = []
        if os.path.exists(IMAGES_DIR):
            for filename in sorted(os.listdir(IMAGES_DIR), reverse=True):
                if filename.endswith((".png", ".jpg", ".jpeg")):
                    filepath = os.path.join(IMAGES_DIR, filename)
                    images.append({
                        "filename": filename,
                        "path": filepath,
                        "url": f"/images/{filename}"
                    })
        
        return {"images": images, "count": len(images)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
