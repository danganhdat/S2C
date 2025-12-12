from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = bool
import cv2
import io
import base64
from typing import Dict
import torchvision.models as models
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import torchvision.transforms as T
from networks.resnet38d import Net as ResNet38D
from segment_anything import sam_model_registry, SamPredictor
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

app = FastAPI(title="CAM-SAM API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
resnet_model = None
feature_maps = []
gradients = []

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Try to load S2C's ResNet38 CAM network (net_main) 
try:
    from networks import resnet38d
except Exception as e:
    print("ERROR importing networks.resnet38d:", e)
    resnet_model = None
else:
    # create a net_main similar to model_WSSS __init__ usage
    # choose C and D consistent with repo (VOC20)
    C = 20   # number of classes for VOC
    D = 256  # feature dimension used in repo
    try:
        net_main = resnet38d.Net_CAM(C=C, D=D)   
    except Exception as e:
        print("ERROR instantiating Net_CAM:", e)
        net_main = None

    # load mxnet-converted weights 
    pretrained_params = "../pretrained/011net_main.pth"
    if net_main is not None and os.path.exists(pretrained_params):
        try:
            print("Loading converted resnet38 params...")
            state_dict = resnet38d.convert_mxnet_to_torch(pretrained_params)
            net_main.load_state_dict(state_dict, strict=False)
            print("Loaded converted params.")
        except Exception as e:
            print("Warning: failed to load converted params:", e)
    else:
        if net_main is not None:
            print("Pretrained resnet_38d.params not found — using random init for demo.")

    if net_main is not None:
        net_main = net_main.to(device)
        net_main.eval()
    resnet_model = net_main


def get_ms_cam(model, image, scales=[0.5, 1.0, 1.5, 2.0]):
    W_org, H_org = image.size

    img_tensor = T.ToTensor()(image).unsqueeze(0).to(device)
    img_tensor = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_tensor)

    ms_cam_list = []
    
    for s in scales:
        target_size = (int(448 * s), int(448 * s))
        img_s = F.interpolate(img_tensor, size=target_size, mode='bilinear', align_corners=True)
        
        img_s_flip = torch.flip(img_s, dims=[-1])
        
        img_batch = torch.cat([img_s, img_s_flip], dim=0)
        
        with torch.no_grad():
            out = model(img_batch)
            cam = F.relu(out['cam']) 
            
        cam = F.interpolate(cam, size=(448, 448), mode='bilinear', align_corners=False)

        cam_flip_back = torch.flip(cam[1:2], dims=[-1])
        
        ms_cam_list.append(cam[0:1])      
        ms_cam_list.append(cam_flip_back)  
        
    combined_cam = torch.sum(torch.stack(ms_cam_list), dim=0) 
    
    cam_max = torch.max(combined_cam.view(1, 20, -1), dim=-1)[0].view(1, 20, 1, 1)
    norm_ms_cam = combined_cam / (cam_max + 1e-5)
    
    return norm_ms_cam


def overlay_cam_on_image(image: Image.Image, cam: np.ndarray):
    # Convert PIL to numpy
    img_array = np.array(image)
    
    # Convert grayscale to RGB if needed
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Apply colormap to CAM
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = cv2.addWeighted(img_array, 0.6, heatmap, 0.4, 0)
    
    return Image.fromarray(overlay)

def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def load_sam_model(checkpoint_path: str, device="cpu"):
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    sam.to(device)
    sam.eval()
    return sam

@app.get("/")
async def root():
    return {"message": "CAM-SAM API is running"}

@app.post("/process")
async def process_image(file: UploadFile = File(...)) -> Dict[str, str]:
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        # Check model exists
        global resnet_model
        if resnet_model is None:
            raise RuntimeError("resnet_model is not initialized. Check server logs for import/initialization errors.")
        
        norm_ms_cam = get_ms_cam(resnet_model, image)

        img_10 = T.Compose([T.Resize((448, 448)), T.ToTensor(), 
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(image).unsqueeze(0).to(device)
        
        VOC_CLASSES = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]

        with torch.no_grad():
            preds = resnet_model(img_10)['pred'].view(-1)

        topc = preds.argmax().item()

        top_score, top_idx = torch.max(preds, dim=0)
        top_class_name = VOC_CLASSES[top_idx.item()]

        print(f"Model đang nhìn thấy: {top_class_name} (Score: {top_score.item():.4f})")
    
        final_cam = norm_ms_cam[0, topc].cpu().numpy()
        
        cam_resized = cv2.resize(final_cam, (image.width, image.height), interpolation=cv2.INTER_LINEAR)
        
        # cam_resized[cam_resized < 0.20] = 0 
        
        cam_image = overlay_cam_on_image(image, cam_resized)
        
        # Implement SAM
        net_sam = load_sam_model("../pretrained/sam_vit_b_01ec64.pth", device=device)
        predictor = SamPredictor(net_sam)

        image_rgb = np.array(image.convert("RGB"))
        predictor.set_image(image_rgb)

        # -- Find points prompt --
        cam_target_np = final_cam 
        h_org, w_org = cam_target_np.shape

        # Global Max
        argmax_indices = np.argmax(cam_target_np)
        gy, gx = np.unravel_index(argmax_indices, cam_target_np.shape)
        peak_max = np.array([[gx, gy]])

        # Local Peaks
        cam_filtered = ndi.maximum_filter(cam_target_np, size=3, mode='constant')
        peaks_temp = peak_local_max(cam_filtered, min_distance=20)

        th_multi = 0.5 
        peaks_valid = peaks_temp[cam_target_np[peaks_temp[:,0], peaks_temp[:,1]] > th_multi]

        if len(peaks_valid) > 0:
            peaks_valid_xy = np.flip(peaks_valid, axis=1)
            point_coords = np.concatenate((peak_max, peaks_valid_xy), axis=0)
        else:
            point_coords = peak_max

        point_labels = np.ones(len(point_coords), dtype=int)

        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )

        idx_max_sam = 2
        final_mask = masks[idx_max_sam]

        # Morphology
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # mask_cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        # mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)

        # Gaussian Blur
        # mask_blurred = cv2.GaussianBlur(mask_cleaned, (5, 5), 0)

        # h, w = final_mask.shape
        # seg_map_bgr = np.zeros((h, w, 3), dtype=np.uint8)

        # seg_map_bgr[final_mask > 0] = [0, 0, 255]

        # _, buffer = cv2.imencode(".png", seg_map_bgr)
        # sam_b64 = base64.b64encode(buffer).decode()

        mask_np = final_mask.astype(np.uint8) * 255
        mask_rgba = cv2.applyColorMap(mask_np, cv2.COLORMAP_JET)
        _, mask_png = cv2.imencode(".png", mask_rgba)
        sam_b64 = base64.b64encode(mask_png).decode()
        
        # Convert to base64
        original_b64 = image_to_base64(image)
        cam_b64 = image_to_base64(cam_image)
        
        return JSONResponse(content={
            "original": original_b64,
            "cam": cam_b64,
            "sam": sam_b64,
            "predicted_class": topc,
            "message": "CAM generated successfully. SAM will be implemented next."
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)