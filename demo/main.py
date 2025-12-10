from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
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

    # try to load mxnet-converted weights if available (optional)
    pretrained_params = os.path.join(ROOT, "pretrained", "resnet_38d.params")
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

        # Preprocess similar to repo (use 448 as used in S2C)
        preprocess = T.Compose([
            T.Resize((448, 448)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        img_tensor = preprocess(image).unsqueeze(0).to(device)

        # Forward
        with torch.no_grad():
            out = resnet_model(img_tensor)

        if isinstance(out, dict):
            # cam: (1, C, h, w)
            cam = out.get('cam', None)

            # prediction/logits often stored as 'pred' or 'pred_main'
            if 'pred' in out:
                preds = out['pred']
            elif 'pred_main' in out:
                preds = out['pred_main']
            elif 'cls' in out:
                preds = out['cls']
            else:
                raise RuntimeError(f"Cannot find prediction key in model output. Keys = {list(out.keys())}")
        else:
            raise RuntimeError("Unexpected model output type: {}".format(type(out)))

        if cam is None or preds is None:
            raise RuntimeError("Model output missing 'cam' or 'pred' keys. Got keys: {}".format(list(out.keys())))
        
        # get class scores -> convert to vector
        # preds may be shape (1, C) or (1, C, 1,1)
        preds = preds.view(-1)              # now shape = (C,)
        topc = preds.argmax().item()        # scalar int
        
        # cam for that class (1, C, h, w)
        cam_cls = cam[0, topc].cpu().numpy()

        # normalize cam to [0,1]
        cam_cls = cam_cls - cam_cls.min()
        if cam_cls.max() > 0:
            cam_cls = cam_cls / (cam_cls.max() + 1e-8)

        # resize cam to original PIL size
        cam_resized = cv2.resize(cam_cls, (image.width, image.height))
        
        # overlay and encode
        cam_image = overlay_cam_on_image(image, cam_resized)
        
        # Implement SAM
        net_sam = load_sam_model("sam_vit_b_01ec64.pth", device=device)
        predictor = SamPredictor(net_sam)

        # find local peaks
        cam_heatmap_raw = cam_cls.copy()
        y, x = np.unravel_index(np.argmax(cam_heatmap_raw), cam_heatmap_raw.shape)
        point_coords = np.array([[x, y]])
        point_labels = np.array([1])  # foreground

        predictor.set_image(np.array(image))

        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )

        mask_np = masks[0].astype(np.uint8) * 255
        mask_rgba = cv2.applyColorMap(mask_np, cv2.COLORMAP_JET)
        _, mask_png = cv2.imencode(".png", mask_rgba)
        sam_b64 = base64.b64encode(mask_png).decode()
        
        # Convert to base64
        original_b64 = image_to_base64(image)
        cam_b64 = image_to_base64(cam_image)
        # sam_b64 = image_to_base64(sam_image)
        
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