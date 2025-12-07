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

class ResNet38CAM(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet38CAM, self).__init__()
        # Load pretrained ResNet50 as base (ResNet38 is not standard, using ResNet50)
        base_model = models.resnet50(pretrained=True)
        
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(base_model.children())[:-2])
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Linear(2048, num_classes)
        
        # Hook for feature maps
        self.features[-1].register_forward_hook(self.save_feature_map)
        self.features[-1].register_backward_hook(self.save_gradient)
    
    def save_feature_map(self, module, input, output):
        global feature_maps
        feature_maps.append(output)
    
    def save_gradient(self, module, grad_input, grad_output):
        global gradients
        gradients.append(grad_output[0])
    
    def forward(self, x):
        x = self.features(x)
        pooled = self.gap(x)
        pooled = pooled.view(pooled.size(0), -1)
        x = self.classifier(pooled)
        return x

def load_model():
    global resnet_model
    if resnet_model is None:
        resnet_model = ResNet38CAM(num_classes=1000)
        resnet_model.eval()
    return resnet_model

def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def generate_cam(image: Image.Image, target_class=None):
    global feature_maps, gradients
    feature_maps = []
    gradients = []
    
    model = load_model()
    
    # Preprocess
    input_tensor = preprocess_image(image)
    
    # Forward pass
    output = model(input_tensor)
    
    # Get predicted class if not specified
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    
    # Backward pass
    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0][target_class] = 1
    output.backward(gradient=one_hot, retain_graph=True)
    
    # Get feature maps and gradients
    features = feature_maps[0].detach().cpu().numpy()[0]
    grads = gradients[0].detach().cpu().numpy()[0]
    
    # Calculate weights
    weights = np.mean(grads, axis=(1, 2))
    
    # Generate CAM
    cam = np.zeros(features.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * features[i]
    
    # Apply ReLU
    cam = np.maximum(cam, 0)
    
    # Normalize
    cam = cam - np.min(cam)
    if np.max(cam) != 0:
        cam = cam / np.max(cam)
    
    # Resize to original image size
    cam = cv2.resize(cam, (image.width, image.height))
    
    return cam, target_class

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

@app.get("/")
async def root():
    return {"message": "CAM-SAM API is running"}

@app.post("/process")
async def process_image(file: UploadFile = File(...)) -> Dict[str, str]:
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Generate CAM
        cam, predicted_class = generate_cam(image)
        
        # Create overlay image
        cam_image = overlay_cam_on_image(image, cam)
        
        # For now, SAM image is the same as CAM (will implement SAM later)
        sam_image = cam_image
        
        # Convert to base64
        original_b64 = image_to_base64(image)
        cam_b64 = image_to_base64(cam_image)
        sam_b64 = image_to_base64(sam_image)
        
        return JSONResponse(content={
            "original": original_b64,
            "cam": cam_b64,
            "sam": sam_b64,
            "predicted_class": int(predicted_class),
            "message": "CAM generated successfully. SAM will be implemented next."
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)