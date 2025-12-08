from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import io
import cv2
import numpy as np
import base64
import torch
import torchvision.transforms as transforms
from model import ResNet38d

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHECKPOINT_PATH = r"C:\Users\admin\Downloads\sam_cam\resnet_38d.params"
cam_model = None

def get_model():
    global cam_model
    if cam_model is None:
        cam_model = ResNet38d(num_classes=20, checkpoint_path=CHECKPOINT_PATH)
        cam_model.eval()
        if torch.cuda.is_available():
            cam_model = cam_model.cuda()
            print("Model loaded on GPU")
        else:
            print("Model loaded on CPU")
    return cam_model

def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((448, 448)), # ResNet38d thường dùng input to để CAM nét hơn
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
    return input_tensor

def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        model = get_model()
        input_tensor = preprocess_image(image)
        
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.nn.functional.softmax(output, dim=1)
            pred_class = prob.argmax(dim=1).item()
        
        cam = model.get_cam(target_class=pred_class)
        
        if cam is None:
            return JSONResponse({"error": "CAM failed (Feature maps empty or Class ID invalid)"})

        cam_resized = cv2.resize(cam, (image.width, image.height))
        cam_resized = cv2.GaussianBlur(cam_resized, (21, 21), 0)
        
        max_val = np.max(cam_resized)
        if max_val > 0:
            cam_resized = cam_resized / max_val
        else:
            pass 

        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        img_arr = np.array(image)
        overlay = cv2.addWeighted(img_arr, 0.6, heatmap, 0.4, 0)
        
        return JSONResponse({
            "original": image_to_base64(image),
            "cam": image_to_base64(Image.fromarray(overlay)),
            "predicted_class": int(pred_class)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)