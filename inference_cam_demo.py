import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from networks import resnet38d
from tools import imutils
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
import cv2

# Constants matching training defaults
NUM_CLASSES = 20
FEATURE_DIM = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CATEGORIES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
              'bus', 'car', 'cat', 'chair', 'cow',
              'diningtable', 'dog', 'horse', 'motorbike', 'person',
              'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

def preprocess_image(img_path, resize_long=None):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at {img_path}")
        
    img = Image.open(img_path).convert("RGB")
    
    if resize_long:
        w, h = img.size
        scale = resize_long / float(max(w, h))
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)

    # Normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    img_tensor = transform(img).unsqueeze(0) # (1, 3, H, W)
    return img, img_tensor

def load_model(weights_path=None):
    # Initialize model architecture
    model = resnet38d.Net_CAM(C=NUM_CLASSES, D=FEATURE_DIM)
    
    if weights_path and os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            
            # Remove 'module.' prefix if present (from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            model.load_state_dict(new_state_dict, strict=True)
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Warning: Error loading state_dict strictly: {e}")
            print("Attempting to load with strict=False...")
            try:
                model.load_state_dict(new_state_dict, strict=False)
                print("Weights loaded with strict=False.")
            except Exception as e2:
                print(f"Error loading weights: {e2}")
    else:
        print("No weights provided or file not found. Initializing with random/default weights.")
        if weights_path:
             print(f"Warning: File {weights_path} does not exist.")
    
    model.eval()
    return model

def get_peaks(cam_data, threshold=0.5):
    """
    Find global maximum and local peaks in CAM.
    cam_data: 2D numpy array (H, W)
    """
    # Global Max
    argmax_indices = np.argmax(cam_data)
    gy, gx = np.unravel_index(argmax_indices, cam_data.shape)
    peak_max = np.array([[gx, gy]]) # x, y

    # Local Peaks
    cam_filtered = ndi.maximum_filter(cam_data, size=3, mode='constant')
    peaks_temp = peak_local_max(cam_filtered, min_distance=20)
    
    # Filter by threshold (using original cam values at peak locations)
    # peak_local_max returns [r, c] (y, x) -> need to flip for (x, y)
    if len(peaks_temp) > 0:
        val_at_peaks = cam_data[peaks_temp[:,0], peaks_temp[:,1]]
        peaks_valid = peaks_temp[val_at_peaks > threshold]
        
        if len(peaks_valid) > 0:
             peaks_valid_xy = np.flip(peaks_valid, axis=1) # y,x -> x,y
             # Combine max and valid local peaks
             all_peaks = np.concatenate((peak_max, peaks_valid_xy), axis=0)
        else:
             all_peaks = peak_max
    else:
        all_peaks = peak_max
        
    return all_peaks

def main():
    parser = argparse.ArgumentParser(description="Inference demo for CAM visualization with ResNet38")
    parser.add_argument("--image", required=True, type=str, help="Path to input image")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights (.pth)")
    parser.add_argument("--output", type=str, default="cam_output_peaks.png", help="Path to save output image")
    parser.add_argument("--resize", type=int, default=512, help="Resize long side of image for inference")
    parser.add_argument("--class_idx", type=int, default=None, help="Specific class index to visualize (0-19). If None, uses max predicted class.")
    
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Model
    model = load_model(args.weights)
    model = model.to(device)

    # Preprocess
    orig_img_pil, img_tensor = preprocess_image(args.image, args.resize)
    img_tensor = img_tensor.to(device)
    
    w, h = orig_img_pil.size

    # Inference
    print("Running inference...")
    with torch.no_grad():
        out = model(img_tensor)
        cam = out['cam'] # Shape (1, C, H, W)
        pred = out['pred'] # (1, C)
        
        # Determine class to visualize
        probs = torch.sigmoid(pred)[0]
        
        if args.class_idx is not None:
             target_class = args.class_idx
             score = probs[target_class].item()
             print(f"Visualizing specified class: {CATEGORIES[target_class]} ({target_class}), Score: {score:.4f}")
        else:
            top_prob, top_label_idx = torch.max(probs, dim=0)
            target_class = top_label_idx.item()
            score = top_prob.item()
            print(f"Top predicted class: {CATEGORIES[target_class]} ({target_class}), Score: {score:.4f}")

        # Get CAM for the target class
        cam_map = cam[0, target_class] # (H_feat, W_feat)
        
        # Upsample CAM to original image size
        cam_map = F.interpolate(cam_map.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False)
        cam_map = cam_map.squeeze() # (H, W)
        
        # Apply ReLU
        cam_map = F.relu(cam_map)
        
        # Normalize CAM to [0, 1]
        cam_map_min = cam_map.min()
        cam_map_max = cam_map.max()
        if cam_map_max > cam_map_min:
            cam_map = (cam_map - cam_map_min) / (cam_map_max - cam_map_min + 1e-7)
        
        cam_np = cam_map.cpu().numpy()
        
    # Visualization with Peaks
    print("Generating visualization with peaks...")
    
    # Prepare base visualization (Heatmap on Image)
    vis_img = np.array(orig_img_pil).astype(np.float32) / 255.0 
    vis_img = vis_img.transpose(2, 0, 1) # (3, H, W)
    vis_result = imutils.cam_on_image(vis_img, cam_np) # (3, H, W) RGB
    
    # Get Peaks
    peaks = get_peaks(cam_np, threshold=0.5)
    print(f"Found {len(peaks)} peaks.")
    
    # Draw points using cv2
    # Convert RGB (from imutils) to format cv2 can draw on (numpy HWC)
    vis_result_cv = vis_result.transpose(1, 2, 0).copy()
    
    for i, (px, py) in enumerate(peaks):
        # Green for global max (first one), Red for others
        color = (0, 255, 0) if i == 0 else (255, 0, 0)
        
        # Draw marker
        cv2.circle(vis_result_cv, (int(px), int(py)), 6, color, -1)
        cv2.circle(vis_result_cv, (int(px), int(py)), 6, (255, 255, 255), 1)

    # Save output
    out_img = Image.fromarray(vis_result_cv)
    out_img.save(args.output)
    print(f"Successfully saved CAM visualization with peaks to {args.output}")

if __name__ == "__main__":
    main()
