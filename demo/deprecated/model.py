"""
Model loader and utility functions for CAM and SAM
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Tuple
import numpy as np

class ResNetCAM(nn.Module):
    """
    ResNet model modified for Class Activation Mapping (CAM)
    """
    def __init__(self, architecture='resnet50', num_classes=1000, pretrained=True):
        super(ResNetCAM, self).__init__()
        
        # Load base model
        if architecture == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
        elif architecture == 'resnet101':
            base_model = models.resnet101(pretrained=pretrained)
        elif architecture == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        # Extract features (all layers except final FC and avgpool)
        self.features = nn.Sequential(*list(base_model.children())[:-2])
        
        # Get the number of features from the last conv layer
        if architecture in ['resnet50', 'resnet101']:
            num_features = 2048
        else:  # resnet34
            num_features = 512
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Linear(num_features, num_classes)
        
        # Storage for hooks
        self.feature_maps = None
        self.gradients = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.feature_maps = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Hook on the last convolutional layer
        self.features[-1].register_forward_hook(forward_hook)
        self.features[-1].register_full_backward_hook(backward_hook)
    
    def forward(self, x):
        # Extract features
        features = self.features(x)
        
        # Global average pooling
        pooled = self.gap(features)
        pooled = pooled.view(pooled.size(0), -1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits
    
    def get_cam(self, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Class Activation Map
        
        Args:
            target_class: Target class index. If None, use predicted class.
            
        Returns:
            CAM as numpy array
        """
        if self.feature_maps is None or self.gradients is None:
            raise RuntimeError("Forward and backward pass must be done first")
        
        # Get gradients and feature maps
        gradients = self.gradients.detach().cpu().numpy()[0]  # [C, H, W]
        features = self.feature_maps.detach().cpu().numpy()[0]  # [C, H, W]
        
        # Calculate weights as mean of gradients
        weights = np.mean(gradients, axis=(1, 2))  # [C]
        
        # Weighted combination of feature maps
        cam = np.zeros(features.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * features[i]
        
        # Apply ReLU to focus on positive contributions
        cam = np.maximum(cam, 0)
        
        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam


class ModelManager:
    """
    Singleton class to manage model loading and caching
    """
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def get_resnet_cam(self, architecture='resnet50') -> ResNetCAM:
        """
        Get or load ResNet CAM model
        
        Args:
            architecture: ResNet architecture ('resnet34', 'resnet50', 'resnet101')
            
        Returns:
            ResNetCAM model
        """
        model_key = f"resnet_cam_{architecture}"
        
        if model_key not in self._models:
            print(f"Loading {architecture} model...")
            model = ResNetCAM(architecture=architecture, pretrained=True)
            model.eval()
            
            # Move to GPU if available
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            self._models[model_key] = model
            print(f"Model loaded on {device}")
        
        return self._models[model_key]
    
    def get_sam_model(self, model_type='vit_h'):
        """
        Get or load SAM model (to be implemented)
        
        Args:
            model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
            
        Returns:
            SAM model
        """
        # TODO: Implement SAM model loading
        raise NotImplementedError("SAM model loading will be implemented next")
    
    def clear_cache(self):
        """Clear all cached models"""
        self._models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_imagenet_class_name(class_idx: int) -> str:
    """
    Get ImageNet class name from index
    
    Args:
        class_idx: Class index (0-999)
        
    Returns:
        Class name string
    """
    # Simplified version - in production, load from file
    imagenet_classes = {
        0: "tench",
        1: "goldfish",
        2: "great white shark",
        # ... (full list would be loaded from a file)
        281: "tabby cat",
        282: "tiger cat",
        283: "Persian cat",
        # Add more as needed
    }
    
    return imagenet_classes.get(class_idx, f"class_{class_idx}")


def download_sam_checkpoint(model_type='vit_h', save_path='./checkpoints'):
    """
    Download SAM checkpoint (to be implemented)
    
    Args:
        model_type: SAM model type
        save_path: Path to save checkpoint
    """
    # TODO: Implement SAM checkpoint download
    pass


if __name__ == "__main__":
    # Test model loading
    manager = ModelManager()
    model = manager.get_resnet_cam('resnet50')
    print(f"Model loaded successfully: {type(model)}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")