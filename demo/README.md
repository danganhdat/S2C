# Models Folder

This folder is a placeholder for model weights.

## Expected Files

Place the following model weights in this folder:

1. **baseline_net_main.pth** - Baseline ResNet38 weights (e.g., epoch 001)
2. **latest_net_main.pth** - Latest/improved ResNet38 weights (e.g., epoch 019)
3. **sam_vit_b.pth** or **sam_vit_h.pth** - Segment Anything Model weights

## Download Links

- ResNet38 pretrained: Check `pretrained/` folder in the main repo
- SAM weights: [Official SAM Repository](https://github.com/facebookresearch/segment-anything)

## Usage

Update the paths in `backend.py` to point to your actual model files:

```python
# In load_baseline_model()
baseline_model.load_state_dict(torch.load("models/baseline_net_main.pth"))

# In load_latest_model()
latest_model.load_state_dict(torch.load("models/latest_net_main.pth"))

# In load_sam_model()
sam_model = sam_model_registry["vit_b"](checkpoint="models/sam_vit_b.pth")
```
