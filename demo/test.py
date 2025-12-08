from model import ResNet38d
import torch

model = ResNet38d(num_classes=20)
full_keys = set(model.state_dict().keys())

loaded_keys = {k for k in full_keys if "num_batches_tracked" not in k and "classifier" not in k}

missing_keys = full_keys - loaded_keys

print(f"Tổng số keys trong model: {len(full_keys)}")
print(f"Số keys 'thực sự' cần load (Weights + Bias): {len(loaded_keys)}")
print(f"Số lượng keys bị thiếu: {len(missing_keys)}")
print("-" * 30)
print("DANH SÁCH CÁC KEYS BỊ THIẾU:")
for k in sorted(list(missing_keys)):
    print(k)