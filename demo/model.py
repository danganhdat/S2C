import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mxnet as mx
import re

class ResBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, dilation=1, mid_planes=None):
        super(ResBlock, self).__init__()
        if mid_planes is None:
            mid_planes = planes
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv2 = nn.Conv2d(mid_planes, planes, kernel_size=3, stride=1,
                               padding=dilation, dilation=dilation, bias=False)
        self.shortcut = None
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class ResNet38d(nn.Module):
    def __init__(self, num_classes=20, checkpoint_path=None):
        super(ResNet38d, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(128, 3, stride=1, dilation=1)
        self.layer2 = self._make_layer(256, 3, stride=2, dilation=1)
        self.layer3 = self._make_layer(512, 6, stride=1, dilation=2)
        self.layer4 = self._make_layer(1024, 3, stride=1, dilation=4, mid_planes=512)
        self.classifier = nn.Conv2d(1024, num_classes, kernel_size=1, bias=True)
        self.feature_maps = None

        if checkpoint_path:
            self.load_mxnet_weights(checkpoint_path)

        self.layer4.register_forward_hook(self.save_feature_map)

    def _make_layer(self, planes, num_blocks, stride, dilation, mid_planes=None):
        layers = []
        layers.append(ResBlock(self.in_planes, planes, stride, dilation, mid_planes))
        self.in_planes = planes
        for i in range(1, num_blocks):
            layers.append(ResBlock(self.in_planes, planes, 1, dilation, mid_planes))
        return nn.Sequential(*layers)

    def load_mxnet_weights(self, path):
        try:
            mx_save_dict = mx.nd.load(path)
            pt_state_dict = self.state_dict()
            converted_weights = {}

            def get_block_index(block_str):
                if block_str == 'a': return 0
                match = re.search(r'b(\d+)', block_str)
                if match: return int(match.group(1))
                return 0

            for k, v in mx_save_dict.items():
                k = k.replace('arg:', '').replace('aux:', '')
                new_k = None

                if 'conv1a' in k:
                    new_k = k.replace('conv1a', 'conv1').replace('_weight', '.weight').replace('_bias', '.bias')

                match = re.search(r'(res|bn)(\d+)([a-z0-9]+)_branch([a-z0-9]+)_(.+)', k)
                if match:
                    prefix_type, stage_idx, block_str, branch_path, param_type = match.groups()
                    stage_idx = int(stage_idx)
                    if stage_idx > 5:
                        continue

                    pt_layer_idx = stage_idx - 1
                    pt_block_idx = get_block_index(block_str)

                    if branch_path == '2a':
                        pt_module = 'conv1' if prefix_type == 'res' else 'bn1'
                    elif branch_path == '2b1':
                        pt_module = 'conv2' if prefix_type == 'res' else 'bn2'
                    elif branch_path == '1':
                        pt_module = 'shortcut'
                    else:
                        continue

                    if param_type == 'weight': pt_param = 'weight'
                    elif param_type == 'bias': pt_param = 'bias'
                    elif param_type == 'gamma': pt_param = 'weight'
                    elif param_type == 'beta': pt_param = 'bias'
                    elif param_type == 'moving_mean': pt_param = 'running_mean'
                    elif param_type == 'moving_var': pt_param = 'running_var'
                    else: continue

                    new_k = f"layer{pt_layer_idx}.{pt_block_idx}.{pt_module}.{pt_param}"

                if new_k:
                    converted_weights[new_k] = torch.from_numpy(v.asnumpy())

            loaded_count = 0
            missing_layers = []

            for pt_k, pt_v in pt_state_dict.items():
                if "num_batches_tracked" in pt_k: continue
                
                if pt_k in converted_weights:
                    src_tensor = converted_weights[pt_k]
                    if src_tensor.shape == pt_v.shape:
                        self.state_dict()[pt_k].copy_(src_tensor)
                        loaded_count += 1
                    else:
                        try:
                            self.state_dict()[pt_k].copy_(src_tensor.view(pt_v.shape))
                            loaded_count += 1
                        except:
                            missing_layers.append(f"{pt_k} (Shape mismatch)")
                else:
                    if "classifier" not in pt_k:
                        missing_layers.append(pt_k)

            print(f"Total loaded layers: {loaded_count}")
            
            if len(missing_layers) > 0:
                print("Missing layers:")
                for layer in missing_layers:
                    print(layer)
            else:
                print("No missing backbone layers.")

        except Exception as e:
            import traceback
            traceback.print_exc()

    def save_feature_map(self, module, input, output):
        self.feature_maps = output.detach()

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x_gap = F.adaptive_avg_pool2d(x, (1, 1))
        out = self.classifier(x_gap)
        return out.view(out.size(0), -1)

    def get_cam(self, target_class):
        if self.feature_maps is None:
            return None
        features = self.feature_maps.cpu().numpy()[0]
        weights = self.classifier.weight.detach().cpu().numpy().squeeze()
        if target_class >= len(weights):
            return None
        w_c = weights[target_class]
        cam = (w_c[:, None, None] * features).sum(axis=0)
        cam = np.maximum(cam, 0)
        cam = cam - np.min(cam)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
        return cam