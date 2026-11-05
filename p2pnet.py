from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import vgg16_bn

WEIGHTS_URL = "https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet/raw/main/weights/SHTechA.pth"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def head_block(in_channels, out_channels, size=256):
    layers = nn.Module()
    layers.conv1 = nn.Conv2d(in_channels, size, 3, padding=1)
    layers.conv2 = nn.Conv2d(size, size, 3, padding=1)
    layers.conv3 = nn.Conv2d(size, size, 3, padding=1)
    layers.conv4 = nn.Conv2d(size, size, 3, padding=1)
    layers.output = nn.Conv2d(size, out_channels, 3, padding=1)
    return layers


def apply_head_block(layers, x):
    x = torch.relu(layers.conv1(x))
    x = torch.relu(layers.conv2(x))
    return layers.output(x).permute(0, 2, 3, 1)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        f = list(vgg16_bn(weights=None).features.children())
        self.body1 = nn.Sequential(*f[:13])
        self.body2 = nn.Sequential(*f[13:23])
        self.body3 = nn.Sequential(*f[23:33])
        self.body4 = nn.Sequential(*f[33:43])

    def forward(self, x):
        outputs = []
        for stage in (self.body1, self.body2, self.body3, self.body4):
            x = stage(x)
            outputs.append(x)
        return outputs


class FPN(nn.Module):
    def __init__(self, c3=256, c4=512, c5=512, size=256):
        super().__init__()
        self.P5_1 = nn.Conv2d(c5, size, 1)
        self.P5_2 = nn.Conv2d(size, size, 3, padding=1)
        self.P4_1 = nn.Conv2d(c4, size, 1)
        self.P4_2 = nn.Conv2d(size, size, 3, padding=1)
        self.P3_1 = nn.Conv2d(c3, size, 1)
        self.P3_2 = nn.Conv2d(size, size, 3, padding=1)

    def forward(self, c3, c4, c5):
        p5 = self.P5_1(c5)
        p4 = self.P4_1(c4) + nn.functional.interpolate(p5, scale_factor=2, mode="nearest")
        return self.P4_2(p4)


class P2PNet(nn.Module):
    def __init__(self, rows=2, columns=2):
        super().__init__()
        self.anchors_per_cell = rows * columns
        self.rows = rows
        self.columns = columns
        self.backbone = Backbone()
        self.fpn = FPN()
        self.regression = head_block(256, self.anchors_per_cell * 2)
        self.classification = head_block(256, self.anchors_per_cell * 2)

    def anchors(self, height, width, step, device):
        dx = (torch.arange(1, self.columns + 1, device=device) - 0.5) * step / self.columns - step / 2
        dy = (torch.arange(1, self.rows + 1, device=device) - 0.5) * step / self.rows - step / 2
        base = torch.stack(torch.meshgrid(dx, dy, indexing="xy"), -1).reshape(-1, 2)
        cx = (torch.arange(width, device=device) + 0.5) * step
        cy = (torch.arange(height, device=device) + 0.5) * step
        centers = torch.stack(torch.meshgrid(cx, cy, indexing="xy"), -1).reshape(-1, 1, 2)
        return (centers + base[None]).reshape(-1, 2)

    def forward(self, x):
        _, c3, c4, c5 = self.backbone(x)
        p4 = self.fpn(c3, c4, c5)
        batch_size, _, height, width = p4.shape
        offsets = apply_head_block(self.regression, p4).reshape(batch_size, -1, 2) * 100
        logits = apply_head_block(self.classification, p4).reshape(batch_size, -1, 2)
        points = offsets + self.anchors(height, width, x.shape[2] // height, x.device)
        return points, logits.softmax(-1)[..., 1]


def load_model(path="models/p2pnet_shtech_a.pth", device=None):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Baixando pesos do P2PNet para {path}...")
        urlretrieve(WEIGHTS_URL, path)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = P2PNet()
    model.load_state_dict(state.get("model", state))
    return model.eval().to(device)


class DensityCounter:
    def __init__(self, weights="models/p2pnet_shtech_a.pth", threshold=0.5, max_side=2048, scale=1.0):
        self.model = load_model(weights)
        self.device = next(self.model.parameters()).device
        self.threshold = threshold
        self.max_side = max_side
        self.scale = scale

    @torch.inference_mode()
    def points(self, image):
        height, width = image.shape[:2]
        factor = min(self.scale, self.max_side / max(height, width))
        new_width = max(128, round(width * factor / 128) * 128)
        new_height = max(128, round(height * factor / 128) * 128)
        rgb = cv2.cvtColor(cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255 - MEAN) / STD
        x = torch.from_numpy(x).permute(2, 0, 1)[None].to(self.device)
        points, confs = self.model(x)
        keep = confs[0] > self.threshold
        points = points[0][keep].cpu().numpy()
        points[:, 0] *= width / new_width
        points[:, 1] *= height / new_height
        return points, confs[0][keep].cpu().numpy()
