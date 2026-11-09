"""torchvision classifier backbones for the uniform classifier (Phase B candidates), shared by inference
(backend/uniform/classifier.py: TorchvisionClassifier) and training (training/train_torchvision.py), so a
checkpoint is always rebuilt with exactly the network it was trained with.

Checkpoint (torch.save of a dict, FORMAT below):
  arch        efficientnet_b0 | mobilenet_v3_large | mobilenet_v3_small
  classes     class names in output order, e.g. ["non_uniform", "uniform"] (mapped by NAME via class_map)
  input_size  [height, width] the network was trained on
  mean, std   input normalisation (RGB, 0..1 scale)
  state_dict  the weights
The upper-body crop is first resized to model.crop_size (as for the YOLO-cls model), then to input_size.
"""
from __future__ import annotations

FORMAT = "udm-torchvision-cls/1"
ARCHS = ("efficientnet_b0", "mobilenet_v3_large", "mobilenet_v3_small")
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
DEFAULT_INPUT = [224, 160]       # height, width: the 112x160 (w x h) torso crop shape, scaled


def build(arch: str, num_classes: int, pretrained: bool = False):
    """The network with a new `num_classes` head. `pretrained` = ImageNet weights (downloaded once by
    torchvision into its cache) - training only; inference loads the checkpoint's own weights."""
    import torch.nn as nn
    from torchvision import models as tvm
    if arch == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif arch == "mobilenet_v3_large":
        m = tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif arch == "mobilenet_v3_small":
        m = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"unknown torchvision arch {arch!r} (one of {', '.join(ARCHS)})")
    return m


def to_tensor(crops: list, input_size: list[int], mean: list[float], std: list[float]):
    """BGR uint8 crops -> normalised NCHW float tensor."""
    import cv2
    import numpy as np
    import torch
    h, w = input_size
    arr = np.stack([cv2.cvtColor(cv2.resize(c, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB) for c in crops])
    x = torch.from_numpy(arr).float().div_(255.0).permute(0, 3, 1, 2)
    return (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)


def load(path: str, device: str = "cpu") -> tuple:
    """(model in eval mode, checkpoint metadata without the weights)."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or ck.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} checkpoint (training/train_torchvision.py)")
    model = build(ck["arch"], len(ck["classes"]))
    model.load_state_dict(ck["state_dict"])
    model.eval().to(device)
    return model, {k: v for k, v in ck.items() if k != "state_dict"}
