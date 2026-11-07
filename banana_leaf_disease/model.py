"""Model construction for banana leaf disease classification."""

import timm
from torch import nn


MODEL_NAME = "vit_small_patch16_224"
NUM_CLASSES = 3
CHECKPOINT_PATH = "checkpoints/banana_leaf_disease_vit.pth"


def build_vit(
    model_name: str = MODEL_NAME,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """Build a ViT classifier with a three-class disease-classification head."""
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")
    return timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )


build_model = build_vit
