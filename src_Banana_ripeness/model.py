# model builder (ViT -> timm)
import timm
import torch.nn as nn

def build_vit(model_name="vit_base_patch16_224", num_classes=2, pretrained=True):
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model
