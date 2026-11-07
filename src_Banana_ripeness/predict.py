# src/predict.py
import torch
from PIL import Image
from torchvision import transforms
from model import build_vit

IMG_SIZE = 224
classes = ["NotBestToEat", "BestToEat"]

def load_image(path):
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    img = Image.open(path).convert("RGB") # Red Green Blue
    return tf(img).unsqueeze(0)

def predict(image_path, checkpoint="checkpoints/best_vit.pth", device="cpu"):
    model = build_vit(model_name="vit_small_patch16_224", num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device).eval()
    img = load_image(image_path).to(device)
    with torch.no_grad():
        out = model(img)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        idx = int(probs.argmax())
        return {
            "label": classes[idx],
            "confidence": float(probs[idx]),
            "probabilities": {
                name: float(probability)
                for name, probability in zip(classes, probs)
            },
        }

if __name__ == "__main__":
    import sys
    img_path = sys.argv[1]
    res = predict(img_path, device="cuda" if torch.cuda.is_available() else "cpu")
    print(res)
