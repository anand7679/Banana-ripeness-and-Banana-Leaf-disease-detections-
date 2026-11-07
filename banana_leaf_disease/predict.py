# Run inference with the ViT banana leaf disease classifier.
import argparse
import torch
from PIL import Image
from torchvision import transforms
from .model import CHECKPOINT_PATH, MODEL_NAME, NUM_CLASSES, build_vit

IMG_SIZE = 224
CLASS_NAMES = ["cordana", "healthy", "sigatoka"]


def predict(
    image_path,
    checkpoint_path=CHECKPOINT_PATH,
    device="cpu",
):
    model = build_vit(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        pretrained=False,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()

    transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(image), dim=1).cpu().numpy()[0]
    index = int(probabilities.argmax())
    return {
        "label": CLASS_NAMES[index],
        "confidence": float(probabilities[index]),
        "probabilities": {
            name: float(probability)
            for name, probability in zip(CLASS_NAMES, probabilities)
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    print(predict(args.image, args.checkpoint, args.device))
