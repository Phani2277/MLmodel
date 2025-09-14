import argparse

import torch
from PIL import Image
from torchvision import transforms

from train import build_model

LABELS = ["dirty", "damaged"]


def load_model(weights_path: str) -> torch.nn.Module:
    model = build_model()
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict(model, image_path: str):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0]
    return {label: float(prob) for label, prob in zip(LABELS, probs)}


def main():
    parser = argparse.ArgumentParser(description="Run inference on a car image")
    parser.add_argument("weights", help="Path to model weights")
    parser.add_argument("image", help="Image to classify")
    args = parser.parse_args()

    model = load_model(args.weights)
    preds = predict(model, args.image)

    for label, prob in preds.items():
        state = "yes" if prob > 0.5 else "no"
        print(f"{label}: {state} (prob={prob:.2f})")


if __name__ == "__main__":
    main()
