import argparse
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

def load_model(path):
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Predict car state")
    parser.add_argument('--model-path', required=True, help='Path to trained model')
    parser.add_argument('--image-path', help='Path to image file')
    parser.add_argument('--fake', action='store_true', help='Use random image instead')
    args = parser.parse_args()

    model = load_model(args.model_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    if args.fake or not args.image_path:
        image = torch.rand(3, 224, 224)
    else:
        image = transform(Image.open(args.image_path).convert('RGB'))

    with torch.no_grad():
        logits = model(image.unsqueeze(0))[0]
        probs = torch.sigmoid(logits)
    dirty_prob, damaged_prob = probs
    print(f"dirty: {dirty_prob.item():.3f}, clean: {1 - dirty_prob.item():.3f}")
    print(f"damaged: {damaged_prob.item():.3f}, intact: {1 - damaged_prob.item():.3f}")


if __name__ == '__main__':
    main()
