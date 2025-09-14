import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from car_dataset import CarStateDataset


def build_model(num_labels: int = 2) -> nn.Module:
    """Create a ResNet model with a custom head.

    If pretrained weights cannot be downloaded (e.g. no internet), the model
    falls back to randomly initialized weights so the rest of the pipeline can
    still run.
    """
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except Exception:
        model = models.resnet18(weights=None)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_labels)
    return model


def train(args):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = CarStateDataset(args.annotations, args.data_dir, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = build_model()
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        for images, labels in loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}: loss={loss.item():.4f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Train car state classification model")
    parser.add_argument("data_dir", help="Directory with images")
    parser.add_argument("annotations", help="CSV file with annotations")
    parser.add_argument("output", help="Path to save trained model")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
