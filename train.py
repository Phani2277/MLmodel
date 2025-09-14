import argparse
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models

class FakeCarDataset(Dataset):
    """Simple dataset generating random images and binary labels."""
    def __init__(self, size=100):
        self.size = size
        self.data = torch.rand(size, 3, 224, 224)
        self.labels = torch.randint(0, 2, (size, 2), dtype=torch.float32)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def load_real_data(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    def target_transform(index):
        class_name = dataset.classes[index]
        dirty = 1 if 'dirty' in class_name else 0
        damaged = 1 if 'damaged' in class_name or 'scratch' in class_name else 0
        return torch.tensor([dirty, damaged], dtype=torch.float32)
    dataset.target_transform = target_transform
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train(args):
    if args.fake_data:
        loader = DataLoader(FakeCarDataset(size=64), batch_size=args.batch_size, shuffle=True)
    else:
        loader = load_real_data(args.data_dir, args.batch_size)

    # Use randomly initialised weights to avoid downloading pretrained models
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for images, targets in loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}: loss {running_loss/len(loader):.4f}")
    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")


def parse_args():
    p = argparse.ArgumentParser(description="Train car state classifier")
    p.add_argument('--data-dir', type=str, default='data', help='Path to dataset root')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--output', type=str, default='model.pth')
    p.add_argument('--fake-data', action='store_true', help='Use random data for quick demo')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
