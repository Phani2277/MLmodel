# Car State Classification

Prototype ML model that predicts whether a car is dirty and/or damaged from a photo.

## Training

Quick demo with synthetic data:

```bash
python train.py --fake-data --epochs 1
```

Training on a real dataset organised as:

```
data/
  clean_intact/
  clean_damaged/
  dirty_intact/
  dirty_damaged/
```

Run:

```bash
python train.py --data-dir data --epochs 10
```

## Prediction

```bash
python predict.py --model-path model.pth --image-path path/to/car.jpg
```

Or test with a random image:

```bash
python predict.py --model-path model.pth --fake
```

## Requirements

Install dependencies:

```bash
pip install torch torchvision pillow
```

