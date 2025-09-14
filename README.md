# Car State Classification Model

Prototype PyTorch model to classify whether a car is **dirty** and/or **damaged** from a photo.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare dataset structured as images and an annotation CSV with columns:
   `filepath,dirty,damaged` where labels are 0 or 1.

## Training

```bash
python train.py DATA_DIR annotations.csv model.pth --epochs 5
```

## Inference

```bash
python demo.py model.pth path/to/image.jpg
```

The script prints whether the car is dirty and damaged with corresponding probabilities.
