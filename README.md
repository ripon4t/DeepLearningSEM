# SEM Nanostructure Defect Classifier

A PyTorch convolutional neural network that classifies fabrication defects in SEM-style images of nanopillar arrays, with **Grad-CAM** explainability showing *where* the model looks.

Built by a nanofabrication scientist, so the problem framing comes from real process-control needs: electron-beam lithography, etching, and lift-off all produce characteristic failure modes that engineers currently inspect by eye.

![Sample images](assets/samples.png)

## Problem

Given a 96×96 grayscale micrograph of a periodic nanopillar array, classify it as:

| Class | Physical analogue |
|---|---|
| `clean` | In-spec pattern |
| `missing` | Pillars absent (failed lift-off or etch, resist loss) |
| `bridged` | Neighbouring pillars fused (resist collapse, over-dose proximity effect) |
| `deformed` | Wrong size, shape or position (dose or etch non-uniformity) |

## Dataset

Real labelled SEM defect data is scarce and proprietary, so images are **procedurally generated** (`src/generate_data.py`): randomized pitch, pillar radius, lattice rotation, edge brightening, blur, Poisson shot noise and Gaussian read noise. This makes the project fully reproducible with no downloads, and lets you control difficulty.

Default: 1,500 images per class (6,000 total), stratified 70/15/15 train/val/test split.

## Model

Compact VGG-style CNN (4 conv blocks, BatchNorm, global average pooling, dropout), trained from scratch with AdamW, one-cycle LR schedule, label smoothing and flip/rotation/brightness augmentation. The last conv block is left un-pooled (12×12) so Grad-CAM maps stay spatially informative.

## Quick start

```bash
git clone https://github.com/<your-username>/sem-defect-classifier.git
cd sem-defect-classifier
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.generate_data --n-per-class 1500   # build dataset
python -m src.train --epochs 20                  # train (minutes on CPU)
python -m src.evaluate --ckpt outputs/best.pt    # metrics + Grad-CAM
```

Outputs land in `outputs/`: `training_curves.png`, `confusion_matrix.png`, `gradcam.png`, `best.pt`.

## Results

> Fill in after running (copy the numbers from the `evaluate` output and embed the figures).

| Metric | Value |
|---|---|
| Test accuracy | _TBD_ |
| Macro F1 | _TBD_ |
| Hardest class | _TBD_ |

![Confusion matrix](outputs/confusion_matrix.png)
![Grad-CAM](outputs/gradcam.png)

## Project structure

```
src/
  generate_data.py   synthetic SEM image generator
  dataset.py         PyTorch Dataset, augmentation, stratified splits
  model.py           DefectCNN architecture
  train.py           training loop, LR schedule, curves
  evaluate.py        test metrics, confusion matrix, Grad-CAM
```

## Ideas for extension

- Replace synthetic data with real SEM images and fine-tune (transfer learning with a pretrained ResNet).
- Measure domain shift: train on one pitch range, test on another.
- Add a defect *localization* head (bounding boxes or segmentation) instead of image-level labels.
- Export to ONNX and serve with a small Streamlit or FastAPI demo.
- Calibrate confidence (temperature scaling) for use as an inline process monitor.

## Author

Ripon Kumar Dey, PhD: nanofabrication and quantum photonics.
Licensed under MIT.
