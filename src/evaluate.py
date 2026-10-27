"""Evaluate on the held-out test set + Grad-CAM explanations.
python -m src.evaluate --ckpt outputs/best.pt
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from .dataset import load_splits
from .model import DefectCNN


def grad_cam(model, x, target=None):
    """Grad-CAM on the last conv block. x: (1,1,H,W). Returns (heatmap[H,W], predicted class)."""
    acts, grads = {}, {}

    def fwd_hook(m, i, o):
        acts["a"] = o
        o.register_hook(lambda g: grads.update(g=g))  # capture d(logit)/d(activation)

    h = model.features[-1].register_forward_hook(fwd_hook)
    model.zero_grad()
    logits = model(x)
    pred = int(logits.argmax(1))
    logits[0, pred if target is None else target].backward()
    h.remove()
    w = grads["g"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * acts["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
    return cam.detach().cpu().numpy(), pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sem_defects.npz")
    ap.add_argument("--ckpt", default="outputs/best.pt")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cpu"  # evaluation is cheap; keeps Grad-CAM simple
    ckpt = torch.load(args.ckpt, map_location=device)
    classes = ckpt["classes"]
    model = DefectCNN(len(classes)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    _, _, te = load_splits(args.data, args.seed)[0]
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in DataLoader(te, 128):
            y_pred += model(xb).argmax(1).tolist()
            y_true += yb.tolist()
    print(classification_report(y_true, y_pred, target_names=classes, digits=3))

    out = Path(args.out)
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("predicted"); plt.ylabel("true"); plt.title("Normalized confusion matrix (test)")
    plt.tight_layout(); plt.savefig(out / "confusion_matrix.png", dpi=130); plt.close()

    # Grad-CAM: 3 test examples per class
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(classes), 6, figsize=(12, 8))
    for c, name in enumerate(classes):
        idx = rng.choice(np.where(te.y == c)[0], 3, replace=False)
        for j, i in enumerate(idx):
            x, _ = te[i]
            cam, pred = grad_cam(model, x.unsqueeze(0).to(device))
            img = te.X[i]
            axes[c, 2 * j].imshow(img, cmap="gray"); axes[c, 2 * j].axis("off")
            axes[c, 2 * j].set_title(f"true: {name}", fontsize=8)
            axes[c, 2 * j + 1].imshow(img, cmap="gray")
            axes[c, 2 * j + 1].imshow(cam, cmap="jet", alpha=0.45); axes[c, 2 * j + 1].axis("off")
            axes[c, 2 * j + 1].set_title(f"Grad-CAM -> {classes[pred]}", fontsize=8)
    plt.tight_layout(); plt.savefig(out / "gradcam.png", dpi=120)
    print("Saved confusion_matrix.png and gradcam.png to", out)


if __name__ == "__main__":
    main()
