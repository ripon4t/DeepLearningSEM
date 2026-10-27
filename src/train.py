"""Train the defect classifier.   python -m src.train --epochs 20"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import load_splits
from .model import DefectCNN


def run_epoch(model, loader, device, criterion, optimizer=None, scheduler=None):
    train = optimizer is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for xb, yb in tqdm(loader, leave=False, disable=not train):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            loss_sum += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)
    return loss_sum / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sem_defects.npz")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("device:", device)

    (tr, va, te), classes = load_splits(args.data, args.seed)
    print(f"train/val/test = {len(tr)}/{len(va)}/{len(te)}  classes={classes}")
    tl = DataLoader(tr, args.batch_size, shuffle=True, num_workers=2)
    vl = DataLoader(va, args.batch_size, num_workers=2)

    model = DefectCNN(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(tl)
    )

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best, t0 = 0.0, time.time()
    for ep in range(1, args.epochs + 1):
        trl, tra = run_epoch(model, tl, device, criterion, optimizer, scheduler)
        val, vaa = run_epoch(model, vl, device, criterion)
        for k, v in zip(hist, (trl, val, tra, vaa)):
            hist[k].append(v)
        print(f"epoch {ep:02d}  train loss {trl:.3f} acc {tra:.3f} | val loss {val:.3f} acc {vaa:.3f}")
        if vaa > best:
            best = vaa
            torch.save({"model": model.state_dict(), "classes": classes}, out / "best.pt")
    print(f"best val acc {best:.4f}  ({time.time() - t0:.0f}s)")

    (out / "history.json").write_text(json.dumps(hist, indent=2))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(hist["train_acc"], label="train"); ax[1].plot(hist["val_acc"], label="val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout(); plt.savefig(out / "training_curves.png", dpi=130)


if __name__ == "__main__":
    main()
