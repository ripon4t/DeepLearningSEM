"""Synthetic SEM-style images of nanopillar arrays with fabrication defects.

Classes
-------
0 clean     - regular pillar array with natural jitter
1 missing   - one or more pillars absent (e.g. failed lift-off / etch)
2 bridged   - neighbouring pillars fused by a bright bridge (e.g. resist collapse)
3 deformed  - pillars with wrong size, shape or position (e.g. dose / etch error)

Usage:  python -m src.generate_data --n-per-class 1500
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

CLASSES = ["clean", "missing", "bridged", "deformed"]
SIZE = 96


def _soft(d, edge):
    """Soft step: 1 inside (d<0), 0 outside, with an edge-width transition."""
    return 1.0 / (1.0 + np.exp(np.clip(d / edge, -30, 30)))


def _segment_distance(xs, ys, p, q):
    px, py = p
    qx, qy = q
    dx, dy = qx - px, qy - py
    t = np.clip(((xs - px) * dx + (ys - py) * dy) / (dx * dx + dy * dy + 1e-9), 0, 1)
    return np.hypot(xs - (px + t * dx), ys - (py + t * dy))


def render(cls, rng, size=SIZE):
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    pitch = rng.uniform(15, 24)
    radius = rng.uniform(0.26, 0.36) * pitch
    off = rng.uniform(0, pitch, 2)
    angle = rng.normal(0, 0.03)  # slight lattice rotation
    edge = rng.uniform(0.8, 1.6)
    brightness = rng.uniform(0.7, 1.0)

    n = int(size / pitch) + 3
    pillars = {}
    for i in range(-1, n):
        for j in range(-1, n):
            x0, y0 = i * pitch, j * pitch
            x = off[0] + x0 * np.cos(angle) - y0 * np.sin(angle)
            y = off[1] + x0 * np.sin(angle) + y0 * np.cos(angle)
            if -pitch < x < size + pitch and -pitch < y < size + pitch:
                pillars[(i, j)] = dict(
                    x=x + rng.normal(0, 0.03 * pitch),
                    y=y + rng.normal(0, 0.03 * pitch),
                    rx=radius * rng.normal(1, 0.04),
                    ry=radius * rng.normal(1, 0.04),
                )

    inside = [k for k, p in pillars.items() if 4 < p["x"] < size - 4 and 4 < p["y"] < size - 4]
    bridges = []

    if cls == 1:  # missing pillars
        for k in rng.choice(len(inside), size=min(len(inside), rng.integers(1, 4)), replace=False):
            pillars.pop(inside[k], None)
    elif cls == 2:  # bridges between neighbours
        for k in rng.choice(len(inside), size=min(len(inside), rng.integers(1, 4)), replace=False):
            i, j = inside[k]
            di, dj = [(1, 0), (0, 1)][rng.integers(0, 2)]
            nb = (i + di, j + dj)
            if nb in pillars and (i, j) in pillars:
                bridges.append((pillars[(i, j)], pillars[nb]))
    elif cls == 3:  # deformed pillars
        for k in rng.choice(len(inside), size=min(len(inside), rng.integers(2, 5)), replace=False):
            p = pillars[inside[k]]
            p["rx"] *= rng.uniform(0.55, 1.7)
            p["ry"] *= rng.uniform(0.55, 1.7)
            p["x"] += rng.normal(0, 0.18 * pitch)
            p["y"] += rng.normal(0, 0.18 * pitch)

    img = np.zeros((size, size), np.float32)
    for p in pillars.values():
        d = np.hypot((xs - p["x"]) / p["rx"], (ys - p["y"]) / p["ry"]) - 1.0
        img = np.maximum(img, _soft(d * radius, edge))
    for a, b in bridges:
        w = rng.uniform(0.35, 0.6) * radius
        d = _segment_distance(xs, ys, (a["x"], a["y"]), (b["x"], b["y"])) - w
        img = np.maximum(img, _soft(d, edge))

    # SEM-like appearance: edge brightening, background texture, blur, shot + read noise
    edges = np.abs(gaussian_filter(img, 1.2) - img)
    img = brightness * (0.75 * img + 1.5 * edges)
    img += 0.12 + 0.3 * gaussian_filter(rng.normal(size=img.shape), 3)
    img = gaussian_filter(img, rng.uniform(0.6, 1.2))
    img = rng.poisson(np.clip(img, 0, None) * 40) / 40.0
    img += rng.normal(0, rng.uniform(0.02, 0.06), img.shape)
    img = (img - img.min()) / (img.max() - img.min() + 1e-9)
    return (img * 255).astype(np.uint8)


def build(n_per_class, seed):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(len(CLASSES)):
        for _ in range(n_per_class):
            X.append(render(c, rng))
            y.append(c)
    X, y = np.stack(X), np.array(y, np.int64)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/sem_defects.npz")
    args = ap.parse_args()
    X, y = build(args.n_per_class, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y, classes=np.array(CLASSES))
    print(f"Saved {len(y)} images to {args.out}  shape={X.shape}")


if __name__ == "__main__":
    main()
