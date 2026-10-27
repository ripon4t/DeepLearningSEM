"""Save a grid of example images per class (used in the README)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .generate_data import CLASSES, render

rng = np.random.default_rng(7)
fig, axes = plt.subplots(len(CLASSES), 5, figsize=(10, 8))
for r, name in enumerate(CLASSES):
    for c in range(5):
        axes[r, c].imshow(render(r, rng), cmap="gray", vmin=0, vmax=255)
        axes[r, c].axis("off")
    axes[r, 0].set_title(name, loc="left", fontsize=11)
plt.tight_layout()
plt.savefig("assets/samples.png", dpi=110)
