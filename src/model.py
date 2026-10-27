import torch
import torch.nn as nn


def block(c_in, c_out, pool):
    layers = [
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class DefectCNN(nn.Module):
    """Compact VGG-style CNN. Last conv block is un-pooled (12x12 for 96x96 input)
    so Grad-CAM heatmaps keep useful spatial resolution."""

    def __init__(self, n_classes=4, widths=(32, 64, 128, 128), dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            block(1, widths[0], True),
            block(widths[0], widths[1], True),
            block(widths[1], widths[2], True),
            block(widths[2], widths[3], False),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(widths[3], n_classes)
        )

    def forward(self, x):
        return self.head(self.features(x))


if __name__ == "__main__":
    m = DefectCNN()
    print(m(torch.zeros(2, 1, 96, 96)).shape, sum(p.numel() for p in m.parameters()) / 1e6, "M params")
