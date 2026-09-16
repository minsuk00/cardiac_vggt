import torch
from torch import nn


class DangiCenterNet(nn.Module):
    """Fig. 1 architecture. Input (N,1,192,192); output (x,y) in pixels.

    Same-padded 3x3+ReLU side convolutions and ReLU after FC2 are the
    implementation interpretation of the figure. Final regression is linear.
    """

    feature_count = 51840

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.sides = nn.ModuleList()
        previous = 1
        for level, channels in enumerate((4, 8, 12, 16, 20)):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(previous, channels, 3, padding=1), nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            ))
            if level < 4:
                self.sides.append(nn.Sequential(
                    nn.Conv2d(channels, 1, 3, padding=1), nn.ReLU(),
                ))
            previous = channels
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Sequential(nn.Linear(self.feature_count, 256), nn.ReLU(),
                                nn.Linear(256, 2))
        self.apply(self._initialize)

    @staticmethod
    def _initialize(layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 192, 192):
            raise ValueError("Expected images shaped (N, 1, 192, 192)")
        features = []
        for level, block in enumerate(self.blocks):
            images = block(images)
            if level < 4:
                features.append(self.sides[level](images).flatten(1))
                images = self.pool(images)
            else:
                features.append(images.flatten(1))
        return self.fc(torch.cat(features, dim=1))
