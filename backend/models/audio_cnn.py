import torch
from torch import nn


class AudioCNN(nn.Module):
    """
    Basic convolutional neural network for classifying
    MFCC audio features as real or fake.

    Class 0: Real audio
    Class 1: Fake audio
    """

    def __init__(self, number_of_classes: int = 2):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Produces the same final size for different audio lengths
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(64, number_of_classes),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(input_tensor)
        output = self.classifier(features)
        return output