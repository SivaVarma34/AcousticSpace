from pathlib import Path

import numpy as np
import torch

from backend.models.audio_cnn import AudioCNN


FEATURE_FILE = Path("backend/features/mfcc_features.npy")


def test_model() -> None:
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"MFCC feature file was not found: {FEATURE_FILE}"
        )

    # Load MFCC values: normally shaped as (13, time_frames)
    mfcc_features = np.load(FEATURE_FILE)

    print("Original MFCC shape:", mfcc_features.shape)

    # Convert NumPy data to a PyTorch tensor
    input_tensor = torch.tensor(
        mfcc_features,
        dtype=torch.float32,
    )

    # Add batch and channel dimensions:
    # (13, time) -> (1, 1, 13, time)
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

    print("Model input shape:", input_tensor.shape)

    model = AudioCNN(number_of_classes=2)
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)

    print("Model output shape:", output.shape)
    print("Raw output:", output)
    print("Probabilities:", probabilities)


if __name__ == "__main__":
    test_model()