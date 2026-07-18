from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path("backend/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def process_audio(audio_path: str) -> dict:
    """
    Load an audio file, extract basic information,
    and generate waveform, spectrogram, and MFCC images.
    """

    # Load the audio without changing its original sample rate
    audio_data, sample_rate = librosa.load(audio_path, sr=None)

    if audio_data.size == 0:
        raise ValueError("The audio file does not contain any audio data.")

    duration = librosa.get_duration(
        y=audio_data,
        sr=sample_rate,
    )

    # Extract 13 MFCC features
    mfcc_features = librosa.feature.mfcc(
        y=audio_data,
        sr=sample_rate,
        n_mfcc=13,
    )

    # ---------------- Waveform ----------------
    waveform_path = OUTPUT_DIR / "waveform.png"

    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(
        audio_data,
        sr=sample_rate,
    )
    plt.title("Audio Waveform")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(waveform_path)
    plt.close()

    # ---------------- Spectrogram ----------------
    spectrogram_path = OUTPUT_DIR / "spectrogram.png"

    stft_result = librosa.stft(audio_data)

    spectrogram_db = librosa.amplitude_to_db(
        np.abs(stft_result),
        ref=np.max,
    )

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        spectrogram_db,
        sr=sample_rate,
        x_axis="time",
        y_axis="log",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Audio Spectrogram")
    plt.tight_layout()
    plt.savefig(spectrogram_path)
    plt.close()
    # ---------------- Save MFCC Features ----------------

    feature_path = OUTPUT_DIR.parent / "features" / "mfcc_features.npy"

    feature_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(feature_path, mfcc_features)

    # ---------------- MFCC ----------------
    mfcc_path = OUTPUT_DIR / "mfcc.png"

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mfcc_features,
        sr=sample_rate,
        x_axis="time",
    )
    plt.colorbar()
    plt.title("MFCC Features")
    plt.ylabel("MFCC Coefficient")
    plt.tight_layout()
    plt.savefig(mfcc_path)
    plt.close()

    return {
        "sample_rate": int(sample_rate),
        "duration_seconds": round(float(duration), 2),
        "number_of_samples": int(len(audio_data)),
        "waveform": str(waveform_path),
        "spectrogram": str(spectrogram_path),
        "mfcc_image": str(mfcc_path),
        "mfcc_shape": [
            int(mfcc_features.shape[0]),
            int(mfcc_features.shape[1]),
        ],
        "mfcc_feature_file": str(feature_path),
    }