from pathlib import Path
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


OUTPUT_DIR = Path("backend/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def process_audio(audio_path: str):
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)

    duration = librosa.get_duration(y=y, sr=sr)

    # -------- Waveform --------
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform")

    waveform_path = OUTPUT_DIR / "waveform.png"
    plt.savefig(waveform_path)
    plt.close()

    # -------- Spectrogram --------
    plt.figure(figsize=(10, 4))

    spectrogram = librosa.amplitude_to_db(
        np.abs(librosa.stft(y)),
        ref=np.max
    )

    librosa.display.specshow(
        spectrogram,
        sr=sr,
        x_axis="time",
        y_axis="log"
    )

    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")

    spectrogram_path = OUTPUT_DIR / "spectrogram.png"
    plt.savefig(spectrogram_path)
    plt.close()

    return {
        "sample_rate": sr,
        "duration_seconds": round(duration, 2),
        "number_of_samples": len(y),
        "waveform": str(waveform_path),
        "spectrogram": str(spectrogram_path),
    }