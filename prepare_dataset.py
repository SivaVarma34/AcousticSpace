from pathlib import Path

import librosa
import numpy as np


DATASET_DIR = Path("dataset")
REAL_DIR = DATASET_DIR / "real"
FAKE_DIR = DATASET_DIR / "fake"

OUTPUT_DIR = Path("backend/training_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_FILE = OUTPUT_DIR / "features.npy"
LABELS_FILE = OUTPUT_DIR / "labels.npy"

SUPPORTED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".ogg",
}

SAMPLE_RATE = 16000
AUDIO_DURATION = 5
NUMBER_OF_MFCCS = 13
NUMBER_OF_SAMPLES = SAMPLE_RATE * AUDIO_DURATION


def extract_mfcc(audio_path: Path) -> np.ndarray:
    audio_data, sample_rate = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    if len(audio_data) < NUMBER_OF_SAMPLES:
        padding_length = NUMBER_OF_SAMPLES - len(audio_data)

        audio_data = np.pad(
            audio_data,
            (0, padding_length),
            mode="constant",
        )
    else:
        audio_data = audio_data[:NUMBER_OF_SAMPLES]

    mfcc_features = librosa.feature.mfcc(
        y=audio_data,
        sr=sample_rate,
        n_mfcc=NUMBER_OF_MFCCS,
    )

    mean = np.mean(mfcc_features)
    standard_deviation = np.std(mfcc_features)

    normalized_mfcc = (
        mfcc_features - mean
    ) / (standard_deviation + 1e-8)

    return normalized_mfcc.astype(np.float32)


def get_audio_files(folder_path: Path) -> list[Path]:
    if not folder_path.exists():
        return []

    return sorted(
        file_path
        for file_path in folder_path.iterdir()
        if (
            file_path.is_file()
            and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    )


def prepare_dataset() -> None:
    features = []
    labels = []

    categories = [
        (REAL_DIR, 0, "Real"),
        (FAKE_DIR, 1, "Fake"),
    ]

    total_processed = 0
    total_failed = 0

    for folder_path, label, category_name in categories:
        audio_files = get_audio_files(folder_path)

        print(
            f"\nFound {len(audio_files)} "
            f"{category_name.lower()} audio file(s)."
        )

        for audio_path in audio_files:
            try:
                mfcc_features = extract_mfcc(audio_path)

                features.append(mfcc_features)
                labels.append(label)

                total_processed += 1

                print(
                    f"Processed: {audio_path.name} "
                    f"-> Label: {label}"
                )

            except Exception as error:
                total_failed += 1

                print(
                    f"Failed: {audio_path.name} "
                    f"-> {error}"
                )

    if not features:
        raise ValueError(
            "No supported audio files were processed. "
            "Add audio files to dataset/real and dataset/fake."
        )

    feature_array = np.stack(features)
    label_array = np.array(labels, dtype=np.int64)

    np.save(FEATURES_FILE, feature_array)
    np.save(LABELS_FILE, label_array)

    real_count = int(np.sum(label_array == 0))
    fake_count = int(np.sum(label_array == 1))

    print("\nDataset preparation completed successfully.")
    print(f"Processed files: {total_processed}")
    print(f"Failed files: {total_failed}")
    print(f"Feature shape: {feature_array.shape}")
    print(f"Label shape: {label_array.shape}")
    print(f"Real samples: {real_count}")
    print(f"Fake samples: {fake_count}")
    print(f"Features saved to: {FEATURES_FILE}")
    print(f"Labels saved to: {LABELS_FILE}")


if __name__ == "__main__":
    prepare_dataset()