"""Music genre classification pipeline translated from CompleteProject.m.

This script extracts audio features for four genres and trains:
1) One-vs-rest logistic regression
2) A simple neural network classifier

Expected folder structure:
    genres/
      blues/*.wav
      metal/*.wav
      country/*.wav
      pop/*.wav
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray


def _zcr(signal: np.ndarray) -> float:
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)
    signs = np.signbit(signal)
    return float(np.mean(signs[1:] != signs[:-1]))


def _rms(signal: np.ndarray) -> float:
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)
    return float(np.sqrt(np.mean(signal**2)))


def _extract_feature_vector(file_path: Path, n_mfcc: int = 13) -> np.ndarray:
    import librosa

    x, sr = librosa.load(file_path.as_posix(), sr=None, mono=True)

    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc).T
    centroid = librosa.feature.spectral_centroid(y=x, sr=sr).T
    rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr).T

    features = np.hstack([mfcc, centroid, rolloff])
    cov_mfcc = np.cov(mfcc, rowvar=False).reshape(-1)

    vector = np.concatenate(
        [
            np.mean(features, axis=0),
            cov_mfcc,
            np.array([_zcr(x), _rms(x)], dtype=np.float64),
        ]
    )
    return vector


def _load_genre_features(genre_dir: Path) -> np.ndarray:
    wav_files = sorted(genre_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {genre_dir}")

    return np.vstack([_extract_feature_vector(w) for w in wav_files])


def build_dataset(base_dir: Path, genres: Iterable[str]) -> Dataset:
    feature_blocks: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for idx, genre in enumerate(genres, start=1):
        block = _load_genre_features(base_dir / genre)
        feature_blocks.append(block)
        labels.append(np.full((block.shape[0],), idx, dtype=np.int64))

    X = np.vstack(feature_blocks)
    y = np.concatenate(labels)
    return Dataset(X=X, y=y)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train genre classifiers.")
    parser.add_argument("--genres-dir", type=Path, default=Path("genres"))
    parser.add_argument(
        "--test-size", type=float, default=0.1, help="Fraction for test split"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.neural_network import MLPClassifier

    genres = ["blues", "metal", "country", "pop"]
    data = build_dataset(args.genres_dir, genres)

    X_train, X_test, y_train, y_test = train_test_split(
        data.X,
        data.y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=data.y,
    )

    lr = OneVsRestClassifier(
        LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")
    )
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)

    print(f"Logistic Regression Test Accuracy: {accuracy_score(y_test, pred_lr) * 100:.2f}")
    print("Logistic Regression Confusion Matrix:")
    print(confusion_matrix(y_test, pred_lr))

    nn = MLPClassifier(
        hidden_layer_sizes=(20,),
        alpha=1.0,
        max_iter=20000,
        random_state=args.seed,
    )
    nn.fit(X_train, y_train)
    pred_nn = nn.predict(X_test)

    print(f"Neural Network Test Accuracy: {accuracy_score(y_test, pred_nn) * 100:.2f}")
    print("Neural Network Confusion Matrix:")
    print(confusion_matrix(y_test, pred_nn))


if __name__ == "__main__":
    main()
