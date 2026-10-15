"""
PyTorch Dataset for training the CNN-LSTM deepfake detector.

Expected directory layout:

    data/
      real/
        video001.mp4
        video002.mp4
        ...
      fake/
        video101.mp4
        video102.mp4
        ...

Each video is turned into a fixed-length sequence of face-cropped,
resized RGB tensors at __getitem__ time. Videos with no detectable face
in a sampled frame fall back to a centre-crop of the raw frame so
training never crashes on a difficult sample.

Recommended public datasets to populate data/real and data/fake with
(not included here due to size / licensing -- download separately):
  - FaceForensics++          https://github.com/ondyari/FaceForensics
  - Celeb-DF                 https://github.com/yuezunli/celeb-deepfakeforensics
  - DFDC (Deepfake Detection Challenge)  https://ai.facebook.com/datasets/dfdc/
"""

import os
import glob
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.face_extraction import extract_frames, crop_largest_face

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _augment_face(face_bgr, rng: random.Random):
    """
    Lightweight, video-consistent augmentation: the SAME random choices
    (flip / brightness / rotation) are applied to every frame of a given
    video by seeding `rng` once per video before calling this per-frame,
    so temporal consistency within a clip is preserved (an un-augmented
    LSTM input would otherwise see fake per-frame "flicker" introduced
    by the augmentation itself, rather than by real manipulation).
    """
    if rng.random() < 0.5:
        face_bgr = cv2.flip(face_bgr, 1)  # horizontal flip

    # Brightness / contrast jitter
    alpha = 1.0 + rng.uniform(-0.15, 0.15)  # contrast
    beta = rng.uniform(-15, 15)              # brightness
    face_bgr = cv2.convertScaleAbs(face_bgr, alpha=alpha, beta=beta)

    # Small rotation to simulate head-pose / camera-angle variation
    angle = rng.uniform(-8, 8)
    h, w = face_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    face_bgr = cv2.warpAffine(face_bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    return face_bgr


def _preprocess_face(face_bgr, image_size=224):
    face_bgr = cv2.resize(face_bgr, (image_size, image_size))
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    face_rgb = (face_rgb - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(face_rgb, (2, 0, 1))  # C, H, W


class DeepfakeVideoDataset(Dataset):
    def __init__(self, data_root: str, n_frames: int = 20, image_size: int = 224,
                 augment: bool = False, transform=None):
        self.n_frames = n_frames
        self.image_size = image_size
        self.augment = augment
        self.transform = transform

        real_videos = sorted(glob.glob(os.path.join(data_root, "real", "*")))
        fake_videos = sorted(glob.glob(os.path.join(data_root, "fake", "*")))

        self.samples = [(p, 0) for p in real_videos] + [(p, 1) for p in fake_videos]
        if not self.samples:
            raise RuntimeError(
                f"No videos found under {data_root}/real or {data_root}/fake. "
                "Populate these folders before training (see dataset.py docstring)."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        frames = extract_frames(video_path, n_samples=self.n_frames)
        face_tensors = []

        # One RNG per video keeps augmentation consistent across all frames
        # of the same clip (see _augment_face docstring).
        rng = random.Random(idx) if self.augment else None

        for frame in frames:
            face_crop, _ = crop_largest_face(frame)
            if face_crop is None or face_crop.size == 0:
                h, w = frame.shape[:2]
                side = min(h, w)
                cy, cx = h // 2, w // 2
                face_crop = frame[
                    max(0, cy - side // 2): cy + side // 2,
                    max(0, cx - side // 2): cx + side // 2,
                ]

            if self.augment:
                face_crop = _augment_face(face_crop, rng)

            face_tensors.append(_preprocess_face(face_crop, self.image_size))

        # Pad with the last frame if the video was shorter than n_frames
        while len(face_tensors) < self.n_frames:
            face_tensors.append(face_tensors[-1] if face_tensors else np.zeros(
                (3, self.image_size, self.image_size), dtype=np.float32))

        clip = np.stack(face_tensors[: self.n_frames], axis=0)  # (T, C, H, W)
        clip_tensor = torch.from_numpy(clip).float()
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return clip_tensor, label_tensor
