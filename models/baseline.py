"""
Classical-CV Heuristic Baseline Detector
==========================================
This baseline is the reason the whole application works "out of the box"
without requiring you to first train the CNN-LSTM network on a large
labelled dataset (which needs GPUs, hours of training, and datasets like
FaceForensics++ / DFDC that are gigabytes in size).

It implements well-known forensic cues from the literature review
(Section 2 of the project report) using only OpenCV + NumPy:

  1. Laplacian sharpness variance   -- GAN-generated faces are often
     slightly blurrier / over-smoothed than camera-captured faces.
  2. High-frequency spectral energy (FFT) -- deepfake generators tend to
     under-represent very high spatial frequencies (upsampling artifacts),
     a cue used in "Fighting Deepfake by Exposing Convolutional Traces"
     (Guarnera et al., referenced in the Literature Review).
  3. Local Binary Pattern-like texture irregularity -- captures unnatural
     skin-texture smoothing common to face-swap generators.
  4. Edge density ratio -- face-swap blending seams and over-smoothed skin
     both reduce the density of strong (Canny) edges relative to a
     genuine camera-captured face.
  5. Frame-to-frame inconsistency -- variance of the above cues *across*
     the sampled frames of the same video (temporal inconsistency is the
     RNN/LSTM's job in the full deep model; here we approximate it with a
     simple statistical measure across frames).

The five cues are combined with hand-calibrated weights into a single
fake-probability score in [0, 1]. This will not match the accuracy of a
properly trained CNN-LSTM on a large dataset, but it is a legitimate,
explainable, fully-offline signal so the demo is never a placeholder.

Swap this out automatically for the trained deep model the moment a
checkpoint exists at `saved_models/deepfake_cnn_lstm.pt` (see
utils/inference.py for the automatic fallback logic).
"""

import cv2
import numpy as np


def _laplacian_sharpness(gray: np.ndarray) -> float:
    """Higher variance = sharper / more high-frequency detail."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _fft_high_freq_ratio(gray: np.ndarray) -> float:
    """
    Ratio of energy in the outer (high-frequency) region of the 2D FFT
    magnitude spectrum to the total energy. GAN upsampling artifacts
    typically suppress genuine high-frequency detail relative to real
    camera sensor noise.
    """
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    radius = min(cy, cx) // 3   # inner circle = "low frequency" region

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    low_freq_mask = dist <= radius
    high_freq_mask = ~low_freq_mask

    total_energy = magnitude.sum() + 1e-8
    high_energy = magnitude[high_freq_mask].sum()
    return float(high_energy / total_energy)


def _texture_irregularity(gray: np.ndarray) -> float:
    """
    A lightweight Local-Binary-Pattern-style texture descriptor:
    compares each pixel to its 8 neighbours and measures the entropy
    of the resulting binary-pattern histogram. Over-smoothed
    (face-swapped) skin tends to reduce texture entropy.
    """
    gray = cv2.resize(gray, (128, 128))
    center = gray[1:-1, 1:-1]
    neighbours = [
        gray[0:-2, 0:-2], gray[0:-2, 1:-1], gray[0:-2, 2:],
        gray[1:-1, 0:-2],                    gray[1:-1, 2:],
        gray[2:, 0:-2], gray[2:, 1:-1], gray[2:, 2:],
    ]
    lbp_code = np.zeros_like(center, dtype=np.uint8)
    for i, n in enumerate(neighbours):
        lbp_code |= ((n >= center).astype(np.uint8) << i)

    hist, _ = np.histogram(lbp_code, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy = float(-(hist * np.log2(hist)).sum())
    return entropy


def _edge_density(gray: np.ndarray) -> float:
    """
    Fraction of pixels that are strong Canny edges. Face-swap blending
    seams around the jawline/hairline and over-smoothed GAN skin both
    tend to *reduce* fine edge density compared to genuine footage,
    which retains natural skin-pore and hair-strand detail.
    """
    edges = cv2.Canny(gray, threshold1=80, threshold2=160)
    return float(np.count_nonzero(edges)) / float(edges.size)


def _motion_magnitude(prev_face_bgr: np.ndarray, curr_face_bgr: np.ndarray) -> float:
    """
    Rough proxy for how much motion occurred between two consecutive face
    crops (both resized to a common size so minor alignment jitter from
    the face detector doesn't dominate the reading).

    This exists to fix a real failure mode: natural motion blur (someone
    talking, turning their head) reduces sharpness/edge-density/high-freq
    content in genuine footage the same way GAN over-smoothing does to a
    manipulated frame. Without accounting for motion, a person simply
    moving gets mistaken for a manipulation artifact. High motion here
    means "this frame's blur is expected," not "trust it less."
    """
    size = (96, 96)
    a = cv2.cvtColor(cv2.resize(prev_face_bgr, size), cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(cv2.resize(curr_face_bgr, size), cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.abs(a - b).mean())


def score_face_crop(face_bgr: np.ndarray) -> dict:
    """
    Computes all heuristic cues for a single face crop.
    Returns a dict of raw feature values (not yet a probability).
    """
    face_bgr = cv2.resize(face_bgr, (224, 224))
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

    return {
        "sharpness": _laplacian_sharpness(gray),
        "hf_ratio": _fft_high_freq_ratio(gray),
        "texture_entropy": _texture_irregularity(gray),
        "edge_density": _edge_density(gray),
    }


def _normalize(value, lo, hi):
    """Clamp+scale a raw feature into [0, 1] given empirically reasonable bounds."""
    return float(np.clip((value - lo) / (hi - lo + 1e-8), 0.0, 1.0))


def _normalize_array(values: np.ndarray, lo, hi) -> np.ndarray:
    """Vectorized version of _normalize for use on whole numpy arrays at once."""
    return np.clip((values - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def compute_fake_probability(face_crops: list) -> dict:
    """
    Aggregates per-frame heuristic cues across an entire video's sampled
    face crops into one overall fake-probability + a per-frame score list
    (used to draw a confidence-over-time chart in the UI).

    Args:
        face_crops: list of BGR np.ndarray face crops (one per sampled frame)

    Returns:
        {
          "fake_probability": float in [0,1],
          "per_frame_fake_scores": [float, ...],
          "feature_summary": {...}
        }
    """
    if not face_crops:
        return {
            "fake_probability": 0.5,
            "per_frame_fake_scores": [],
            "feature_summary": {},
        }

    raw_features = [score_face_crop(f) for f in face_crops]

    sharpness_vals = np.array([f["sharpness"] for f in raw_features])
    hf_vals = np.array([f["hf_ratio"] for f in raw_features])
    entropy_vals = np.array([f["texture_entropy"] for f in raw_features])
    edge_vals = np.array([f["edge_density"] for f in raw_features])

    # --- Motion magnitude between consecutive frames ---
    # Bounds calibrated empirically against real talking-head footage:
    # frame-to-frame mean pixel difference typically ranges ~5 (mostly
    # still) to ~30 (fast head/mouth movement) on a 0-255 grayscale scale.
    motion_vals = [0.0]
    for i in range(1, len(face_crops)):
        motion_vals.append(_motion_magnitude(face_crops[i - 1], face_crops[i]))
    motion_vals = np.array(motion_vals)
    # Discount factor in [0, 0.85]: high motion -> heavily discount the
    # blur-sensitive cues below, since natural motion blur and GAN
    # over-smoothing both reduce sharpness/high-freq/edge content, but
    # only one of them is actually suspicious.
    motion_discount = np.clip(_normalize_array(motion_vals, 5, 28), 0.0, 1.0) * 0.85

    # --- Empirically-reasoned normalization bounds ---
    # Lower sharpness, entropy, edge density, and high-freq ratio all
    # *increase* suspicion of manipulation (over-smoothing / blending is a
    # classic GAN & face-swap artifact) -- EXCEPT when explained by motion.
    sharp_suspicion = (1.0 - np.array([_normalize(v, 15, 400) for v in sharpness_vals])) * (1 - motion_discount)
    hf_suspicion = (1.0 - np.array([_normalize(v, 0.15, 0.55) for v in hf_vals])) * (1 - motion_discount)
    edge_suspicion = (1.0 - np.array([_normalize(v, 0.02, 0.18) for v in edge_vals])) * (1 - motion_discount)
    # Texture entropy is not directly caused by motion blur the same way
    # (it reflects GAN skin-smoothing specifically), so it's left unscaled.
    entropy_suspicion = 1.0 - np.array([_normalize(v, 3.0, 6.5) for v in entropy_vals])

    # Temporal inconsistency: unnaturally *high* frame-to-frame variance in
    # sharpness raises suspicion (flickering artifacts between frames) --
    # but only measured across relatively STILL frames. Comparing a sharp
    # still frame to a naturally-blurred moving frame would otherwise look
    # like "flicker" even in genuine footage, so high-motion frames are
    # excluded from this specific check.
    still_mask = motion_vals <= np.percentile(motion_vals, 70) if len(motion_vals) > 2 else np.ones_like(motion_vals, dtype=bool)
    still_sharpness = sharpness_vals[still_mask]
    temporal_variance = float(np.std(still_sharpness)) if len(still_sharpness) > 1 else 0.0
    temporal_suspicion = _normalize(temporal_variance, 5, 120)

    per_frame_scores = (
        0.28 * sharp_suspicion
        + 0.28 * hf_suspicion
        + 0.16 * entropy_suspicion
        + 0.18 * edge_suspicion
        + 0.10 * temporal_suspicion
    )
    per_frame_scores = np.clip(per_frame_scores, 0.0, 1.0)

    overall_score = float(np.mean(per_frame_scores))

    return {
        "fake_probability": overall_score,
        "per_frame_fake_scores": per_frame_scores.tolist(),
        "feature_summary": {
            "mean_sharpness": float(np.mean(sharpness_vals)),
            "mean_hf_ratio": float(np.mean(hf_vals)),
            "mean_texture_entropy": float(np.mean(entropy_vals)),
            "mean_edge_density": float(np.mean(edge_vals)),
            "mean_motion_magnitude": float(np.mean(motion_vals)),
            "temporal_sharpness_std": temporal_variance,
        },
    }
