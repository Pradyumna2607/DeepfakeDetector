"""
Pretrained Deepfake Image Classifier (Hugging Face)
======================================================
This gives you *genuinely learned* predictions -- not a hand-calibrated
heuristic -- without needing to collect a dataset or train anything
yourself. It uses a publicly available image-classification model that
has already been fine-tuned specifically on real vs. fake/deepfake face
images.

Default model:
    "dima806/deepfake_vs_real_image_detection"
    A Vision Transformer (ViT) fine-tuned on a large set of real and
    GAN/deepfake-generated face images.

How it fits the pipeline:
    Each sampled video frame's face crop (from utils/face_extraction.py)
    is classified independently by the ViT model, and the per-frame fake
    probabilities are averaged into one video-level verdict -- the same
    aggregation pattern used by models/baseline.py, so the rest of the
    app (result page, per-frame chart, etc.) doesn't need to change.

Requirements:
    pip install transformers torch pillow

First-run note:
    The first time this runs, it downloads the model weights (a few
    hundred MB) from huggingface.co and caches them locally
    (~/.cache/huggingface). This requires an internet connection ONCE;
    every run after that loads from the local cache and works offline.
"""

import logging
import numpy as np
import cv2

logger = logging.getLogger("deepfake_detector.pretrained_classifier")

DEFAULT_MODEL_ID = "dima806/deepfake_vs_real_image_detection"

_classifier = None
_load_attempted = False
_load_error = None


def _get_classifier():
    """Lazily loads the HF pipeline once, and remembers failures so we
    don't retry a slow/broken download on every single request."""
    global _classifier, _load_attempted, _load_error

    if _classifier is not None:
        return _classifier
    if _load_attempted:
        return None  # already tried and failed this process; don't retry per-request

    _load_attempted = True
    try:
        from transformers import pipeline
        logger.info("Loading pretrained deepfake classifier '%s' "
                    "(first run downloads weights, then caches locally)...", DEFAULT_MODEL_ID)
        _classifier = pipeline("image-classification", model=DEFAULT_MODEL_ID)
        logger.info("Pretrained deepfake classifier loaded successfully.")
    except Exception as exc:
        _load_error = exc
        logger.warning("Pretrained classifier unavailable (%s). "
                        "Falling back to the next tier.", exc)
        _classifier = None

    return _classifier


def is_available() -> bool:
    """Cheap check used by the inference engine to decide whether this
    tier can be used at all (without forcing a load if already failed)."""
    if _classifier is not None:
        return True
    if _load_attempted and _load_error is not None:
        return False
    return _get_classifier() is not None


def _face_to_pil(face_bgr):
    from PIL import Image
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _extract_fake_score(predictions) -> float:
    """
    HF image-classification pipelines return a list like:
        [{"label": "Fake", "score": 0.93}, {"label": "Real", "score": 0.07}]
    Label text varies by model ("fake"/"real", "deepfake"/"real", etc.),
    so we match case-insensitively for anything that looks like the
    "fake" class rather than hardcoding exact label strings.
    """
    fake_score = 0.0
    for pred in predictions:
        label = str(pred.get("label", "")).lower()
        if "fake" in label or "deepfake" in label or "manipulat" in label or "synthetic" in label:
            fake_score = max(fake_score, float(pred["score"]))
    return fake_score


def compute_fake_probability(face_crops: list) -> dict:
    """
    Same return schema as models/baseline.py's compute_fake_probability,
    so utils/inference.py can treat both tiers interchangeably.
    """
    clf = _get_classifier()
    if clf is None:
        raise RuntimeError(f"Pretrained classifier unavailable: {_load_error}")

    per_frame_scores = []
    for face_bgr in face_crops:
        try:
            pil_img = _face_to_pil(face_bgr)
            predictions = clf(pil_img)
            per_frame_scores.append(_extract_fake_score(predictions))
        except Exception as exc:
            logger.warning("Pretrained classifier failed on one frame (%s); scoring it neutral.", exc)
            per_frame_scores.append(0.5)

    per_frame_scores = np.array(per_frame_scores)
    overall = float(np.mean(per_frame_scores)) if len(per_frame_scores) else 0.5

    return {
        "fake_probability": overall,
        "per_frame_fake_scores": per_frame_scores.tolist(),
        "feature_summary": {
            "model_id": DEFAULT_MODEL_ID,
            "num_frames_scored": len(face_crops),
        },
    }
