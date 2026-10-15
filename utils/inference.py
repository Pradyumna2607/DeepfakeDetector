"""
DeepfakeInferenceEngine
=========================
The single entry point the Flask app uses to analyze an uploaded video.
It handles:

  1. Frame sampling + face extraction (utils/face_extraction.py)
  2. Saving extracted frames & faces into static/results/<job_id>/ so the
     web UI can display them (mirrors the original project's "Extracted
     Frames" / "Extracted Faces" gallery, see User Manual section 5.2)
  3. Classifying the video using a three-tier fallback chain, so the app
     always gives the best available answer:

       Tier 1: Trained CNN-LSTM checkpoint (models/cnn_lstm.py)
               -- used automatically once you run train.py on your own
               labelled dataset. Matches the architecture in the project
               report exactly.

       Tier 2: Pretrained ViT classifier (models/pretrained_classifier.py)
               -- a publicly available, already-trained real-vs-fake face
               classifier (Hugging Face). Gives genuinely learned
               predictions with zero training/dataset work on your part.
               Requires an internet connection the first time it runs
               (to download weights, then it's cached locally).

       Tier 3: Classical-CV heuristic baseline (models/baseline.py)
               -- fully offline, zero dependencies beyond OpenCV. Used
               only if neither of the above is available, so the app
               never breaks entirely.

  4. Returning a uniform result dict + writing meta.json for the result page.
"""

import os
import json
import logging
import cv2
import numpy as np

from utils.face_extraction import extract_frames, crop_largest_face, draw_face_box, get_active_backend
from models.baseline import compute_fake_probability as compute_heuristic_probability

logger = logging.getLogger("deepfake_detector.inference")

# Torch / the trained CNN-LSTM model are optional dependencies. If torch
# isn't installed (or no checkpoint exists yet) we simply never import it.
try:
    import torch
    from models.cnn_lstm import build_model
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# The pretrained ViT classifier is also optional (needs `transformers`).
try:
    from models.pretrained_classifier import is_available as pretrained_is_available
    from models.pretrained_classifier import compute_fake_probability as compute_pretrained_probability
    _PRETRAINED_MODULE_AVAILABLE = True
except ImportError:
    _PRETRAINED_MODULE_AVAILABLE = False


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

METHOD_LABELS = {
    "cnn_lstm": "your trained CNN-LSTM model",
    "pretrained_vit": "a pretrained Vision Transformer (Hugging Face)",
    "ensemble": "an ensemble of the pretrained ViT + classical-CV heuristic",
    "heuristic_baseline": "the classical-CV heuristic baseline",
}


def _preprocess_face_for_cnn(face_bgr, image_size=224):
    face_bgr = cv2.resize(face_bgr, (image_size, image_size))
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    face_rgb = (face_rgb - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(face_rgb, (2, 0, 1))


# How close to the 50/50 boundary a fake-probability has to be before the
# system honestly reports "UNCERTAIN" instead of forcing a confident-sounding
# REAL/FAKE call. A verdict forced out of what's essentially a coin flip is
# far more dangerous in front of an audience than an honest "this one's
# borderline" -- the latter is defensible, the former can blow up on you
# the moment someone tests it live with a different video.
UNCERTAINTY_BAND = 0.12  # fake_prob in [0.38, 0.62] -> reported as UNCERTAIN


def _prob_to_verdict(fake_prob: float) -> tuple:
    """
    Converts a raw fake-probability (0-1) into a (prediction, confidence)
    pair shared by all three inference tiers, so every tier's borderline
    cases are handled identically and honestly instead of each one
    quietly forcing a coin-flip into a confident-looking label.
    """
    if abs(fake_prob - 0.5) < UNCERTAINTY_BAND:
        prediction = "UNCERTAIN"
        # Confidence here reflects "how far from a pure coin flip", capped
        # to stay below the band's own threshold so it never looks like a
        # normal high-confidence call.
        confidence = round(50 + abs(fake_prob - 0.5) * 100, 2)
    else:
        prediction = "FAKE" if fake_prob >= 0.5 else "REAL"
        confidence = round((fake_prob if prediction == "FAKE" else 1 - fake_prob) * 100, 2)
    return prediction, confidence


class DeepfakeInferenceEngine:
    def __init__(self, model_path: str, frame_sample_count: int = 20, device: str = "cpu",
                 target_fps: float = None, max_frames_per_video: int = None):
        self.model_path = model_path
        # frame_sample_count is kept as the fallback/fixed-count value (also
        # used as the CNN-LSTM's expected sequence length for padding), while
        # target_fps + max_frames_per_video drive the new rate-based sampling
        # used for the pretrained/heuristic tiers.
        self.frame_sample_count = frame_sample_count
        self.target_fps = target_fps
        self.max_frames_per_video = max_frames_per_video or frame_sample_count
        self.device = device
        self.model = None
        self.method = "heuristic_baseline"

        if _TORCH_AVAILABLE and os.path.exists(model_path):
            self._load_trained_model()

        if self.method == "heuristic_baseline" and _PRETRAINED_MODULE_AVAILABLE:
            # Only probe availability (which may trigger a one-time model
            # download) if we don't already have the trained model active.
            if pretrained_is_available():
                self.method = "pretrained_vit"
                logger.info("No trained CNN-LSTM checkpoint found -- using the "
                            "pretrained ViT deepfake classifier instead.")

        if self.method == "heuristic_baseline":
            logger.info("No trained checkpoint and no pretrained classifier available. "
                        "Using the classical-CV heuristic baseline (%s). "
                        "Run train.py once you have a labelled dataset, or install "
                        "`transformers` + connect to the internet once, for better accuracy.",
                        self.model_path)

    def _load_trained_model(self):
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model = build_model(checkpoint.get("config", {}))
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
            self.method = "cnn_lstm"
            self._val_acc = checkpoint.get("val_acc")
            self._val_f1 = checkpoint.get("val_f1")
            logger.info("Loaded trained CNN-LSTM checkpoint (val_acc=%s, val_f1=%s).",
                        self._val_acc, self._val_f1)
        except Exception as exc:
            logger.warning("Failed to load checkpoint (%s). Trying the next tier.", exc)
            self.model = None
            self.method = "heuristic_baseline"

    # ----------------------------------------------------------------
    def get_model_metrics(self):
        """Used by the /accuracy page."""
        metrics_path = os.path.join(os.path.dirname(self.model_path), "training_metrics.json")
        if self.method == "cnn_lstm" and os.path.exists(metrics_path):
            with open(metrics_path) as f:
                data = json.load(f)
            return {
                "method": self.method,
                "method_label": METHOD_LABELS[self.method],
                "trained_at": data.get("trained_at"),
                "best_val_acc": data.get("best_val_acc"),
                "best_val_f1": data.get("best_val_f1"),
                "dataset_size": data.get("dataset_size"),
                "history": data.get("history", []),
            }

        notes = {
            "pretrained_vit": (
                "No trained CNN-LSTM checkpoint found yet, so the system is using a "
                "pretrained Vision Transformer (Hugging Face model "
                "'dima806/deepfake_vs_real_image_detection') fine-tuned specifically on "
                "real-vs-fake face images. This gives genuinely learned predictions "
                "without requiring you to train anything yourself. Run `python train.py` "
                "after populating ./data/real and ./data/fake to switch to your own "
                "trained CNN-LSTM model, which matches the architecture in the project report."
            ),
            "heuristic_baseline": (
                "No trained CNN-LSTM checkpoint and no pretrained classifier available. "
                "The system is currently using the classical-CV heuristic baseline "
                "(Laplacian sharpness, FFT high-frequency ratio, texture entropy, edge "
                "density, temporal consistency) -- an explainable but hand-calibrated "
                "fallback, not a learned model. Install `transformers` (pip install "
                "transformers) and ensure an internet connection to enable the pretrained "
                "classifier tier, or run `python train.py` on your own dataset for the "
                "full CNN-LSTM model."
            ),
        }

        return {
            "method": self.method,
            "method_label": METHOD_LABELS[self.method],
            "trained_at": None,
            "best_val_acc": None,
            "history": [],
            "note": notes.get(self.method, ""),
        }

    # ----------------------------------------------------------------
    def analyze_video(self, video_path: str, results_dir: str, job_id: str, max_seconds: float = None) -> dict:
        logger.info("Analyzing job %s (%s) with method=%s, face_backend=%s, target_fps=%s, max_frames=%s",
                    job_id, os.path.basename(video_path), self.method, get_active_backend(),
                    self.target_fps, self.max_frames_per_video)

        frames = extract_frames(
            video_path,
            n_samples=self.frame_sample_count,
            max_seconds=max_seconds,
            target_fps=self.target_fps,
            max_frames=self.max_frames_per_video,
        )
        if not frames:
            raise ValueError("Could not extract any frames from the uploaded video.")

        frame_files, face_files, face_crops = [], [], []

        for i, frame in enumerate(frames):
            face_crop, bbox = crop_largest_face(frame)

            frame_name = f"frame_{i}.jpg"
            frame_path = os.path.join(results_dir, frame_name)
            annotated = draw_face_box(frame, bbox) if bbox is not None else frame
            cv2.imwrite(frame_path, annotated)
            frame_files.append(frame_name)

            if face_crop is not None and face_crop.size > 0:
                face_name = f"face_{i}.jpg"
                face_path = os.path.join(results_dir, face_name)
                cv2.imwrite(face_path, face_crop)
                face_files.append(face_name)
                face_crops.append(face_crop)

        faces_detected = len(face_crops)

        if faces_detected == 0:
            result = {
                "prediction": "UNKNOWN",
                "confidence": 0.0,
                "faces_detected": 0,
                "method": self.method,
                "message": "No human faces were detected in the sampled frames.",
            }
            per_frame_scores = []
        elif self.method == "cnn_lstm":
            result, per_frame_scores = self._predict_with_cnn_lstm(face_crops)
            result["faces_detected"] = faces_detected
            result["method"] = self.method
        elif self.method == "pretrained_vit":
            result, per_frame_scores = self._predict_with_pretrained(face_crops)
            result["faces_detected"] = faces_detected
            result["method"] = result.pop("method_override", self.method)
        else:
            result, per_frame_scores = self._predict_with_heuristic(face_crops)
            result["faces_detected"] = faces_detected
            result["method"] = self.method

        result["face_detector"] = get_active_backend()

        # Persist metadata so the /result page can be revisited / refreshed
        meta = {
            "frame_files": frame_files,
            "face_files": face_files,
            "per_frame_scores": per_frame_scores,
            "result": result,
        }
        with open(os.path.join(results_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        return result

    # ----------------------------------------------------------------
    def _predict_with_cnn_lstm(self, face_crops):
        tensors = [_preprocess_face_for_cnn(f) for f in face_crops]

        target_len = self.max_frames_per_video
        while len(tensors) < target_len:
            tensors.append(tensors[-1])
        tensors = tensors[:target_len]

        clip = np.stack(tensors, axis=0)[np.newaxis, ...]  # (1, T, C, H, W)
        clip_tensor = torch.from_numpy(clip).float().to(self.device)

        with torch.no_grad():
            fake_prob = float(self.model.predict_proba(clip_tensor).item())

        prediction, confidence = _prob_to_verdict(fake_prob)

        result = {"prediction": prediction, "confidence": confidence}
        # A genuine per-frame LSTM attention score would need extra
        # instrumentation; approximate with the overall probability
        # repeated so the UI's per-frame chart still has data to show.
        per_frame_scores = [fake_prob] * len(face_crops)
        return result, per_frame_scores

    def _predict_with_pretrained(self, face_crops):
        try:
            pretrained = compute_pretrained_probability(face_crops)
        except Exception as exc:
            logger.warning("Pretrained classifier failed at inference time (%s); "
                            "falling back to heuristic baseline for this request.", exc)
            return self._predict_with_heuristic(face_crops)

        # Also compute the classical-CV heuristic and BLEND the two signals
        # rather than trusting the pretrained model alone. The pretrained
        # ViT was trained on GAN-generated face datasets, which don't
        # perfectly match every real-world manipulation style (e.g.
        # face-swap videos like DFDC look different from pure GAN faces).
        # The heuristic uses independent, hand-inspectable cues (sharpness,
        # frequency content, texture, edges) that catch different failure
        # modes -- combining them is more robust than either alone.
        try:
            heuristic = compute_heuristic_probability(face_crops)
            heuristic_prob = heuristic["fake_probability"]
            feature_summary = heuristic["feature_summary"]
        except Exception as exc:
            logger.warning("Heuristic computation failed during ensembling (%s); "
                            "using the pretrained score alone.", exc)
            heuristic_prob = None
            feature_summary = {}

        pretrained_prob = pretrained["fake_probability"]
        pretrained_per_frame = pretrained["per_frame_fake_scores"]

        if heuristic_prob is not None:
            # Weighted toward the pretrained model (it's a learned classifier)
            # but the heuristic still meaningfully pulls the result when the
            # two disagree, instead of being silently discarded.
            fake_prob = 0.7 * pretrained_prob + 0.3 * heuristic_prob
            method_used = "ensemble"
            per_frame_scores = pretrained_per_frame  # chart still reflects per-frame detail
        else:
            fake_prob = pretrained_prob
            method_used = "pretrained_vit"
            per_frame_scores = pretrained_per_frame

        prediction, confidence = _prob_to_verdict(fake_prob)

        result = {
            "prediction": prediction,
            "confidence": confidence,
            "method_override": method_used,
            "feature_summary": {
                **feature_summary,
                "pretrained_fake_probability": round(pretrained_prob, 4),
                "heuristic_fake_probability": round(heuristic_prob, 4) if heuristic_prob is not None else None,
            },
        }
        return result, per_frame_scores

    def _predict_with_heuristic(self, face_crops):
        heuristic = compute_heuristic_probability(face_crops)
        fake_prob = heuristic["fake_probability"]
        per_frame_scores = heuristic["per_frame_fake_scores"]

        prediction, confidence = _prob_to_verdict(fake_prob)

        result = {
            "prediction": prediction,
            "confidence": confidence,
            "feature_summary": heuristic["feature_summary"],
        }
        return result, per_frame_scores
