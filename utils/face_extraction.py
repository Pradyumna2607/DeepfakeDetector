"""
Frame & face extraction utilities.

Face detection now uses a two-tier strategy:

  1. PRIMARY: MTCNN (facenet-pytorch) -- a proper deep-learning face
     detector. Far more accurate than Haar cascades: handles side
     profiles, partial occlusion, varied lighting, and small/blurry faces
     much better, which directly improves the quality of every face crop
     fed into both the CNN-LSTM model and the heuristic baseline.

  2. FALLBACK: OpenCV Haar Cascade -- used automatically if
     `facenet-pytorch`/`torch` isn't installed, so the app still runs
     fully offline with zero extra dependencies if needed.

The public function signatures (extract_frames, detect_faces,
crop_largest_face, draw_face_box) are unchanged, so nothing else in the
pipeline (dataset.py, inference.py) needs to know which detector is
actually running underneath.
"""

import os
import cv2
import numpy as np

# --------------------------------------------------------------------
# Detector selection (done once, at import time)
# --------------------------------------------------------------------
_MTCNN_DETECTOR = None
_DETECTOR_BACKEND = "haar"  # "mtcnn" or "haar"

try:
    import torch
    from facenet_pytorch import MTCNN
    from PIL import Image

    _MTCNN_DETECTOR = MTCNN(
        keep_all=True,          # return all faces found, we pick the largest ourselves
        min_face_size=40,
        thresholds=[0.6, 0.7, 0.7],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    _DETECTOR_BACKEND = "mtcnn"
    print("[face_extraction] Using MTCNN (facenet-pytorch) for face detection.")
except Exception as exc:
    print(f"[face_extraction] MTCNN unavailable ({exc}); "
          f"falling back to OpenCV Haar Cascade for face detection.")

_CASCADE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
_HAAR_DETECTOR = cv2.CascadeClassifier(_CASCADE_PATH)


def get_active_backend() -> str:
    """Lets the UI/logs report which detector is actually in use."""
    return _DETECTOR_BACKEND


# --------------------------------------------------------------------
# Frame sampling
# --------------------------------------------------------------------
def sample_frame_indices(total_frames: int, n_samples: int) -> list:
    """Pick n_samples frame indices, evenly spaced across the whole video."""
    if total_frames <= 0:
        return []
    n_samples = min(n_samples, total_frames)
    return sorted(set(np.linspace(0, total_frames - 1, num=n_samples, dtype=int).tolist()))


def extract_frames(video_path: str, n_samples: int = 20, max_seconds: float = None,
                    target_fps: float = None, max_frames: int = None):
    """
    Samples frames from the video, either as a FIXED TOTAL COUNT or as a
    RATE PER SECOND OF VIDEO -- the latter scales the number of analyzed
    frames with the video's length, so a 3-second clip and a 60-second
    clip both get proportionally thorough coverage instead of both being
    squeezed into the same fixed frame budget.

    Args:
        video_path: path to the video file.
        n_samples: fixed total frame count to sample (used only when
            `target_fps` is not given -- kept for backward compatibility).
        max_seconds: if set, only consider the first `max_seconds` of the
            video (bounds worst-case processing time on long uploads).
        target_fps: if set, sample at this many frames per second of the
            video's actual duration (e.g. target_fps=10 analyzes 10
            frames for every second of footage) instead of a fixed count.
        max_frames: hard cap on the number of frames ever returned, even
            in fps mode -- protects against a long video at a high
            target_fps producing an impractically large frame count.

    Returns:
        List[np.ndarray] of BGR frames (as read by OpenCV).

    Raises:
        ValueError if the file can't be opened or contains no readable frames
        (corrupted file, unsupported codec, zero-byte upload, etc).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(
            "Could not open this video file. It may be corrupted, empty, "
            "or use a codec that isn't supported by OpenCV on this system."
        )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0

    if max_seconds and fps > 0:
        total_frames = min(total_frames, int(max_seconds * fps))

    if target_fps and fps > 0:
        duration_seconds = total_frames / fps
        n_samples = max(1, int(round(duration_seconds * target_fps)))
        if max_frames:
            n_samples = min(n_samples, max_frames)

    indices = sample_frame_indices(total_frames, n_samples)

    frames = []
    idx_set = set(indices)
    current = 0

    if not idx_set:
        # Fallback: frame count metadata was unreliable (common with some
        # codecs/containers) -- just read sequentially instead.
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
            if len(frames) >= n_samples:
                break
    else:
        max_idx = max(idx_set)
        while current <= max_idx:
            ok, frame = cap.read()
            if not ok:
                break
            if current in idx_set:
                frames.append(frame)
            current += 1

    cap.release()

    if not frames:
        raise ValueError(
            "No readable frames could be extracted from this video. "
            "Please check the file isn't corrupted and try a different format (mp4 recommended)."
        )

    return frames


# --------------------------------------------------------------------
# Face detection
# --------------------------------------------------------------------
def _detect_faces_mtcnn(frame_bgr: np.ndarray):
    """Returns list of (x, y, w, h) boxes using MTCNN."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    boxes, probs = _MTCNN_DETECTOR.detect(pil_img)
    if boxes is None:
        return []

    results = []
    for box, prob in zip(boxes, probs):
        if prob is None or prob < 0.90:
            continue
        x1, y1, x2, y2 = box
        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
        if w > 0 and h > 0:
            results.append((x, y, w, h))
    return results


def _detect_faces_haar(frame_bgr: np.ndarray, min_size=(60, 60)):
    """Returns list of (x, y, w, h) boxes using OpenCV Haar Cascade."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = _HAAR_DETECTOR.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=min_size,
    )
    return list(faces)


def detect_faces(frame_bgr: np.ndarray, min_size=(60, 60)):
    """
    Detects faces in a single BGR frame using whichever backend is active.

    Returns:
        List of (x, y, w, h) bounding boxes.
    """
    if _DETECTOR_BACKEND == "mtcnn":
        try:
            return _detect_faces_mtcnn(frame_bgr)
        except Exception as exc:
            print(f"[face_extraction] MTCNN detection error, using Haar for this frame: {exc}")
            return _detect_faces_haar(frame_bgr, min_size)
    return _detect_faces_haar(frame_bgr, min_size)


def crop_largest_face(frame_bgr: np.ndarray, padding: float = 0.25):
    """
    Detects faces in a frame and returns a cropped, padded image of the
    largest detected face (most likely the primary subject), plus its box.

    Returns:
        (face_crop_bgr or None, bbox or None)
    """
    faces = detect_faces(frame_bgr)
    if not faces:
        return None, None

    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])

    pad_w, pad_h = int(w * padding), int(h * padding)
    H, W = frame_bgr.shape[:2]
    x0 = max(0, x - pad_w)
    y0 = max(0, y - pad_h)
    x1 = min(W, x + w + pad_w)
    y1 = min(H, y + h + pad_h)

    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None, None
    return crop, (x, y, w, h)


def draw_face_box(frame_bgr: np.ndarray, bbox, color=(0, 255, 0), thickness=2):
    """Returns a copy of the frame with the bounding box drawn on it."""
    out = frame_bgr.copy()
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    return out
