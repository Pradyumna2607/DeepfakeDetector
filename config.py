"""Central configuration for the DeepfakeDetector Flask application."""

import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    # --- Flask / security ---
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-this-in-production")

    # --- Database ---
    # Render/Railway/Heroku inject a DATABASE_URL like "postgres://..." for
    # their managed Postgres add-ons, but SQLAlchemy 1.4+ requires the
    # "postgresql://" scheme -- normalize it here so no code changes are
    # needed when you attach a hosted database.
    _raw_db_url = os.environ.get(
        "DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'instance', 'deepfake.db')}"
    )
    if _raw_db_url.startswith("postgres://"):
        _raw_db_url = _raw_db_url.replace("postgres://", "postgresql://", 1)
    SQLALCHEMY_DATABASE_URI = _raw_db_url
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # --- Uploads ---
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
    RESULTS_FOLDER = os.path.join(BASE_DIR, "static", "results")
    MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200 MB max video size
    MAX_VIDEO_SECONDS = 120  # only the first 2 minutes are analyzed, to bound processing time

    # --- Session / cookie security ---
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    # Enable this once you deploy behind HTTPS (it blocks the cookie over
    # plain http, which would break local http://127.0.0.1 testing):
    SESSION_COOKIE_SECURE = os.environ.get("FORCE_HTTPS_COOKIES", "false").lower() == "true"

    # --- Model / inference ---
    # If a trained checkpoint exists here, the CNN-LSTM model is used.
    # Otherwise the system automatically falls back to the pretrained ViT
    # classifier, then the explainable classical-CV heuristic baseline,
    # so the app always works out of the box.
    MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, "saved_models", "deepfake_cnn_lstm.pt")

    # --- Frame sampling ---
    # Frames are sampled at a RATE PER SECOND of the video's actual length,
    # rather than a fixed total count -- so a 3-second clip and a 60-second
    # clip both get proportionally thorough coverage instead of both being
    # squeezed into the same frame budget. This is what prevents brief or
    # rapid manipulation artifacts from being missed on longer videos.
    TARGET_FPS = 10             # analyze 10 frames for every second of video
    MAX_FRAMES_PER_VIDEO = 150  # hard cap regardless of length/fps, so a long
                                  # video can't blow up processing time
                                  # (150 frames ≈ 15s of footage at 10 fps)
    DEVICE = "cpu"             # set to "cuda" if a GPU + torch-cuda build is available
