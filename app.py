"""
DeepfakeDetector - Main Flask Application
===========================================
Deepfake Detection For Human Faces in Videos
Group 12 | Department of Emerging Technologies

This is the main application entry point. It wires together:
  - User authentication (Flask-Login + SQLAlchemy + hashed passwords)
  - Video upload & validation
  - The detection pipeline (utils/inference.py)
  - Result rendering with extracted frames, extracted faces, and
    a confidence-scored prediction (REAL / FAKE)

Run with:
    python app.py
Then open http://127.0.0.1:5000
"""

import os
import uuid
import json
import logging
from datetime import datetime

from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, jsonify, abort
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from config import Config
from utils.inference import DeepfakeInferenceEngine

# --------------------------------------------------------------------------
# Logging (replaces scattered print() calls with real, level-based logs)
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("deepfake_detector")

# --------------------------------------------------------------------------
# App & Extensions setup
# --------------------------------------------------------------------------
app = Flask(__name__)
app.config.from_object(Config)

db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to access the Deepfake Detector."
login_manager.login_message_category = "warning"

# Single, shared instance of the detection engine (loads model / baseline once)
engine = DeepfakeInferenceEngine(
    model_path=app.config["MODEL_WEIGHTS_PATH"],
    frame_sample_count=app.config["MAX_FRAMES_PER_VIDEO"],
    device=app.config["DEVICE"],
    target_fps=app.config["TARGET_FPS"],
    max_frames_per_video=app.config["MAX_FRAMES_PER_VIDEO"],
)


# --------------------------------------------------------------------------
# Database models
# --------------------------------------------------------------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class DetectionRecord(db.Model):
    """Stores a history of every detection run, per user, for the dashboard."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True)
    job_id = db.Column(db.String(64), unique=True, nullable=False)
    original_filename = db.Column(db.String(255))
    prediction = db.Column(db.String(10))          # "REAL" or "FAKE"
    confidence = db.Column(db.Float)                # 0-100
    faces_detected = db.Column(db.Integer, default=0)
    method = db.Column(db.String(30))                # "cnn_lstm" or "heuristic_baseline"
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# --------------------------------------------------------------------------
# Routes: Public pages
# --------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/abstract")
def abstract():
    return render_template("abstract.html")


@app.route("/accuracy")
def accuracy():
    metrics = engine.get_model_metrics()
    return render_template("accuracy.html", metrics=metrics)


# --------------------------------------------------------------------------
# Routes: Authentication
# --------------------------------------------------------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if not username or not email or not password:
            flash("All fields are required.", "danger")
            return redirect(url_for("register"))

        if password != confirm:
            flash("Passwords do not match.", "danger")
            return redirect(url_for("register"))

        if len(password) < 6:
            flash("Password must be at least 6 characters long.", "danger")
            return redirect(url_for("register"))

        if User.query.filter((User.username == username) | (User.email == email)).first():
            flash("Username or email already registered.", "danger")
            return redirect(url_for("register"))

        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        flash("Account created successfully. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        remember = bool(request.form.get("remember"))

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user, remember=remember)
            next_page = request.args.get("next")
            flash(f"Welcome back, {user.username}!", "success")
            return redirect(next_page or url_for("dashboard"))

        flash("Invalid username or password.", "danger")
        return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))


# --------------------------------------------------------------------------
# Routes: Core detection workflow
# --------------------------------------------------------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    history = (
        DetectionRecord.query.filter_by(user_id=current_user.id)
        .order_by(DetectionRecord.created_at.desc())
        .limit(20)
        .all()
    )
    return render_template("dashboard.html", history=history)


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "GET":
        return render_template("upload.html")

    if "video" not in request.files:
        flash("No file part in request.", "danger")
        return redirect(url_for("upload"))

    file = request.files["video"]
    if file.filename == "":
        flash("No file selected.", "danger")
        return redirect(url_for("upload"))

    if not allowed_file(file.filename):
        flash("Unsupported file type. Please upload mp4, avi, mov, mkv or webm.", "danger")
        return redirect(url_for("upload"))

    # Save upload under a unique job id so concurrent users never collide
    job_id = uuid.uuid4().hex[:12]
    filename = secure_filename(file.filename) or "upload.mp4"
    upload_dir = os.path.join(app.config["UPLOAD_FOLDER"], job_id)
    os.makedirs(upload_dir, exist_ok=True)
    video_path = os.path.join(upload_dir, filename)
    file.save(video_path)

    # Reject empty / near-empty saves (e.g. interrupted upload) with a clear
    # message instead of letting OpenCV fail deep inside the pipeline.
    if os.path.getsize(video_path) < 1024:
        flash("The uploaded file appears empty or corrupted. Please try again.", "danger")
        logger.warning("Rejected near-empty upload: %s (job %s)", filename, job_id)
        return redirect(url_for("upload"))

    # Where extracted frames / faces for this job will be written so the
    # browser can display them (must live under /static to be servable).
    results_dir = os.path.join(app.config["RESULTS_FOLDER"], job_id)
    os.makedirs(results_dir, exist_ok=True)

    try:
        result = engine.analyze_video(
            video_path, results_dir, job_id=job_id,
            max_seconds=app.config.get("MAX_VIDEO_SECONDS"),
        )
    except ValueError as exc:
        # Expected, user-facing errors (corrupted file, no frames, etc.)
        logger.warning("Video processing rejected for job %s: %s", job_id, exc)
        flash(str(exc), "danger")
        return redirect(url_for("upload"))
    except Exception:
        # Unexpected internal errors: log the full stack trace for debugging,
        # but never leak internals to the user.
        logger.exception("Unexpected error processing job %s", job_id)
        flash("Something went wrong while analyzing this video. Please try a different file.", "danger")
        return redirect(url_for("upload"))

    record = DetectionRecord(
        user_id=current_user.id,
        job_id=job_id,
        original_filename=filename,
        prediction=result["prediction"],
        confidence=result["confidence"],
        faces_detected=result["faces_detected"],
        method=result["method"],
    )
    db.session.add(record)
    db.session.commit()

    return redirect(url_for("result", job_id=job_id))


@app.route("/result/<job_id>")
@login_required
def result(job_id):
    record = DetectionRecord.query.filter_by(job_id=job_id).first_or_404()

    if record.user_id != current_user.id:
        abort(403)

    results_dir = os.path.join(app.config["RESULTS_FOLDER"], job_id)
    meta_path = os.path.join(results_dir, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    frame_files = sorted(meta.get("frame_files", []))
    face_files = sorted(meta.get("face_files", []))
    feature_summary = meta.get("result", {}).get("feature_summary", {})

    return render_template(
        "result.html",
        record=record,
        job_id=job_id,
        frame_files=frame_files,
        face_files=face_files,
        per_frame_scores=meta.get("per_frame_scores", []),
        feature_summary=feature_summary,
    )


# --------------------------------------------------------------------------
# JSON API (for programmatic / AJAX use, e.g. a future mobile client)
# --------------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    if "video" not in request.files:
        return jsonify({"error": "no video file provided"}), 400

    file = request.files["video"]
    if not allowed_file(file.filename):
        return jsonify({"error": "unsupported file type"}), 400

    job_id = uuid.uuid4().hex[:12]
    filename = secure_filename(file.filename)
    upload_dir = os.path.join(app.config["UPLOAD_FOLDER"], job_id)
    os.makedirs(upload_dir, exist_ok=True)
    video_path = os.path.join(upload_dir, filename)
    file.save(video_path)

    results_dir = os.path.join(app.config["RESULTS_FOLDER"], job_id)
    os.makedirs(results_dir, exist_ok=True)

    result = engine.analyze_video(video_path, results_dir, job_id=job_id)
    result["job_id"] = job_id
    return jsonify(result)


# --------------------------------------------------------------------------
# Error handlers
# --------------------------------------------------------------------------
@app.errorhandler(403)
def forbidden(e):
    return render_template("error.html", code=403, message="Forbidden"), 403


@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", code=404, message="Page not found"), 404


@app.errorhandler(413)
def too_large(e):
    return render_template("error.html", code=413, message="File too large (max 200MB)"), 413


# --------------------------------------------------------------------------
# CLI bootstrap
# --------------------------------------------------------------------------
def create_db():
    with app.app_context():
        db.create_all()
        print("[DeepfakeDetector] Database tables created.")


if __name__ == "__main__":
    create_db()
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    host = os.environ.get("HOST", "127.0.0.1")
    app.run(debug=debug, host=host, port=int(os.environ.get("PORT", 5000)))
