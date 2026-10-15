# DeepfakeDetector

**Deepfake detection for human faces in videos** — a full-stack Flask application combining a
**CNN-LSTM** (ResNet-18 + bidirectional LSTM) architecture with a pretrained classifier and an
offline classical-CV baseline, so it works end-to-end from the first run.

[![CI](https://github.com/YOUR_USERNAME/DeepfakeDetector/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/DeepfakeDetector/actions)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Flask](https://img.shields.io/badge/flask-web%20app-lightgrey)
![PyTorch](https://img.shields.io/badge/pytorch-optional-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)

> Academic major project — *Deepfake Video Detection* (Group 12, Department of Emerging Technologies).

---

## Table of contents
- [Features](#features)
- [How it works](#how-it-works)
- [Tech stack](#tech-stack)
- [Quick start](#quick-start)
- [Training your own model](#training-your-own-model)
- [Docker and deployment](#docker-and-deployment)
- [Configuration](#configuration)
- [Project structure](#project-structure)
- [Testing](#testing)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Contributing, security, license](#contributing-security-license)

## Features
- Secure user registration/login (Flask-Login, PBKDF2 password hashing)
- Video upload (mp4, avi, mov, mkv, webm; up to 200 MB) with per-user detection history
- Frame sampling and face extraction (MTCNN if installed, OpenCV Haar Cascade fallback)
- Verdict (**REAL / FAKE / UNCERTAIN**) with confidence, per-frame suspicion chart, and extracted frame/face galleries
- JSON endpoint `POST /api/predict` for programmatic use
- Automatic model fallback chain so the app never breaks when a dependency is missing

## How it works

```
Video -> frame sampling (10 fps, max 150 frames) -> face detection & crop
      -> Tier 1: trained CNN-LSTM checkpoint        (saved_models/deepfake_cnn_lstm.pt)
      -> Tier 2: pretrained ViT image classifier    (Hugging Face, downloaded on first use)
      -> Tier 3: classical-CV heuristic baseline    (Laplacian sharpness, FFT high-frequency energy,
                                                      texture entropy, temporal variance)
      -> verdict + confidence + per-frame scores
```

The highest available tier is used automatically.

**CNN-LSTM:** ResNet-18 encoder (256-d per-frame embedding) -> bidirectional LSTM (256 hidden units)
-> 2-layer MLP head -> sigmoid P(fake). Trained with `BCEWithLogitsLoss`, Adam and `ReduceLROnPlateau`.

## Tech stack
Python, Flask, Flask-SQLAlchemy, Flask-Login, OpenCV, NumPy, PyTorch / torchvision, facenet-pytorch (MTCNN),
Hugging Face Transformers, SQLite/PostgreSQL, Gunicorn, Docker.

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/DeepfakeDetector.git
cd DeepfakeDetector
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # then edit SECRET_KEY
python app.py
```

Open http://127.0.0.1:5000, register, log in and upload a video.
On Windows you can also double-click `run_windows.bat`.

> **Lightweight install:** to skip the large PyTorch/Transformers downloads, install only
> Flask, Flask-SQLAlchemy, Flask-Login, Werkzeug, opencv-python and numpy. The app runs on the heuristic baseline.

## Training your own model

1. Download labelled data (e.g. [FaceForensics++](https://github.com/ondyari/FaceForensics),
   [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics),
   [DFDC](https://www.kaggle.com/c/deepfake-detection-challenge)).
2. Sort it into `data/real/` and `data/fake/`:
   ```bash
   python prepare_dataset.py --source-dir ./downloads/dfdc_sample --metadata metadata.json
   python prepare_dataset.py --source-dir ./downloads/celeb_df --csv labels.csv --prefix celebdf_
   ```
3. Train:
   ```bash
   python train.py --data-root ./data --epochs 15 --batch-size 4
   ```
4. Restart the app — the checkpoint is detected automatically.

| Flag | Default | Meaning |
|---|---|---|
| `--epochs` | 15 | Training epochs |
| `--batch-size` | 4 | Videos per batch |
| `--n-frames` | 20 | Frames sampled per video |
| `--lr` | 1e-4 | Adam learning rate |
| `--val-split` | 0.2 | Stratified validation fraction |
| `--early-stop-patience` | 5 | Epochs without val-F1 gain before stopping (0 disables) |
| `--resume` | none | Checkpoint to continue from |
| `--pretrained-cnn` | off | Warm-start ResNet-18 from ImageNet |

Training features: stratified split, class-weighted loss, accuracy/precision/recall/F1 per epoch,
best-checkpoint selection by F1, early stopping and resume. Metrics are written to
`saved_models/training_metrics.json` and shown on the app's Accuracy page.

## Docker and deployment

```bash
docker build -t deepfake-detector .
docker run -p 8000:8000 -e SECRET_KEY=change-me deepfake-detector
```

`Procfile` (Heroku-style) and `render.yaml` (Render) are included. For production use gunicorn behind
an HTTPS reverse proxy and set `FORCE_HTTPS_COOKIES=true`.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | dev key | **Must** be changed in production |
| `DATABASE_URL` | local SQLite | Postgres URL supported (`postgres://` auto-normalised) |
| `FLASK_DEBUG` | `0` | Set `1` for development only |
| `HOST` / `PORT` | `127.0.0.1` / `5000` | Dev server bind address |
| `FORCE_HTTPS_COOKIES` | `false` | Secure cookies when behind HTTPS |

Frame sampling, upload limits and the device (`cpu`/`cuda`) are set in `config.py`.

## Project structure

```
DeepfakeDetector/
├── app.py                  # Flask routes, auth, upload pipeline
├── wsgi.py                 # Production entry point
├── config.py               # Configuration
├── train.py                # CNN-LSTM training
├── dataset.py              # PyTorch Dataset (video -> face sequence)
├── prepare_dataset.py      # Sort datasets into data/real and data/fake
├── models/                 # cnn_lstm.py, pretrained_classifier.py, baseline.py
├── utils/                  # face_extraction.py, inference.py
├── templates/  static/     # Jinja2 UI, CSS, JS
├── tests/                  # pytest suite
├── Dockerfile  Procfile  render.yaml
└── .github/                # CI, issue and PR templates
```

## Testing

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Limitations
- Detection accuracy depends heavily on training data diversity; no detector is 100% reliable,
  which is why an `UNCERTAIN` verdict exists.
- The heuristic baseline is a demonstration fallback, not a state-of-the-art detector.
- Only the first 2 minutes of each video are analysed.
- Results should not be used as sole evidence in legal or high-stakes decisions.

## Roadmap
- [ ] Audio branch (MFCC) for multimodal detection
- [ ] Grad-CAM explainability on the CNN encoder
- [ ] Adversarial training (FGSM/PGD)
- [ ] Rate limiting and background job queue
- [ ] Real-time webcam / streaming mode

## Contributing, security, license
See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md) and [LICENSE](LICENSE) (MIT).
