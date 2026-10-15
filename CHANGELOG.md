# Changelog

## [1.0.0] - 2026
### Added
- Flask web app with authentication, upload, result and dashboard pages
- Three-tier inference chain: trained CNN-LSTM -> pretrained ViT classifier -> classical-CV heuristic baseline
- Training pipeline with stratified split, class-weighted loss, precision/recall/F1 and early stopping
- Dataset preparation utility, Docker, Procfile and Render deployment files
- Unit/smoke tests and GitHub Actions CI
