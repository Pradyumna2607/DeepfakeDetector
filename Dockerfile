# DeepfakeDetector — production container image
FROM python:3.11-slim

# OpenCV needs these system libs to import correctly on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Torch/torchvision make the image large & slow to build. If you only need
# the heuristic baseline (no trained model yet), comment those two lines
# out of requirements.txt before building to get a much smaller image.
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p instance static/uploads static/results saved_models data/real data/fake

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["gunicorn", "wsgi:app", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "180"]
