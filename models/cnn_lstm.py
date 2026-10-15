"""
Hybrid CNN-LSTM Deepfake Detection Model
=========================================
Architecture (matches the project's Literature Review / Proposed Work):

    Video --> [Frame Sampling] --> [Face Crop per Frame]
           --> [CNN Spatial Feature Extractor]   (per-frame embedding)
           --> [LSTM Temporal Aggregator]         (models the sequence)
           --> [Fully-Connected Classifier]       (REAL vs FAKE)

The CNN backbone is a ResNet-18 (torchvision), used as a spatial feature
extractor. We deliberately load it with `weights=None` so that this file
runs in fully offline environments too -- when you actually train the
model (see train.py) you can flip `pretrained=True` to warm-start from
ImageNet weights if you have internet access to download them once.

This module is intentionally dependency-light: it only needs torch +
torchvision. If they are not installed, `utils/inference.py` gracefully
falls back to the classical-CV heuristic baseline, so the rest of the
application still works.
"""

import torch
import torch.nn as nn

try:
    import torchvision.models as tv_models
    _TORCHVISION_AVAILABLE = True
except ImportError:
    _TORCHVISION_AVAILABLE = False


class CNNEncoder(nn.Module):
    """ResNet-18-based per-frame spatial feature extractor."""

    def __init__(self, embedding_dim: int = 256, pretrained: bool = False, freeze_backbone: bool = False):
        super().__init__()
        if not _TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required for CNNEncoder. `pip install torchvision`.")

        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.resnet18(weights=weights)

        # Drop the final classification layer; keep everything up to the
        # global average pool so we get a 512-d feature vector per frame.
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.project = nn.Linear(512, embedding_dim)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        # x: (B, C, H, W)
        feats = self.backbone(x)              # (B, 512, 1, 1)
        feats = torch.flatten(feats, 1)        # (B, 512)
        return self.project(feats)             # (B, embedding_dim)


class DeepfakeCNNLSTM(nn.Module):
    """
    Full hybrid model: CNN encodes each frame independently, an LSTM
    consumes the resulting sequence of embeddings, and a small MLP head
    outputs a single fake-probability logit.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        lstm_hidden_dim: int = 256,
        lstm_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.4,
        pretrained_cnn: bool = False,
    ):
        super().__init__()
        self.encoder = CNNEncoder(embedding_dim=embedding_dim, pretrained=pretrained_cnn)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        lstm_out_dim = lstm_hidden_dim * (2 if bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),   # single logit: sigmoid -> P(fake)
        )

    def forward(self, x):
        """
        x: (B, T, C, H, W) -- a batch of frame sequences (already face-cropped
           and resized, e.g. T=20 frames of 224x224).
        Returns:
            logits: (B, 1) raw scores (apply sigmoid for probability of FAKE)
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        frame_embeddings = self.encoder(x)             # (B*T, embedding_dim)
        frame_embeddings = frame_embeddings.view(B, T, -1)

        lstm_out, (h_n, c_n) = self.lstm(frame_embeddings)
        # Use the representation at the final timestep (concat of both
        # directions if bidirectional) as the video-level summary.
        video_repr = lstm_out[:, -1, :]                  # (B, lstm_out_dim)

        logits = self.classifier(video_repr)              # (B, 1)
        return logits

    @torch.no_grad()
    def predict_proba(self, x):
        """Convenience wrapper returning P(fake) in [0, 1] for a batch."""
        self.eval()
        logits = self.forward(x)
        return torch.sigmoid(logits).squeeze(-1)


def build_model(config: dict = None) -> DeepfakeCNNLSTM:
    """Factory used by both train.py and inference.py so hyperparameters
    always stay in sync between training and inference."""
    config = config or {}
    return DeepfakeCNNLSTM(
        embedding_dim=config.get("embedding_dim", 256),
        lstm_hidden_dim=config.get("lstm_hidden_dim", 256),
        lstm_layers=config.get("lstm_layers", 1),
        bidirectional=config.get("bidirectional", True),
        dropout=config.get("dropout", 0.4),
        pretrained_cnn=config.get("pretrained_cnn", False),
    )


if __name__ == "__main__":
    # Quick smoke test: random input through the network
    model = build_model()
    dummy = torch.randn(2, 20, 3, 224, 224)  # batch=2, 20 frames, RGB 224x224
    out = model(dummy)
    print("Output logits shape:", out.shape)          # -> (2, 1)
    print("Predicted P(fake):", model.predict_proba(dummy))
