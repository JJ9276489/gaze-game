from dataclasses import asdict, dataclass

import torch
from torch import nn

from shared_gaze.config import (
    EYE_CROP_HEIGHT,
    EYE_CROP_WIDTH,
    VISION_EYE_GEOMETRY_FEATURE_KEYS,
    VISION_HEAD_FEATURE_KEYS,
)


@dataclass
class EyeCropModelConfig:
    encoder_channels: tuple[int, ...] = (16, 32, 64, 64)
    encoder_type: str = "cnn"
    encoder_pooling: str = "avg"
    eye_coord_channels: bool = False
    head_hidden_dims: tuple[int, ...] = (32, 32)
    extra_feature_keys: tuple[str, ...] = ()
    extra_hidden_dims: tuple[int, ...] = (32,)
    regressor_hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.1
    fusion_mode: str = "concat"
    token_dim: int = 128
    attention_heads: int = 4
    attention_layers: int = 1
    attention_dropout: float = 0.1
    patch_size: tuple[int, int] = (8, 8)
    patch_layers: int = 2
    patch_heads: int = 4
    patch_dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict | None) -> "EyeCropModelConfig":
        if not payload:
            return cls()
        defaults = cls()
        return cls(
            encoder_channels=tuple(payload.get("encoder_channels", defaults.encoder_channels)),
            encoder_type=str(payload.get("encoder_type", defaults.encoder_type)),
            encoder_pooling=str(payload.get("encoder_pooling", defaults.encoder_pooling)),
            eye_coord_channels=bool(payload.get("eye_coord_channels", defaults.eye_coord_channels)),
            head_hidden_dims=tuple(payload.get("head_hidden_dims", defaults.head_hidden_dims)),
            extra_feature_keys=tuple(payload.get("extra_feature_keys", defaults.extra_feature_keys)),
            extra_hidden_dims=tuple(payload.get("extra_hidden_dims", defaults.extra_hidden_dims)),
            regressor_hidden_dims=tuple(
                payload.get("regressor_hidden_dims", defaults.regressor_hidden_dims)
            ),
            dropout=float(payload.get("dropout", defaults.dropout)),
            fusion_mode=str(payload.get("fusion_mode", defaults.fusion_mode)),
            token_dim=int(payload.get("token_dim", defaults.token_dim)),
            attention_heads=int(payload.get("attention_heads", defaults.attention_heads)),
            attention_layers=int(payload.get("attention_layers", defaults.attention_layers)),
            attention_dropout=float(payload.get("attention_dropout", defaults.attention_dropout)),
            patch_size=tuple(payload.get("patch_size", defaults.patch_size)),
            patch_layers=int(payload.get("patch_layers", defaults.patch_layers)),
            patch_heads=int(payload.get("patch_heads", defaults.patch_heads)),
            patch_dropout=float(payload.get("patch_dropout", defaults.patch_dropout)),
        )


def spatial_geometry_frame_vision_config() -> EyeCropModelConfig:
    return EyeCropModelConfig(
        encoder_channels=(24, 48, 96, 96),
        encoder_pooling="flatten",
        eye_coord_channels=True,
        head_hidden_dims=(48, 48),
        extra_feature_keys=tuple(VISION_EYE_GEOMETRY_FEATURE_KEYS),
        extra_hidden_dims=(48, 48),
        regressor_hidden_dims=(256, 128),
        dropout=0.15,
    )


class EyeEncoder(nn.Module):
    def __init__(
        self,
        channels: tuple[int, ...],
        input_channels: int = 1,
        pooling: str = "avg",
    ) -> None:
        super().__init__()
        if pooling not in {"avg", "flatten"}:
            raise ValueError(f"Unsupported CNN encoder pooling: {pooling}")
        self.pooling = pooling
        layers: list[nn.Module] = []
        in_channels = input_channels
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.GELU(),
                ]
            )
            in_channels = out_channels
        self.layers = nn.Sequential(*layers)
        if pooling == "avg":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.output_dim = channels[-1]
        else:
            self.pool = nn.Identity()
            with torch.no_grad():
                dummy = torch.zeros(1, input_channels, EYE_CROP_HEIGHT, EYE_CROP_WIDTH)
                encoded = self.pool(self.layers(dummy))
            self.output_dim = int(encoded.flatten(start_dim=1).shape[1])

    def forward(self, eye_image: torch.Tensor) -> torch.Tensor:
        encoded = self.pool(self.layers(eye_image))
        return encoded.flatten(start_dim=1)


class PatchEyeEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        token_dim: int,
        patch_size: tuple[int, int],
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if token_dim % heads != 0:
            raise ValueError("token_dim must be divisible by patch transformer heads")
        if EYE_CROP_HEIGHT % patch_size[0] != 0 or EYE_CROP_WIDTH % patch_size[1] != 0:
            raise ValueError("Eye crop size must be divisible by patch_size")

        self.patch_embed = nn.Conv2d(
            input_channels,
            token_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        token_count = (EYE_CROP_HEIGHT // patch_size[0]) * (EYE_CROP_WIDTH // patch_size[1])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, token_count + 1, token_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(token_dim)
        self.output_dim = token_dim

    def forward(self, eye_image: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(eye_image).flatten(start_dim=2).transpose(1, 2)
        cls_token = self.cls_token.expand(eye_image.shape[0], -1, -1)
        tokens = torch.cat([cls_token, patches], dim=1)
        tokens = tokens + self.pos_embedding
        encoded = self.encoder(tokens)
        return self.norm(encoded[:, 0, :])


class EyeCropRegressor(nn.Module):
    def __init__(
        self,
        head_feature_dim: int = len(VISION_HEAD_FEATURE_KEYS),
        extra_feature_dim: int = 0,
        config: EyeCropModelConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or EyeCropModelConfig()
        self.extra_feature_dim = extra_feature_dim
        self.eye_input_channels = 3 if self.config.eye_coord_channels else 1
        if self.config.encoder_type == "cnn":
            self.eye_encoder = EyeEncoder(
                self.config.encoder_channels,
                input_channels=self.eye_input_channels,
                pooling=self.config.encoder_pooling,
            )
        elif self.config.encoder_type == "patch_transformer":
            self.eye_encoder = PatchEyeEncoder(
                input_channels=self.eye_input_channels,
                token_dim=self.config.token_dim,
                patch_size=self.config.patch_size,
                layers=self.config.patch_layers,
                heads=self.config.patch_heads,
                dropout=self.config.patch_dropout,
            )
        else:
            raise ValueError(f"Unsupported eye encoder type: {self.config.encoder_type}")

        head_layers: list[nn.Module] = []
        in_dim = head_feature_dim
        for hidden_dim in self.config.head_hidden_dims:
            head_layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU()])
            in_dim = hidden_dim
        self.head_mlp = nn.Sequential(*head_layers)
        head_output_dim = in_dim

        extra_layers: list[nn.Module] = []
        in_dim = extra_feature_dim
        for hidden_dim in self.config.extra_hidden_dims:
            if in_dim <= 0:
                break
            extra_layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU()])
            in_dim = hidden_dim
        self.extra_mlp = nn.Sequential(*extra_layers)
        extra_output_dim = in_dim if extra_feature_dim > 0 else 0

        self.fusion_mode = self.config.fusion_mode
        if self.fusion_mode not in {"concat", "attention"}:
            raise ValueError(f"Unsupported fusion mode: {self.fusion_mode}")

        if self.fusion_mode == "attention":
            if self.config.token_dim % self.config.attention_heads != 0:
                raise ValueError("token_dim must be divisible by attention_heads")
            self.left_token_projection = nn.Linear(self.eye_encoder.output_dim, self.config.token_dim)
            self.right_token_projection = nn.Linear(self.eye_encoder.output_dim, self.config.token_dim)
            self.head_token_projection = nn.Linear(
                head_output_dim + extra_output_dim,
                self.config.token_dim,
            )
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.config.token_dim))
            self.modality_embeddings = nn.Parameter(torch.zeros(4, self.config.token_dim))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.config.token_dim,
                nhead=self.config.attention_heads,
                dim_feedforward=self.config.token_dim * 4,
                dropout=self.config.attention_dropout,
                activation="gelu",
                batch_first=True,
            )
            self.fusion_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=self.config.attention_layers,
            )
            self.fusion_norm = nn.LayerNorm(self.config.token_dim)
            regressor_in_dim = self.config.token_dim
        else:
            regressor_in_dim = self.eye_encoder.output_dim * 2 + head_output_dim + extra_output_dim

        regressor_layers: list[nn.Module] = []
        for index, hidden_dim in enumerate(self.config.regressor_hidden_dims):
            regressor_layers.extend([nn.Linear(regressor_in_dim, hidden_dim), nn.GELU()])
            if self.config.dropout > 0.0 and index == 0:
                regressor_layers.append(nn.Dropout(p=self.config.dropout))
            regressor_in_dim = hidden_dim
        regressor_layers.append(nn.Linear(regressor_in_dim, 2))
        self.regressor = nn.Sequential(*regressor_layers)

    def forward(
        self,
        left_eye: torch.Tensor,
        right_eye: torch.Tensor,
        head_features: torch.Tensor,
        extra_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        left_eye = self._prepare_eye(left_eye)
        right_eye = self._prepare_eye(right_eye)
        left_embedding = self.eye_encoder(left_eye)
        right_embedding = self.eye_encoder(right_eye)
        head_embedding = self.head_mlp(head_features)
        if self.extra_feature_dim > 0 and extra_features is None:
            extra_features = head_features.new_zeros((head_features.shape[0], self.extra_feature_dim))
        if extra_features is not None and extra_features.shape[-1] > 0:
            extra_embedding = self.extra_mlp(extra_features)
            head_embedding = torch.cat([head_embedding, extra_embedding], dim=1)
        if self.fusion_mode == "attention":
            batch_size = left_embedding.shape[0]
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            tokens = torch.stack(
                [
                    self.left_token_projection(left_embedding),
                    self.right_token_projection(right_embedding),
                    self.head_token_projection(head_embedding),
                ],
                dim=1,
            )
            cls_token = cls_token + self.modality_embeddings[0].view(1, 1, -1)
            tokens = tokens + self.modality_embeddings[1:].unsqueeze(0)
            fused_tokens = self.fusion_encoder(torch.cat([cls_token, tokens], dim=1))
            fused = self.fusion_norm(fused_tokens[:, 0, :])
        else:
            fused = torch.cat([left_embedding, right_embedding, head_embedding], dim=1)
        logits = self.regressor(fused)
        return torch.sigmoid(logits)

    def _prepare_eye(self, eye: torch.Tensor) -> torch.Tensor:
        if not self.config.eye_coord_channels:
            return eye
        batch_size, _, height, width = eye.shape
        x_coords = torch.linspace(-1.0, 1.0, width, device=eye.device, dtype=eye.dtype)
        y_coords = torch.linspace(-1.0, 1.0, height, device=eye.device, dtype=eye.dtype)
        x_grid = x_coords.view(1, 1, 1, width).expand(batch_size, 1, height, width)
        y_grid = y_coords.view(1, 1, height, 1).expand(batch_size, 1, height, width)
        return torch.cat([eye, x_grid, y_grid], dim=1)

