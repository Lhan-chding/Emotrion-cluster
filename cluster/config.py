from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


MUSIC_ALLOWED_VIEWS = ("audio", "lyrics")
MUSIC_ALLOWED_SPLITS = ("70_15_15", "40_30_30")
MUSIC_ALLOWED_SCALE_MODES = ("none", "standard_train_only")
MUSIC_MODEL_VARIANTS = ("legacy", "residual_gate", "shallow_axis_gate", "gaussian_region")
MUSIC_LABEL_NAMES: Dict[int, str] = {
    0: "Q1",
    1: "Q2",
    2: "Q3",
    3: "Q4",
}
MUSIC_LABEL_TO_ID = {name: idx for idx, name in MUSIC_LABEL_NAMES.items()}


@dataclass
class MusicTrainConfig:
    data_path: str
    batch_size: int
    mse_epochs: int
    con_epochs: int
    learning_rate: float
    weight_decay: float
    temperature_l: float
    normalized: bool
    lmd: float
    beta: float
    dim_high_feature: int
    dim_low_feature: int
    dims: List[int]
    views: List[str]
    split_protocol: str
    model_variant: str
    consistency_weight: float
    fused_dec_weight: float
    axis_loss_weight: float
    gate_floor_weight: float
    gate_floor: float
    region_view_ce_weight: float
    region_separation_weight: float
    region_anchor_weight: float
    region_var_reg_weight: float
    loader_scale_mode: str
    use_best_checkpoint: bool
    save_model: bool
    load_model: bool
    model_path: str
    results_path: str


@dataclass
class DiscoveryTrainConfig:
    """Training configuration for the MusicMetadataDiscoveryNet pipeline."""
    # Architecture
    dropout: float = 0.1
    metadata_logit_offset: float = -0.5
    # Optimizer
    weight_decay: float = 1e-4
    # LR scheduler (CosineAnnealingWarmRestarts); set scheduler_T0=0 to disable
    scheduler_T0: int = 20
    scheduler_Tmult: int = 2
    scheduler_eta_min: float = 1e-6
    # Early stopping; set to 0 to disable
    early_stopping_patience: int = 15
    # Gradient clipping; set to 0.0 to disable
    grad_clip_norm: float = 1.0
    # Mixed precision (BF16 on A100)
    use_amp: bool = True
    # Gate entropy regularization weight
    gate_entropy_weight: float = 0.01


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_music_views(views_csv: str) -> List[str]:
    raw_views = [part.strip().lower() for part in str(views_csv).split(",") if part.strip()]
    if not raw_views:
        return list(MUSIC_ALLOWED_VIEWS)

    views = list(dict.fromkeys(raw_views))
    invalid = [view for view in views if view not in MUSIC_ALLOWED_VIEWS]
    if invalid:
        raise ValueError(
            f"Invalid music views: {invalid}. Allowed views are {list(MUSIC_ALLOWED_VIEWS)}."
        )
    if set(views) != set(MUSIC_ALLOWED_VIEWS) or len(views) != len(MUSIC_ALLOWED_VIEWS):
        raise ValueError(
            "Music pipeline only supports the two-view setup 'audio,lyrics'."
        )
    return views


def parse_music_dims(dims_csv: str) -> List[int]:
    dims = [int(part) for part in str(dims_csv).split(",") if str(part).strip()]
    if not dims:
        raise ValueError("music_dims must contain at least one hidden dimension.")
    return dims


def parse_split_protocol(value: str) -> str:
    protocol = str(value).strip()
    if protocol not in MUSIC_ALLOWED_SPLITS:
        raise ValueError(
            f"Unsupported music split protocol '{protocol}'. Allowed: {list(MUSIC_ALLOWED_SPLITS)}."
        )
    return protocol


def parse_loader_scale_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in MUSIC_ALLOWED_SCALE_MODES:
        raise ValueError(
            f"Unsupported music_loader_scale_mode '{mode}'. "
            f"Allowed: {list(MUSIC_ALLOWED_SCALE_MODES)}."
        )
    return mode


def parse_model_variant(value: str) -> str:
    variant = str(value).strip().lower()
    if variant not in MUSIC_MODEL_VARIANTS:
        raise ValueError(
            f"Unsupported music_model_variant '{variant}'. Allowed: {list(MUSIC_MODEL_VARIANTS)}."
        )
    return variant


def build_music_config(args) -> MusicTrainConfig:
    return MusicTrainConfig(
        data_path=str(args.data_path),
        batch_size=int(args.batch_size),
        mse_epochs=int(args.mse_epochs),
        con_epochs=int(args.con_epochs),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        temperature_l=float(args.temperature_l),
        normalized=_as_bool(args.normalized),
        lmd=float(args.music_lmd),
        beta=float(args.music_beta),
        dim_high_feature=int(args.music_dim_high_feature),
        dim_low_feature=int(args.music_dim_low_feature),
        dims=parse_music_dims(args.music_dims),
        views=parse_music_views(args.music_views),
        split_protocol=parse_split_protocol(args.music_split_protocol),
        model_variant=parse_model_variant(args.music_model_variant),
        consistency_weight=float(args.music_consistency_weight),
        fused_dec_weight=float(args.music_fused_dec_weight),
        axis_loss_weight=float(args.music_axis_loss_weight),
        gate_floor_weight=float(args.music_gate_floor_weight),
        gate_floor=float(args.music_gate_floor),
        region_view_ce_weight=float(args.music_region_view_ce_weight),
        region_separation_weight=float(args.music_region_separation_weight),
        region_anchor_weight=float(args.music_region_anchor_weight),
        region_var_reg_weight=float(args.music_region_var_reg_weight),
        loader_scale_mode=parse_loader_scale_mode(args.music_loader_scale_mode),
        use_best_checkpoint=_as_bool(args.music_use_best_checkpoint),
        save_model=_as_bool(args.save_model),
        load_model=_as_bool(args.load_model),
        model_path=str(args.music_model_path),
        results_path=str(args.music_results_path),
    )
