from collections import deque
from dataclasses import dataclass

import numpy as np


def pack_depth(depth_m, min_m=0.2, max_m=10.0):
    """Fixed metric quantization plus validity; invalid is NOT free space."""
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2 or not 0 < min_m < max_m:
        raise ValueError("Expected HxW metric depth and 0 < min_m < max_m")
    valid = np.isfinite(depth) & (depth >= min_m) & (depth <= max_m)
    safe = np.where(valid, depth, max_m)
    quantized = np.rint((safe - min_m) * (255.0 / (max_m - min_m)))
    return np.stack(
        [np.clip(quantized, 0, 255).astype(np.uint8),
         valid.astype(np.uint8) * 255],
        axis=-1,
    )


def pack_multiscale_depth(depth_m, min_m=0.2, near_max_m=2.0, max_m=10.0):
    """Near depth, full-range depth, validity, all in fixed metric uint8 scales.

    Valid distant pixels saturate the near channel; validity describes the full
    sensor range. Invalid pixels saturate both depth channels with a zero mask.
    """
    if not 0 < min_m < near_max_m < max_m:
        raise ValueError('Expected min_depth < near_max_depth < max_depth')
    far = pack_depth(depth_m, min_m, max_m)
    safe = np.where(far[..., 1] != 0, depth_m, max_m)
    near = np.rint((safe - min_m) * (255. / (near_max_m - min_m)))
    return np.stack([np.clip(near, 0, 255).astype(np.uint8),
                     far[..., 0], far[..., 1]], axis=-1)


@dataclass(frozen=True)
class NoiseConfig:
    enabled: bool = True
    sigma_base_m: float = 0.01
    sigma_depth_squared: float = 0.001
    dropout_probability: float = 0.03
    outlier_probability: float = 0.01
    patch_dropout_probability: float = 0.1
    patch_size: int = 8
    scale_jitter: float = 0.05
    latency_frames: int = 1

    def __post_init__(self):
        for name in ("dropout_probability", "outlier_probability",
                     "patch_dropout_probability"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.sigma_base_m < 0 or self.sigma_depth_squared < 0:
            raise ValueError("Noise standard deviations must be nonnegative")
        if not 0 <= self.scale_jitter < 1:
            raise ValueError("scale_jitter must be in [0, 1)")
        if self.latency_frames < 0 or self.patch_size < 1:
            raise ValueError("Invalid latency or patch size")


class DepthNoise:
    """Seeded sensor disturbances with episode-local latency state."""

    def __init__(self, config=NoiseConfig(), seed=0, max_m=10.0):
        if max_m <= 0:
            raise ValueError("max_m must be positive")
        self.config = config
        self.max_m = float(max_m)
        self.reset(seed)

    def reset(self, seed=None):
        if seed is not None or not hasattr(self, "rng"):
            self.rng = np.random.default_rng(seed)
        self.frames = deque(maxlen=self.config.latency_frames + 1)
        self.scale = self.rng.uniform(
            1 - self.config.scale_jitter, 1 + self.config.scale_jitter)

    def __call__(self, depth_m):
        depth = np.array(depth_m, dtype=np.float32, copy=True)
        if depth.ndim != 2:
            raise ValueError("Expected HxW depth")
        cfg = self.config
        if not cfg.enabled:
            return depth
        valid = np.isfinite(depth) & (depth > 0) & (depth <= self.max_m)
        safe = np.where(valid, depth, 0.0)
        sigma = cfg.sigma_base_m + cfg.sigma_depth_squared * safe ** 2
        depth = safe * self.scale + self.rng.normal(0, sigma)
        outliers = valid & (
            self.rng.random(depth.shape) < cfg.outlier_probability)
        depth[outliers] = self.rng.uniform(0.2, self.max_m, outliers.sum())
        valid &= self.rng.random(depth.shape) >= cfg.dropout_probability
        if self.rng.random() < cfg.patch_dropout_probability:
            h, w = depth.shape
            ph, pw = min(cfg.patch_size, h), min(cfg.patch_size, w)
            y = self.rng.integers(h - ph + 1)
            x = self.rng.integers(w - pw + 1)
            valid[y:y + ph, x:x + pw] = False
        depth[~valid] = np.nan
        depth = depth.astype(np.float32)
        self.frames.append(depth)
        # Warm-up repeats this episode's first frame, never an old episode.
        return self.frames[0].copy()

