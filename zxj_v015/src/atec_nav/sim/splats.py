"""Read dataset SOG v2 without changing its native coordinate frame.

Encoding reference: https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/sog-v2/
LCC node ranges, not filenames, select one LOD without duplicate parent splats.
"""
import io
import json
import zipfile

import numpy as np
from PIL import Image


def read_sog(path):
    with zipfile.ZipFile(path) as archive:
        meta = json.loads(archive.read("meta.json"))
        if meta["version"] != 2 or "shN" in meta:
            raise ValueError("This decoder supports the dataset's DC-only SOG v2")
        def pixels(name):
            return np.asarray(Image.open(io.BytesIO(archive.read(name))).convert("RGBA")).reshape(-1, 4)[:meta["count"]]
        low, high = [pixels(f)[:, :3].astype(np.uint16) for f in meta["means"]["files"]]
        quantized = (low | (high << 8)).astype(np.float32) / 65535
        lo, hi = [np.asarray(meta["means"][k], np.float32) for k in ("mins", "maxs")]
        means = quantized * (hi - lo) + lo
        means = np.sign(means) * np.expm1(np.abs(means))
        packed = pixels(meta["quats"]["files"][0])
        modes = packed[:, 3].astype(int) - 252
        if not np.isin(modes, [0, 1, 2, 3]).all():
            raise ValueError("Invalid quaternion encoding")
        kept = (packed[:, :3].astype(np.float32) / 255 - .5) * np.sqrt(2)
        omitted = np.sqrt(np.maximum(0, 1 - (kept * kept).sum(axis=1)))
        quats = np.zeros((len(means), 4), np.float32)
        for mode in range(4):
            mask = modes == mode
            quats[np.ix_(mask, [j for j in range(4) if j != mode])] = kept[mask]
            quats[mask, mode] = omitted[mask]
        scales = np.exp(np.asarray(meta["scales"]["codebook"], np.float32)[pixels(meta["scales"]["files"][0])[:, :3]])
        packed = pixels(meta["sh0"]["files"][0])
        colors = np.clip(.5 + .28209479177387814 * np.asarray(meta["sh0"]["codebook"], np.float32)[packed[:, :3]], 0, 1)
        return dict(means=means, quats=quats, scales=scales, colors=colors,
                    opacities=packed[:, 3].astype(np.float32) / 255)


class SplatRenderer:
    """Native 3DGS RGB and expected optical depth; no gradients allocated."""
    def __init__(self, path, device="cuda"):
        import torch
        self.torch = torch
        self.device = device
        with np.load(path) as data:
            self.params = {k: torch.as_tensor(data[k], device=device) for k in
                           ("means", "quats", "scales", "colors", "opacities")}

    def render(self, view, intrinsic, width, height):
        from gsplat import rasterization
        torch = self.torch
        with torch.inference_mode():
            frame, alpha, _ = rasterization(
                **self.params,
                viewmats=torch.as_tensor(view[None], dtype=torch.float32, device=self.device),
                Ks=torch.as_tensor(intrinsic[None], dtype=torch.float32, device=self.device),
                width=width, height=height, near_plane=.05, far_plane=5000,
                packed=True, render_mode="RGB+ED", radius_clip=.3)
            frame = frame[0].cpu().numpy()
            alpha = alpha[0, ..., 0].cpu().numpy()
            frame[..., :3] += (1-alpha[..., None]) * np.array([.75, .83, .9])
            return frame, alpha


def mujoco_camera(scene, width, height):
    """Exact shared pinhole: MuJoCo GL camera to OpenCV +Z-forward coordinates."""
    camera = scene.camera[0]
    position = np.mean([c.pos for c in scene.camera], axis=0)
    forward = np.asarray(camera.forward)
    up = np.asarray(camera.up)
    right = np.cross(forward, up)
    rotation = np.stack([right, -up, forward])
    view = np.eye(4)
    view[:3, :3] = rotation
    view[:3, 3] = -rotation @ position
    fy = height * camera.frustum_near / (camera.frustum_top - camera.frustum_bottom)
    intrinsic = np.array([[fy, 0, width / 2], [0, fy, height / 2], [0, 0, 1]])
    return view.astype(np.float32), intrinsic.astype(np.float32)
