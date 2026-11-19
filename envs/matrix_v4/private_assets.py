"""Prepare and receive the private ATEC runtime inputs. No network or simulator calls."""
import argparse
import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

import numpy as np


TERRAIN = Path("data/trail_a_mujoco")
VERSIONS = ("full_v1", "full_v2")
MODEL = Path("UeSim/Content/model")
WEIGHTS = Path("checkpoints/nav4_dreamer_v8_s0")
BUNDLE = "matrix-v4-private.tar.gz"


def sha256(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
        return digest.hexdigest()


def pack(source, output, version):
    full = TERRAIN / version
    overlay = full / "matrix" / MODEL
    output.mkdir(parents=True, exist_ok=True)
    # Same centreline and scan-frame metres as nav4/matrix/full_scan.py:19-23.
    with np.load(source / "data/trail_a_v8_frozen/forward/map.npz") as bank:
        s = np.arange(14441) * 0.05
        xy = np.column_stack([np.interp(s, bank["s"], col) for col in bank["route"][:, :2].T])
    route = io.BytesIO()
    np.savez_compressed(route, station_m=s, xy_m=xy)
    zgws = source / overlay / "zgws"
    paths = [WEIGHTS / "latest.pkl", WEIGHTS / "config.json"]
    paths += [full / "assembly.json", full / "USAGE.md"]
    if version == "full_v1":
        paths.extend(full / name for name in ("frame_alignment.json", "route_vs_v8.csv"))
    else:
        paths.append(full / "work/spawns.json")
    paths += [overlay / "config" / f"config_atec_{scene}_candidate.json" for scene in ("full", "window_00")]
    for pattern in ("zgws_atec_*.xml", "atec_scene_*.xml", "atec_full"):
        hits = sorted(zgws.glob(pattern))
        if not hits:
            raise FileNotFoundError(zgws / pattern)
        paths.extend(item.relative_to(source) for item in hits)
    with tempfile.TemporaryDirectory(dir=output) as scratch:
        archive = Path(scratch) / BUNDLE
        with tarfile.open(archive, "w:gz", compresslevel=2, dereference=True) as tar:
            for rel in paths:
                tar.add(source / rel, arcname=str(rel))
            info = tarfile.TarInfo(str(full / "route.npz"))
            info.size = route.tell()
            route.seek(0)
            tar.addfile(info, route)
        digest = sha256(archive)
        checksum = archive.with_suffix(".gz.sha256")
        checksum.write_text(f"{digest}  {BUNDLE}\n")
        archive.replace(output / BUNDLE)
        checksum.replace(output / checksum.name)
    print(f"{digest}  {output / BUNDLE}")


def scene_directory(inputs):
    matches = [inputs / TERRAIN / v for v in VERSIONS if (inputs / TERRAIN / v).is_dir()]
    if len(matches) != 1:
        raise ValueError("Expected one terrain version in private bundle")
    return matches[0]


def receive(inbox, private):
    archive = inbox / BUNDLE
    expected, filename = (inbox / (BUNDLE + ".sha256")).read_text().split()
    if filename != BUNDLE or sha256(archive) != expected:
        raise ValueError(f"Checksum mismatch: {archive}")
    private.mkdir(parents=True, exist_ok=True)
    target = private / expected
    if not target.exists():
        with tempfile.TemporaryDirectory(dir=private) as tmp:
            staged = Path(tmp) / "inputs"
            staged.mkdir()
            with tarfile.open(archive, "r:gz") as tar:
                for member in tar:
                    rel = Path(member.name)
                    if (rel.is_absolute() or ".." in rel.parts or
                            not (any(rel.is_relative_to(TERRAIN / v) for v in VERSIONS) or rel.is_relative_to(WEIGHTS))):
                        raise ValueError(f"Unexpected input path: {rel}")
                    dst = staged / rel
                    if member.isdir():
                        dst.mkdir(parents=True, exist_ok=True)
                    elif member.isfile():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        with tar.extractfile(member) as src, dst.open("xb") as out:
                            shutil.copyfileobj(src, out)
                    else:
                        raise ValueError(f"Unsupported archive entry: {rel}")
            full = scene_directory(staged)
            required = [WEIGHTS / "latest.pkl", WEIGHTS / "config.json", full / "assembly.json",
                        full / "route.npz"]
            if full.name == "full_v1":
                required.extend([full / "route_vs_v8.csv", full / "frame_alignment.json"])
            else:
                required.append(full / "work/spawns.json")
            for rel in required:
                if not (staged / rel).stat().st_size:
                    raise ValueError(f"Empty asset: {rel}")
            staged.rename(target)
    return target


def configure(inputs, release, scene):
    assets = scene_directory(inputs) / "matrix" / MODEL
    model = release / MODEL
    config = model / "config"
    candidate = json.loads((assets / "config" / f"config_atec_{scene}_candidate.json").read_text())["robot"]
    active = json.loads((config / "config.json").read_text())
    sensors = config / "sensors"
    dual = json.loads((sensors / "config_dual_airy_gpu.json").read_text())
    default = json.loads((sensors / "config_default.json").read_text())
    presets = {**dual, "IMU": default["IMU"], "Odom": default["Odom"]}
    if sorted(s["sensor_type"] for s in presets.values()) != ["airy", "airy", "imu", "odom"]:
        raise ValueError("Unexpected vendor sensor preset")
    # Terrain and robot share one UE mounting actor in the combined model.
    for sensor in presets.values():
        if sensor["sensor_type"] == "airy":
            sensor["ignore_mounting_actor"] = False
    candidate.update(inside_mc=True, mujoco_running=False, sensor_sync_mode=True, sensors=presets)
    active["robot"].update(candidate)
    zgws = model / "zgws"
    if not (zgws / "assets" / "BASE_LINK.STL").is_file():
        raise FileNotFoundError("Vendor M1 meshes missing")
    # Keep model entry XMLs next to the vendor assets. Link the immutable private terrain.
    with tempfile.TemporaryDirectory(dir=zgws) as tmp:
        for path in (assets / "zgws").glob("*.xml"):
            staged = Path(tmp) / path.name
            shutil.copyfile(path, staged)
            staged.replace(zgws / path.name)
        terrain = assets / "zgws" / "atec_full"
        if not terrain.is_dir():
            raise FileNotFoundError(terrain)
        link = Path(tmp) / "atec_full"
        link.symlink_to(terrain)
        link.replace(zgws / link.name)
    # Keep the combined model outside UeSim/Content. Under Content, the UE
    # importer requires cooked StaticMeshes and skips our private OBJ meshes.
    external = release.parent / "models" / "zgws"
    external.mkdir(parents=True, exist_ok=True)
    selected = release / "UeSim" / candidate["mujoco_model"]
    with tempfile.TemporaryDirectory(dir=external) as tmp:
        stage = Path(tmp)
        for source in [zgws / "assets", zgws / "atec_full", *(assets / "zgws").glob("*.xml")]:
            link = stage / source.name
            link.symlink_to(source.resolve())
            link.replace(external / source.name)
        shutil.copyfile(selected, stage / "zgws.xml")
        (stage / "zgws.xml").replace(external / "zgws.xml")
    active["robot"]["mujoco_model"] = str(external / "zgws.xml")
    with tempfile.NamedTemporaryFile("w", dir=config, delete=False) as out:
        json.dump(active, out, indent=2)
        out.write("\n")
    Path(out.name).replace(config / "config.json")


def main():
    parser = argparse.ArgumentParser(__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("pack")
    create.add_argument("source", type=Path, help="main checkout with the private data and checkpoint")
    create.add_argument("output", type=Path)
    create.add_argument("--version", choices=VERSIONS, default="full_v1")
    install = commands.add_parser("install")
    install.add_argument("inbox", type=Path)
    install.add_argument("runtime", type=Path)
    install.add_argument("--scene", choices=["window_00", "full"], default="full")
    args = parser.parse_args()
    os.umask(0o077)
    if args.command == "pack":
        pack(args.source.resolve(), args.output.resolve(), args.version)
    else:
        root = args.runtime.resolve()
        inputs = receive(args.inbox, root / "private")
        configure(inputs, root / "MATRiX_v1.1.2", args.scene)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            link = Path(tmp) / "inputs"
            link.symlink_to(inputs)
            link.replace(root / "inputs")
        print(f"Prepared {args.scene}; route and checkpoint under {root / 'inputs'}")


if __name__ == "__main__":
    main()
