from pathlib import Path
import json
import numpy as np
import yaml


def mounts(path='configs/calibration.yaml'):
    """Column-vector transforms: T_base_front = T_base_imu @ T_imu_front."""
    data = yaml.safe_load(Path(path).read_text())['extrinsic_transforms']
    front = np.array(data['imu_front_to_base']) @ np.array(data['lidar_front_to_imu_front'])
    rear = front @ np.array(data['lidar_back_to_lidar_front'])
    return np.stack([front, rear]).astype(np.float32)


if __name__ == '__main__':
    ts = mounts()
    Path('metrics/calibration.json').write_text(json.dumps({
        'transforms_base_lidar': ts.tolist(),
        'horizontal_axis_native': '+z',
        'axes_base': ts[:, :3, 2].tolist(),
        'base_ground_height_assumed_m': 0.54,
        'base_height_randomization_m': 0.05,
        'rotation_orthogonality_error': [float(np.max(np.abs(t[:3,:3].T @ t[:3,:3] - np.eye(3)))) for t in ts],
    }, indent=2))
