"""ATEC local navigation experiments. Map truth stays inside the simulator."""
import os
from pathlib import Path

_cache=Path(__file__).resolve().parents[1]/'cache'
for _name,_folder in [('CUDA_CACHE_PATH','cuda'),('TORCH_HOME','torch'),('TRITON_CACHE_DIR','triton')]:
    os.environ.setdefault(_name,str(_cache/_folder))
