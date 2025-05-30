import torch
from functools import wraps

original_load = torch.load

@wraps(original_load)
def patched_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = 'cpu' if not torch.cuda.is_available() else None
    return original_load(*args, **kwargs)

torch.load = patched_load
print("PyTorch load function patched to handle CPU/GPU transitions")
