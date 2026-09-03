"""Fail fast if this host cannot actually run CUDA (driver/torch mismatch).

A job that silently falls back to CPU looks healthy and crawls for hours;
this runs a real GPU op before anything expensive starts.
"""

import sys

import torch

if not torch.cuda.is_available():
    print(f"CUDA not available: torch {torch.__version__} (cuda {torch.version.cuda})", file=sys.stderr)
    sys.exit(1)
x = torch.randn(256, 256, device="cuda")
_ = (x @ x).sum().item()
free, total = torch.cuda.mem_get_info()
print(f"cuda ok: {torch.cuda.get_device_name(0)}, torch {torch.__version__}, cuda {torch.version.cuda}, {total / 2**30:.0f} GiB total")
