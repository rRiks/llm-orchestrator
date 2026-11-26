"""
env_probe.py
==============
Utility script to probe the Kaggle TPU environment and host system.
It collects:
- TPU device info via `jax` and `torch_xla`
- Host CPU cores, RAM, and disk availability via `psutil`
- Python package versions for key dependencies
- A timestamped snapshot written as JSON to stdout (or a file when redirected)

The script is intended to be run as the first cell in the notebook, and its
output can be captured and stored in ``storage/metadata_*.json`` for later
reference.
"""

import json
import platform
import datetime

# Optional imports – handled gracefully if unavailable in the Kaggle env
try:
    import jax
    import jax.numpy as jnp
except Exception:
    jax = None

try:
    import torch_xla.core.xla_model as xm
except Exception:
    xm = None

try:
    import psutil
except Exception:
    psutil = None

def get_tpu_info():
    """Return a dict with TPU device details.
    Uses JAX if available, otherwise falls back to torch_xla.
    """
    info = {}
    if jax is not None:
        devices = jax.devices()
        info["jax_devices"] = [str(d) for d in devices]
        # Attempt to extract HBM size from device attributes if present
        try:
            # JAX devices expose device_kind and memory_limit (bytes)
            info["tpu_type"] = devices[0].device_kind if devices else "unknown"
            info["tpu_hbm_bytes"] = devices[0].memory_limit if devices else None
        except Exception:
            pass
    elif xm is not None:
        # torch_xla provides a simpler device list
        devices = xm.xla_device()
        info["torch_xla_device"] = str(devices)
    else:
        info["error"] = "No TPU probing libraries available"
    return info


def get_host_info():
    """Gather CPU, RAM, and disk stats using psutil when possible."""
    info = {}
    if psutil is None:
        return {"error": "psutil not available"}
    info["cpu_count"] = psutil.cpu_count(logical=True)
    vm = psutil.virtual_memory()
    info["host_ram_bytes"] = vm.total
    info["host_ram_used_bytes"] = vm.used
    # Disk usage for the working directory (Kaggle default is /kaggle/working)
    # Use the current working directory of the script's location.
    cwd = platform.node()
    try:
        disk = psutil.disk_usage('.')
        info["disk_total_bytes"] = disk.total
        info["disk_used_bytes"] = disk.used
        info["disk_free_bytes"] = disk.free
    except Exception:
        pass
    return info


def get_package_versions():
    """Return a mapping of important package versions."""
    pkgs = ["torch", "torch_xla", "transformers", "datasets", "accelerate", "safetensors", "jax", "jaxlib"]
    versions = {}
    for pkg in pkgs:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[pkg] = None
    return versions


def snapshot():
    snapshot = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "tpu_info": get_tpu_info(),
        "host_info": get_host_info(),
        "package_versions": get_package_versions(),
    }
    return snapshot

if __name__ == "__main__":
    print(json.dumps(snapshot(), indent=2))
