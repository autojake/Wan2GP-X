import os
import importlib.util
import torch

# Apply universal patches first
_spec = importlib.util.spec_from_file_location(
    "wgp_x", os.path.join(os.path.dirname(__file__) or ".", "wgp-x.py")
)
assert _spec is not None and _spec.loader is not None, "wgp-x.py not found"
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# GB10 (DGX Spark / Grace Blackwell 10) patches for unified HBM3e memory.
# No separate VRAM/sysram pool -- the assumptions mmgp and VAE tile sizing make
# about discrete GPUs do not apply.

import mmgp.offload as _mmgp_offload
_mmgp_offload._get_perc_reserved_mem_max = lambda perc_reserved_mem_max=0: 0

# VAE tile sizing: >= 24000 MB -> no tiling (bandwidth disaster on unified memory).
# Cap at 20GB to land in the 256-tile bucket. Also keeps mmgp's VRAM budget
# calculations from treating the full 128GB pool as available headroom.
_DGX_VRAM_CAP = 20 * 1024 ** 3
_real_gdp = torch.cuda.get_device_properties

class _MemCapProxy:
    def __init__(self, props):
        self._props = props
        self.total_memory = min(props.total_memory, _DGX_VRAM_CAP)
    def __getattr__(self, name):
        return getattr(self._props, name)

torch.cuda.get_device_properties = lambda dev: _MemCapProxy(_real_gdp(dev))

# sage2_core.py dispatches on arch string but has no sm121 branch (added sm120, missed sm121).
# Alias sm121 -> sm120 so the sm120 kernel path is used. The compiled .so has sm_120a cubins
# plus PTX (built with 12.0+PTX) so the driver JIT covers sm121.
try:
    import shared.sage2_core as _sage2
    _sage2._CUDA_ARCHS = tuple("sm120" if a == "sm121" else a for a in _sage2._CUDA_ARCHS)
    _real_get_arch = _sage2._get_cuda_arch
    _sage2._get_cuda_arch = lambda dev: "sm120" if _real_get_arch(dev) == "sm121" else _real_get_arch(dev)
except ImportError:
    pass

if __name__ == "__main__":
    import runpy
    runpy.run_path("wgp.py", init_globals={"__name__": "__main__"}, run_name="__main__")
