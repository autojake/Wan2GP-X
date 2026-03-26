import torch  # noqa: F401 - ensure torch is resident before any CUDA extensions load

# nvfp4.py hardcodes lightx2v as backend (line 83), overriding env var and auto-detection.
# comfy-kitchen ships NVFP4 kernels for Ampere and up: non-Blackwell GPUs get a cheap
# Q4_0 with smaller, faster blocks than GGUF. lightx2v requires explicit SM target
# compilation and has no SM121 image.
from shared.qtypes import nvfp4 as _nvfp4
_nvfp4.set_nvfp4_backend("comfy")

if __name__ == "__main__":
    import runpy
    runpy.run_path("wgp.py", init_globals={"__name__": "__main__"}, run_name="__main__")
