import torch  # noqa: F401 - ensure torch is resident before any CUDA extensions load

# nvfp4.py hardcodes lightx2v as backend (line 83), overriding env var and auto-detection.
# comfy-kitchen ships NVFP4 kernels for Ampere and up: non-Blackwell GPUs get a cheap
# Q4_0 with smaller, faster blocks than GGUF. lightx2v requires explicit SM target
# compilation and has no SM121 image.
from shared.qtypes import nvfp4 as _nvfp4
_nvfp4.set_nvfp4_backend("comfy")

# If triton is importable, enable the int8 kernel unconditionally and prevent
# server_config["enable_int8_kernels"]=0 from disabling it via disable_quanto_int8_kernel.
# WAN2GP_QUANTO_INT8_TRITON=0 still opts out (maybe_enable_quanto_int8_kernel checks it).
try:
    import triton as _triton_check  # noqa: F401
    from shared.kernels import quanto_int8_inject as _int8
    _int8.maybe_enable_quanto_int8_kernel()
    _int8.disable_quanto_int8_kernel = lambda *a, **kw: None
except ImportError:
    pass

# models/_settings.json has never carried settings_version, so clean_settings() sees
# version 0 for any request that didn't go through the normal save/load roundtrip
# (API calls, queued jobs). fix_settings() then runs all migration blocks including
# the < 2.35 one that appends "V" to audio_prompt_type, forcing vocal separation.
# Patch builtins.open to inject settings_version into that file on read.
import builtins as _builtins
import json as _json
import os as _os
import tempfile as _tempfile

_real_open = _builtins.open
_settings_abspath = _os.path.abspath("models/_settings.json")

def _open_patched(file, mode='r', *args, **kwargs):
    try:
        if ('r' in str(mode) and 'b' not in str(mode) and
                _os.path.abspath(str(file)) == _settings_abspath):
            with _real_open(file, mode, *args, **kwargs) as _f:
                _data = _json.load(_f)
            _data.setdefault("settings_version", 2.55)
            _tmp = _tempfile.NamedTemporaryFile(
                mode='w', suffix='.json', delete=False,
                encoding=kwargs.get('encoding', 'utf-8')
            )
            _json.dump(_data, _tmp, indent=4)
            _tmp_path = _tmp.name
            _tmp.close()
            _handle = _real_open(_tmp_path, mode, *args, **kwargs)
            _os.unlink(_tmp_path)
            return _handle
    except Exception:
        pass
    return _real_open(file, mode, *args, **kwargs)

_builtins.open = _open_patched

if __name__ == "__main__":
    import runpy
    runpy.run_path("wgp.py", init_globals={"__name__": "__main__"}, run_name="__main__")
