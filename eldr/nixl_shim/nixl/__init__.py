# ROCm: vLLM imports `nixl`, the ROCm drop-in is packaged as `rixl`.
# Alias the whole package + submodules so `import nixl[._api/._bindings/...]`
# resolves to rixl. Additive, no image change, pin preserved.
import sys as _sys
import rixl as _rixl
_sys.modules[__name__] = _rixl
for _s in ("_api", "_bindings", "_utils", "logging"):
    try:
        _m = __import__("rixl." + _s, fromlist=[_s])
        _sys.modules[__name__ + "." + _s] = _m
    except Exception:
        pass
