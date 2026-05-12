"""api_shim - V1 deep_ep.Buffer -> V2 deep_ep.ElasticBuffer compatibility layer.

Usage from a consumer entry point (e.g. vLLM launcher):

    import api_shim
    api_shim.install()            # monkeypatches deep_ep.Buffer -> CompatBuffer
    # ... rest of app continues to import deep_ep.Buffer unchanged

See buffer_v1_compat.CompatBuffer for which V1 signatures are bridged.
"""
from .buffer_v1_compat import CompatBuffer, install

__all__ = ["CompatBuffer", "install"]
