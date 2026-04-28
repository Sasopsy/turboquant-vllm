"""Plugin registration entrypoint for TileLang TurboQuant C3 integration."""

from tilelang_turboquant.quantization.compat import (
    install_cache_dtype_admission_shims,
    install_kv_spec_dispatch_shim,
)

_REGISTERED = False


def register_all() -> None:
    """Install C3 registration hooks in the current process.

    This is the plugin's process-local bootstrap step for vLLM integration.
    Importing the quant config module registers the `tq_*` quantization names,
    then the two shims teach vLLM how to accept plugin cache dtypes and how to
    ask the plugin for a custom KV cache spec.
    """

    global _REGISTERED
    if _REGISTERED:
        return

    # Importing the module triggers the quantization-config decorators and
    # populates vLLM's customized quantization registry.
    from tilelang_turboquant.quantization.quant_config import (  # noqa: F401
        TileLangTQ3BitConfig,
        TileLangTQ4BitConfig,
    )

    # These shims patch the small number of vLLM surfaces that would otherwise
    # reject plugin-owned cache dtype literals or skip our plugin KV spec.
    install_cache_dtype_admission_shims()
    install_kv_spec_dispatch_shim()
    _REGISTERED = True
