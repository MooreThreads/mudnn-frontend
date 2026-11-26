"""mudnn.linear_attention: the linear-attention operation family and its engines."""

from typing import Any

_LAZY_EXPORTS = {
    "gated_delta_net": ("mudnn.linear_attention.ops", "gated_delta_net"),
    "kimi_delta_attention": ("mudnn.linear_attention.ops", "kimi_delta_attention"),
    "kimi_delta_attention_bwd": (
        "mudnn.linear_attention.ops",
        "kimi_delta_attention_bwd",
    ),
    "kimi_delta_attention_summary": (
        "mudnn.linear_attention.ops",
        "kimi_delta_attention_summary",
    ),
    "kimi_delta_attention_summary_bwd": (
        "mudnn.linear_attention.ops",
        "kimi_delta_attention_summary_bwd",
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = list(_LAZY_EXPORTS)


# --- family factories -------------------------------------------------------
def _collect(ids, *specs):
    """Build the engines named in ``ids``, skipping any whose dependency is missing.

    ``ids`` maps engine name to engine id; each spec is ``(name, module, attr)``.
    Returns the built engines, each with ``engine_id`` set.
    """
    import importlib
    import logging

    out = []
    for name, module, attr in specs:
        if name not in ids:
            continue
        try:
            engine = getattr(importlib.import_module(module), attr)()
        except ImportError as exc:
            logging.getLogger(__name__).info("engine %s is unavailable in this environment: %s", name, exc)
            continue
        engine.engine_id = ids[name]
        out.append(engine)
    return out


def GdnEngines(ids):
    """The engines serving the GDN family, in candidate order."""
    return _collect(
        ids,
        ("gdn_tilelang", "mudnn.linear_attention.tilelang.gdn_engine", "GdnTileLangEngine"),
        ("gdn_prefill_tilelang", "mudnn.linear_attention.tilelang.gdn_prefill_engine",
         "GdnPrefillTileLangEngine"),
        ("gdn_decode_tilelang", "mudnn.linear_attention.tilelang.gdn_decode_engine",
         "GdnDecodeTileLangEngine"),
        ("gdn_tilelang_v2", "mudnn.linear_attention.tilelang.gdn_v2_engine", "GdnV2Engine"),
    )


def KdaEngines(ids):
    """The engines serving the KDA family, in candidate order.

    One engine per MUSA architecture, matching how cuDNN frontend splits its
    KDA backends; each states its own architecture in ``check_support``, so a
    device only ever reaches the kernels built for it.
    """
    return _collect(
        ids,
        ("kda_mutlass_mp31", "mudnn.linear_attention.mutlass.kda_engine",
         "KdaMutlassMp31Engine"),
        ("kda_mutlass_mp32", "mudnn.linear_attention.mutlass.kda_engine",
         "KdaMutlassMp32Engine"),
    )
