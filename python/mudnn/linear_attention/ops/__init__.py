"""Backend-independent linear-attention operation contracts (torch custom ops)."""

from .gdn import gated_delta_net
from .kda import (
    kimi_delta_attention,
    kimi_delta_attention_bwd,
    kimi_delta_attention_summary,
    kimi_delta_attention_summary_bwd,
)

__all__ = [
    "gated_delta_net",
    "kimi_delta_attention",
    "kimi_delta_attention_bwd",
    "kimi_delta_attention_summary",
    "kimi_delta_attention_summary_bwd",
]
