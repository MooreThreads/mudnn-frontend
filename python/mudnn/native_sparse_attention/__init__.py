from ._torch_fallback import (
    CompressionAttention,
    SelectionAttention,
    SlidingWindowAttention,
    TopKReduction,
    compression_attention_wrapper,
    selection_attention_wrapper,
    sliding_window_attention_wrapper,
    topk_reduction_wrapper,
)


class NSANamespace:
    SelectionAttention = staticmethod(SelectionAttention)
    selection_attention_wrapper = staticmethod(selection_attention_wrapper)
    SlidingWindowAttention = staticmethod(SlidingWindowAttention)
    sliding_window_attention_wrapper = staticmethod(sliding_window_attention_wrapper)
    CompressionAttention = staticmethod(CompressionAttention)
    compression_attention_wrapper = staticmethod(compression_attention_wrapper)
    TopKReduction = staticmethod(TopKReduction)
    topk_reduction_wrapper = staticmethod(topk_reduction_wrapper)


NSA = NSANamespace()

__all__ = [
    "NSA",
    "CompressionAttention",
    "SelectionAttention",
    "SlidingWindowAttention",
    "TopKReduction",
    "compression_attention_wrapper",
    "selection_attention_wrapper",
    "sliding_window_attention_wrapper",
    "topk_reduction_wrapper",
]
