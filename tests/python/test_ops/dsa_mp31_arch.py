# Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
# rights reserved.

"""Architecture gate for the DSA indexer ``mp_31``  kernel tests.

``indexer_forward_mp31`` is built for the ``mp_31`` ISA (its TileLang target is
``mp_31``, and its pass recipe is the MP31 one), and
``indexer_backward_mp31``, although its kernels are arch-generic, is validated
here in its MP31 configuration (``block_I=64`` under the ~192 KB opt-in
shared-memory ceiling). Their presence in the test tree must therefore never
make a non-``mp_31`` run (``mp_32`` CI in particular) attempt a build or
exercise an untuned configuration -- that fails in the compiler or in an
assertion, which is a confusing signal that says nothing about the change under
test.

Skipping has to be decided **before** the factory is called, which is why this is
a module-level marker rather than a ``pytest.raises`` around the build: by the
time the factory raises, TileLang has already been asked to compile.

The capability query lives in the library (``api._device_capability``) so there
is one definition of "which arch is this"; it returns ``None`` rather than
raising when the driver or the device cannot be queried, which keeps collection
safe on a host with no MUSA runtime.
"""

from typing import Optional

try:
    import torch
except ImportError:  # pragma: no cover - torch is a hard test dependency
    torch = None


def mp31_skip_reason(device: str = "musa:0") -> Optional[str]:
    """Return a skip reason when *device* is not ``mp_31``, else ``None``.

    ``None`` means "go ahead": the kernels are buildable here.
    """
    if torch is None:  # pragma: no cover - guarded by importorskip upstream
        return "torch is not available"

    from mudnn.deepseek_sparse_attention.indexer_backward.api import (
        _MP31_CAPABILITY,
        _device_capability,
    )

    capability = _device_capability(torch.device(device))
    if capability == _MP31_CAPABILITY:
        return None
    if capability is None:
        return (
            "the DSA indexer mp_31 kernels are MP31-specific and "
            f"the capability of {device} could not be queried"
        )
    return (
        "the DSA indexer mp_31 kernels are built for mp_31 "
        f"(capability {_MP31_CAPABILITY}); this device reports {capability}. "
        "Nothing is wrong with the change under test."
    )
