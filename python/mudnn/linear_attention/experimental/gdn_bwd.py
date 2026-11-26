# SPDX-License-Identifier: Apache-2.0
"""Compatibility entry for the standalone GDN backward API."""

from ..tilelang.gdn_backward import (
    PreparedBackward,
    _beta_sigmoid_db,
    _l2norm_vjp,
    _safe_gate_dg,
    gdn_bwd,
    prepare_gdn_bwd,
)

__all__ = ["PreparedBackward", "gdn_bwd", "prepare_gdn_bwd"]
