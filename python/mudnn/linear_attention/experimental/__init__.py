# SPDX-License-Identifier: Apache-2.0
"""Standalone backward compatibility API; production dispatch uses the shared core."""

from .gdn_bwd import gdn_bwd, prepare_gdn_bwd


__all__ = ["gdn_bwd", "prepare_gdn_bwd"]
