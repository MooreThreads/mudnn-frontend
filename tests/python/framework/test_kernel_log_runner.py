# Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
# rights reserved.

import pytest

from .kernel_log_runner import CASE_SPECS
from .kernel_log_runner import assert_case_matches_golden
from .kernel_log_runner import run_case


@pytest.mark.parametrize("case_name", sorted(CASE_SPECS))
def test_framework_graph_kernel_log_matches_golden(case_name):
    # These tests do not depend on torch_musa.  Each case is executed in a fresh
    # subprocess with MUSA_LOG=0xffff, then muModuleGetFunction entries are used
    # to validate the backend kernel path and kernel count.
    result = run_case(case_name)
    assert_case_matches_golden(result)
