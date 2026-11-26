# Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
# rights reserved.

import os
from pathlib import Path
import sys


_THIS_DIR = Path(__file__).resolve().parent
_FRONTEND_DIR = _THIS_DIR.parents[2]
_MODULE_DIRS = [
    os.getenv("MUDNN_FRONTEND_CPP_TEST_BUILD_DIR"),
    os.getenv("MUDNN_FRONTEND_FRAMEWORK_TEST_BUILD_DIR"),
    _FRONTEND_DIR / "build" / "tests" / "cpp",
    Path.cwd() / "build" / "tests" / "cpp",
    Path.cwd(),
]

for module_dir in _MODULE_DIRS:
    if module_dir and Path(module_dir).exists():
        sys.path.insert(0, str(module_dir))

import mudnn_frontend_cpp_tests as cpp_tests  # noqa: E402
