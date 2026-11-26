# Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
# rights reserved.

from pathlib import Path
import sys


FRONTEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = FRONTEND_ROOT.parent

sys.path.insert(0, str(FRONTEND_ROOT / "python"))
sys.path.insert(0, str(FRONTEND_ROOT / "tests" / "python"))

repo_tests_python = REPO_ROOT / "tests" / "python"
repo_tests_build = REPO_ROOT / "tests" / "build"
if repo_tests_python.exists():
    sys.path.insert(0, str(repo_tests_python))
if repo_tests_build.exists():
    sys.path.insert(0, str(repo_tests_build))
