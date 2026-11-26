# Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
# rights reserved.

from . import frontend_legacy


def export_tests(target_globals, *, include, exclude=()):
    include = tuple(include)
    exclude = tuple(exclude)
    exported = []
    for name in dir(frontend_legacy):
        if not name.startswith("test_"):
            continue
        if include and not any(token in name for token in include):
            continue
        if exclude and any(token in name for token in exclude):
            continue
        target_globals[name] = getattr(frontend_legacy, name)
        exported.append(name)
    if not exported:
        raise RuntimeError(f"no frontend legacy tests matched include={include}")
