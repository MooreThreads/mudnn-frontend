"""End-to-end smoke test for the runtime JIT facility.

The acceptance test for the toolchain itself: a MUSA source is compiled at
runtime by mcc through ninja and loaded with tvm_ffi.  Nothing here needs a
pre-build step.

The suite is split deliberately:

* compile/load tests need only tvm_ffi and the MUSA toolchain, so they run on a
  build host with no GPU and no torch;
* the numeric test additionally needs a MUSA torch, and is skipped without one.

The arithmetic is a fused multiply-add, so a failure points at the
build/load/launch chain rather than at kernel math.
"""

from __future__ import annotations

import pytest

pytest.importorskip("tvm_ffi")

from mudnn._jit import env as jit_env  # noqa: E402
from mudnn._jit import cpp_ext as jit_cpp_ext  # noqa: E402
from mudnn._jit import gen_jit_spec, jit_spec_registry  # noqa: E402

VEC_ADD_SOURCE = jit_env.resolve_kernels_dir() / "smoke" / "vec_add.mu"
VEC_ADD_FUNC = "mudnn_jit_smoke_vec_add"


def _make_vec_add_spec():
    return gen_jit_spec(VEC_ADD_FUNC, [VEC_ADD_SOURCE])


# ---------------------------------------------------------------------------
# Environment / arch resolution (no device, no torch)
# ---------------------------------------------------------------------------
def test_musa_arch_parse_and_naming():
    arch = jit_env.MusaArch.parse("3.1")
    assert (arch.major, arch.minor) == (3, 1)
    assert arch.cc == 31
    assert arch.name == "mp31"
    assert arch.target_flag == "--offload-arch=mp_31"

    mp32 = jit_env.MusaArch.parse("3.2")
    assert mp32.name == "mp32"
    assert mp32.target_flag == "--offload-arch=mp_32"


def test_musa_arch_ordering_puts_mp32_last():
    assert jit_env.MusaArch.parse("3.1") < jit_env.MusaArch.parse("3.2")


def test_musa_arch_rejects_malformed_names():
    for bad in ("31", "3", "3.x", "a.b", ""):
        with pytest.raises(ValueError):
            jit_env.MusaArch.parse(bad)


def test_musa_home_accepts_explicit_sdk_override(tmp_path, monkeypatch):
    """The JIT must not depend on tvm_ffi's optional private SDK finder."""
    sdk = tmp_path / "musa"
    (sdk / "include").mkdir(parents=True)
    monkeypatch.setenv("MUDNN_JIT_MUSA_HOME", str(sdk))
    jit_cpp_ext.get_musa_home.cache_clear()
    try:
        assert jit_cpp_ext.get_musa_home() == sdk.resolve()
    finally:
        jit_cpp_ext.get_musa_home.cache_clear()


def test_kernel_source_root_exists():
    assert jit_env.resolve_kernels_dir().is_dir()
    assert VEC_ADD_SOURCE.is_file(), f"missing smoke kernel: {VEC_ADD_SOURCE}"


def test_repo_root_detection_requires_the_project_layout(tmp_path):
    """A stray ``.git`` in some parent must not be mistaken for the checkout.

    An installed package can sit under a directory that happens to carry a
    ``.git`` or ``CMakeLists.txt`` (a virtualenv created inside another
    repository, a shared ``/tmp``).  Matching on those alone would point every
    source lookup at an unrelated tree, so the ``python/mudnn`` layout is part
    of the marker.
    """
    # A bare version-control / build marker, without the project layout.
    (tmp_path / ".git").mkdir()
    assert not jit_env._looks_like_repo_root(tmp_path), (
        "a bare .git was accepted as the checkout root"
    )

    bare = tmp_path / "bare_build"
    bare.mkdir()
    (bare / "CMakeLists.txt").write_text("")
    assert not jit_env._looks_like_repo_root(bare), (
        "a bare CMakeLists.txt was accepted as the checkout root"
    )

    # The project layout plus either marker is a checkout.
    (tmp_path / "python" / "mudnn").mkdir(parents=True)
    assert jit_env._looks_like_repo_root(tmp_path)


def test_detected_repo_root_is_this_checkout():
    """In a checkout, the detected root really is the one holding ``kernels/``.

    An installed wheel has no checkout, so this asserts nothing there and skips.
    """
    if not jit_env._prefer_repo_tree():
        pytest.skip("not running from a checkout")
    root = jit_env._repo_root
    assert (root / "CMakeLists.txt").is_file() or (root / ".git").exists()
    assert (root / "python" / "mudnn").is_dir()
    assert (root / "kernels").is_dir()


def test_packaged_data_tree_is_used_when_not_in_a_checkout():
    """The installed layout is found through mudnn/data, not through the parent
    directory of the package."""
    if jit_env._prefer_repo_tree():
        pytest.skip("running from a checkout, where the repo tree wins")
    data = jit_env._package_root / "data"
    assert (data / "__init__.py").is_file(), "mudnn.data marker missing"
    assert jit_env.resolve_kernels_dir().is_dir()
    assert jit_env.MUTLASS_INCLUDE_DIR.is_dir()


def test_arch_is_resolved_lazily_so_import_never_needs_a_device():
    """Importing the package must not require a device or an explicit arch.

    The arch-dependent names are properties of the first *use*, not of import:
    otherwise a build host or a CPU-only lint run could not import at all.
    """
    import importlib

    module = importlib.import_module("mudnn._jit.env")
    assert module.MUDNN_JIT_VERSION  # import-time constant, always available
    assert module.MUDNN_JIT_BASE_VERSION


def test_resolved_arch_names_supported_architectures():
    try:
        key = jit_env.resolve_musa_arch_key()
    except RuntimeError as exc:
        pytest.skip(f"no MUSA device and no explicit arch list: {exc}")
    assert key
    for part in key.split("_"):
        assert part.startswith("mp"), part
        jit_env.MusaArch.parse(f"{part[2]}.{part[3:]}")


def test_target_flags_match_the_resolved_arch_list():
    try:
        arch_list = jit_env.resolve_musa_arch_list()
    except RuntimeError as exc:
        pytest.skip(f"no MUSA device and no explicit arch list: {exc}")
    flags = jit_env.resolve_musa_target_flags()
    assert len(flags) == len(arch_list)
    for name, flag in zip(arch_list, flags):
        assert flag == f"--offload-arch=mp_{name[2:]}"


# ---------------------------------------------------------------------------
# The build/load chain (no device, no torch)
# ---------------------------------------------------------------------------
def test_smoke_kernel_builds_and_loads():
    spec = _make_vec_add_spec()
    assert VEC_ADD_FUNC in jit_spec_registry.get_all_specs()

    module = spec.build_and_load()
    assert callable(module[VEC_ADD_FUNC])
    assert spec.jit_library_path.is_file()


def test_build_is_cached_across_calls():
    """A second build_and_load reuses the on-disk module instead of recompiling."""
    spec = _make_vec_add_spec()
    # Keep this test runnable on its own; pytest's collection order is not an
    # API contract and must not be the thing that populates the cache.
    first_module = spec.build_and_load()
    assert callable(first_module[VEC_ADD_FUNC])
    mtime_before = spec.jit_library_path.stat().st_mtime_ns

    module = spec.build_and_load()
    assert callable(module[VEC_ADD_FUNC])
    assert spec.jit_library_path.stat().st_mtime_ns == mtime_before, (
        "the cached library was rewritten; the cache key is not stable"
    )


def test_arch_is_part_of_the_build_directory():
    """Two architectures must not share one build directory.

    A process that resolves two arches would otherwise hand back one arch's
    module for the other -- the failure mode the arch-keyed namespace prevents.
    """
    spec = _make_vec_add_spec()
    assert jit_env.resolve_musa_jit_dir(
        jit_env.MusaArch.parse("3.1")
    ) != jit_env.resolve_musa_jit_dir(jit_env.MusaArch.parse("3.2"))

    # Whichever arch this process resolved must name its own build directory.
    resolved = jit_env.resolve_musa_arch_list()
    for name in resolved:
        assert name in spec.build_dir.as_posix()


def test_spec_status_is_reported():
    spec = _make_vec_add_spec()
    if not spec.is_compiled:
        spec.build_and_load()
    status = jit_spec_registry.get_spec_status(spec.name)
    assert status is not None
    assert status.name == spec.name
    assert status.is_compiled
    assert status.num_sources >= 1


# ---------------------------------------------------------------------------
# Device execution, without a framework
# ---------------------------------------------------------------------------
SELFCHECK_SOURCE = jit_env.resolve_kernels_dir() / "smoke" / "selfcheck.mu"
MUTLASS_PROBE_SOURCE = jit_env.resolve_kernels_dir() / "smoke" / "mutlass_probe.mu"


def _numpy():
    return pytest.importorskip("numpy")


def _expected_cc() -> int:
    """The compute capability the current process compiles for.

    Derived from the resolved architecture list rather than from a device, so
    it is well defined even when the host has no matching hardware.
    """
    names = jit_env.resolve_musa_arch_list()
    return max(int(name[2:]) for name in names)


def _run_selfcheck(device_index: int = 0):
    """Build the torch-free self-check kernel and run it on one device.

    Returns ``(status, device_cc)``.  ``status`` is 1 when the device does not
    exist and 8 when the device is older than the compiled architecture -- both
    are skips, not failures, because neither says anything about the toolchain.
    """
    np = _numpy()
    import tvm_ffi

    spec = gen_jit_spec("mudnn_jit_selfcheck", [SELFCHECK_SOURCE])
    module = spec.build_and_load()
    out = np.zeros(2, dtype=np.int64)
    module["mudnn_jit_selfcheck_fn"](
        tvm_ffi.from_dlpack(out), int(device_index), _expected_cc()
    )
    return int(out[0]), int(out[1])


def test_kernel_executes_on_device_and_verifies_its_own_result():
    """The kernel allocates, launches, copies back and compares, so a pass
    means real execution on a real device -- not merely that it linked.

    Needs no torch: the only value crossing the FFI boundary is an integer.
    """
    status, device_cc = _run_selfcheck(0)
    if status == 1:
        pytest.skip("no MUSA device available")
    if status == 8:
        pytest.skip(f"device mp{device_cc} is older than the compiled architecture")
    assert status == 0, f"device self-check failed with status {status}"
    assert device_cc in (22, 31, 32, 42), f"unexpected device capability {device_cc}"


def test_device_reported_arch_matches_the_compiled_arch():
    """The arch compiled into the cubin and the device's own arch must agree
    for anything to launch; this is the pairing the KDA builder is careful
    about, so assert it here rather than discovering it in a kernel."""
    status, device_cc = _run_selfcheck(0)
    if status == 1:
        pytest.skip("no MUSA device available")
    if status == 8:
        pytest.skip(f"device mp{device_cc} is older than the compiled architecture")
    assert status == 0
    resolved = jit_env.resolve_musa_arch_list()
    # "mp31" -> 31, matching the major * 10 + minor the device reports.
    expected = {int(name[2:]) for name in resolved}
    assert device_cc in expected, (
        f"compiled for {resolved} but the device is mp{device_cc}"
    )


def test_mutlass_headers_are_present():
    include = jit_env.MUTLASS_INCLUDE_DIR
    assert include.is_dir(), f"mutlass include dir missing: {include}"
    for header in (
        "mutlass/mutlass.h",
        "mutlass/bfloat16.h",
        "mutlass/numeric_conversion.h",
        "mutlass/arch/arch.h",
        "mute/tensor.hpp",
    ):
        assert (include / header).is_file(), f"missing mutlass header: {header}"


def test_mutlass_kernel_compiles_and_executes():
    """A kernel that includes mutlass + mute and uses mutlass types in device
    code, so this is the closest thing to the KDA kernel the toolchain can
    prove without the KDA sources."""
    np = _numpy()
    import tvm_ffi

    spec = gen_jit_spec(
        "mudnn_jit_mutlass_probe",
        [MUTLASS_PROBE_SOURCE],
        extra_include_paths=[jit_env.MUTLASS_INCLUDE_DIR],
    )
    module = spec.build_and_load()
    out = np.zeros(4, dtype=np.int64)
    module["mudnn_jit_mutlass_probe_fn"](
        tvm_ffi.from_dlpack(out), 0, _expected_cc()
    )

    probe_status = int(out[0])
    if probe_status == 1:
        pytest.skip("no MUSA device available")
    if probe_status == 8:
        pytest.skip(
            f"device mp{int(out[1])} is older than the compiled architecture"
        )

    compiled_arch = int(out[2])
    tag_capability = int(out[3])
    assert probe_status == 0, (
        f"mutlass probe failed with status {probe_status} "
        f"(compiled __MUSA_ARCH__={compiled_arch}, device_cc={int(out[1])})"
    )
    # 310 / 320 are the __MUSA_ARCH__ encodings for mp31 / mp32.
    assert compiled_arch in (310, 320), compiled_arch
    assert tag_capability == compiled_arch // 10, (
        f"mutlass ArchTag {tag_capability} disagrees with __MUSA_ARCH__ {compiled_arch}"
    )
    assert int(out[1]) == tag_capability, (
        "the kernel ran on a different architecture than it was compiled for"
    )


# ---------------------------------------------------------------------------
# Numeric execution (needs torch + MUSA)
# ---------------------------------------------------------------------------
def _torch_musa():
    try:
        import torch
    except ImportError:
        return None, None
    musa = getattr(torch, "musa", None)
    if musa is None or not musa.is_available():
        return None, None
    return torch, musa


def test_smoke_kernel_computes_the_right_answer():
    torch, musa = _torch_musa()
    if torch is None:
        pytest.skip("no MUSA-enabled torch available")

    module = _make_vec_add_spec().build_and_load()

    n = 1024
    a = torch.randn(n, dtype=torch.float32, device="musa")
    b = torch.randn(n, dtype=torch.float32, device="musa")
    c = torch.empty_like(a)
    alpha = 1.75

    module[VEC_ADD_FUNC](a, b, c, float(alpha))
    musa.synchronize()

    torch.testing.assert_close(c, a * alpha + b, rtol=0.0, atol=0.0)
