"""Public import surface for the renamed muDNN Python package."""


def test_mudnn_package_exports_frontend_and_optional_submodules():
    import mudnn

    graph = mudnn.pygraph()
    assert graph.name == "test_graph"
    assert mudnn.__version__ == "0.1.0"
    assert mudnn._pybind_module.__name__ == "mudnn._compiled_module"

    import mudnn.datatypes

    assert mudnn.datatypes.cudnn_data_type is mudnn.data_type


def test_mudnn_jit_keeps_the_compatibility_decorator():
    """``mudnn.jit`` is cuDNN's graph-JIT decorator, not the kernel JIT.

    ``cudnn.jit`` (``cudnn/graph.py``) is public API and this package re-exports
    it, so the runtime kernel-JIT facility must not answer to the same name:
    importing a submodule called ``mudnn.jit`` rebinds the attribute from the
    decorator to the module and breaks the compatibility surface for the rest
    of the process.
    """
    import types

    import mudnn

    assert callable(mudnn.jit), "mudnn.jit must be the compatibility decorator"
    assert not isinstance(mudnn.jit, types.ModuleType)

    from mudnn import _jit

    assert hasattr(_jit, "gen_jit_spec"), "the kernel JIT must stay importable"
    assert callable(mudnn.jit), "importing mudnn._jit shadowed mudnn.jit"
    assert not isinstance(mudnn.jit, types.ModuleType)
    assert mudnn.jit is not _jit
