import json

import pytest


def test_pygraph_1221_defaults_and_compiled_module_surface():
    import mudnn

    graph = mudnn.pygraph()

    assert mudnn.__version__ == "0.1.0"
    assert mudnn._pybind_module.__name__ == "mudnn._compiled_module"
    assert graph.name == "test_graph"
    assert graph.io_data_type == mudnn.data_type.NOT_SET
    assert graph.intermediate_data_type == mudnn.data_type.NOT_SET
    assert graph.compute_data_type == mudnn.data_type.NOT_SET


def test_graph_json_uses_1221_protocol_and_reads_upstream_only_payload():
    import mudnn

    graph = mudnn.pygraph(
        "json_graph",
        mudnn.data_type.FLOAT,
        mudnn.data_type.FLOAT,
        mudnn.data_type.FLOAT,
    )
    a = graph.tensor(name="a", dim=[2, 3], stride=[3, 1])
    b = graph.tensor(name="b", dim=[2, 3], stride=[3, 1])
    graph.add(a, b, name="sum").set_output(True)

    payload = json.loads(graph.serialize())
    assert payload["json_version"] == "1.0"
    assert payload["context"]["name"] == "json_graph"
    assert payload["nodes"][0]["tag"] == "POINTWISE"
    assert payload["nodes"][0]["inputs"] == {"IN_0": 1, "IN_1": 2}
    assert set(payload) >= {"context", "nodes", "tensors", "json_version"}

    payload.pop("mudnn_payload")
    restored = mudnn.pygraph()
    restored.deserialize(payload)
    assert restored.name == "json_graph"
    assert restored.nodes[0].op_type == "pointwise"
    assert len(restored.tensors) == 3


def test_fp8_sdpa_and_autotune_are_explicitly_unsupported():
    import mudnn

    graph = mudnn.pygraph()
    tensor = graph.tensor(dim=[1, 1, 1, 1])

    # Graph construction stays source-compatible with upstream; rejection is
    # deferred to native lowering.
    graph.sdpa_fp8(
        tensor,
        tensor,
        tensor,
        descale_q=tensor,
        descale_k=tensor,
        descale_v=tensor,
        descale_s=tensor,
        scale_s=tensor,
        scale_o=tensor,
        generate_stats=False,
    )
    with pytest.raises(NotImplementedError, match="sdpa_fp8 native lowering"):
        graph.validate().build_operation_graph()

    with pytest.raises(NotImplementedError, match="execution-plan timing"):
        graph.autotune()
