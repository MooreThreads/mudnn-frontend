"""PyTorch-friendly wrapper compatibility for cuDNN frontend imports."""

from __future__ import annotations

import atexit
import inspect
import itertools
import logging
from collections import OrderedDict
from typing import Optional

import mudnn
import mudnn.datatypes

try:
    import torch
except ImportError:  # pragma: no cover - exercised only without torch installed.
    torch = None

data_type = mudnn.data_type
heur_mode = mudnn.heur_mode
CudnnHandle = int
logger = logging.getLogger(__name__)

_default_cudnn_handle = None


def _graph_tensor(graph: mudnn.pygraph, tensor):
    if hasattr(tensor, "requires_grad") and tensor.requires_grad:
        tensor = tensor.detach()
    return graph.tensor_like(tensor)


def _find_tensor(tensor, tensor_map, dlpack_map):
    if isinstance(tensor, str):
        if tensor in tensor_map:
            return tensor
        for name, value in tensor_map.items():
            if value.get_name() == tensor:
                return name
    elif isinstance(tensor, int):
        for name, value in tensor_map.items():
            if value.get_uid() == tensor:
                return name
    elif isinstance(tensor, mudnn.tensor):
        for name, value in tensor_map.items():
            if value is tensor:
                return name
    elif hasattr(tensor, "__dlpack__") and id(tensor) in dlpack_map:
        graph_tensor = dlpack_map[id(tensor)]
        for name, value in tensor_map.items():
            if value is graph_tensor:
                return name
    raise ValueError("Input not found in tensor map")


def _extract_tensor(name, tensor, arg_dict):
    for key in (name, tensor, tensor.get_name(), tensor.get_uid()):
        if key in arg_dict:
            return arg_dict[key]
    return None


def _tensor_like(cudnn_tensor, device=None):
    if torch is None:
        raise RuntimeError("PyTorch is not available")
    dtype = mudnn.datatypes._cudnn_to_torch_data_type(cudnn_tensor.get_data_type())
    if dtype is None:
        raise TypeError(
            f"cuDNN uses an unsupported PyTorch data type: {cudnn_tensor.get_data_type()}"
        )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.empty_strided(
        cudnn_tensor.get_dim(),
        cudnn_tensor.get_stride(),
        device=device,
        dtype=dtype,
    )


def _first_tensor_device(values):
    for value in values:
        device = getattr(value, "device", None)
        if device is not None:
            return device
    return None


def _contains_identity(values, candidate):
    return any(value is candidate for value in values)


def _tensor_is_virtual(tensor):
    return bool(getattr(tensor, "get_is_virtual", lambda: False)())


def get_default_handle(stream: Optional[object] = None):
    global _default_cudnn_handle
    if _default_cudnn_handle is None:
        _default_cudnn_handle = mudnn.create_handle()
    if stream is None and torch is not None and torch.cuda.is_available():
        stream = torch.cuda.current_stream().cuda_stream
    if stream is not None:
        mudnn.set_stream(handle=_default_cudnn_handle, stream=stream)
    return _default_cudnn_handle


def destroy_default_handle():
    global _default_cudnn_handle
    if _default_cudnn_handle is not None:
        mudnn.destroy_handle(_default_cudnn_handle)
        _default_cudnn_handle = None


atexit.register(destroy_default_handle)


class Graph:
    def __init__(
        self,
        *,
        handle: Optional[CudnnHandle] = None,
        inputs: Optional[list] = None,
        outputs: Optional[list] = None,
        heuristics: Optional[list] = None,
        workspace_alloc: bool = True,
        **kwargs,
    ):
        if torch is None:
            raise RuntimeError("PyTorch is not available")
        if inputs is not None and not isinstance(inputs, (list, tuple)):
            raise ValueError("inputs must be a list or tuple")
        if outputs is not None and not isinstance(outputs, (list, tuple)):
            raise ValueError("outputs must be a list or tuple")
        if heuristics is not None and not isinstance(heuristics, (list, tuple)):
            raise ValueError("heuristics must be a list or tuple")
        for key in ("io_data_type", "intermediate_data_type", "compute_data_type"):
            if key in kwargs:
                kwargs[key] = (
                    mudnn.datatypes._torch_to_cudnn_data_type(kwargs[key])
                    or kwargs[key]
                )
        self.__handle = handle
        self.__kwargs = kwargs
        self.__graph = None
        self.__tensor_map = {}
        self.__tensor_in = OrderedDict()
        self.__tensor_out = OrderedDict()
        self.__tensor_unknown = []
        self.__node_count = {}
        self.__node_names = set()
        self.__input_tuples = None
        self.__output_tuples = None
        self.__inputs = list(inputs or [])
        self.__outputs = list(outputs or [])
        self.__heuristics = tuple(heuristics or (heur_mode.A, heur_mode.FALLBACK))
        self.__workspace = None if workspace_alloc else False

    def __del__(self):
        pass

    def __enter__(self):
        if self.__graph is not None:
            raise RuntimeError("Graph already created")
        self.__graph = mudnn.pygraph(
            **(
                {"handle": self.__handle} if self.__handle not in ("auto", None) else {}
            ),
            **self.__kwargs,
        )
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            self.__graph = None
            return False
        self.__graph.validate()
        self.__graph.build_operation_graph()
        self.__graph.create_execution_plans(self.__heuristics)
        self.__graph.check_support()
        self.__graph.build_plans()
        if self.__workspace is None:
            workspace_size = int(self.__graph.get_workspace_size())
            self.__workspace = torch.empty(
                workspace_size,
                device="cuda" if torch.cuda.is_available() else "cpu",
                dtype=torch.uint8,
            )
        if self.__inputs or self.__outputs:
            self.set_io_tuples(self.__inputs, self.__outputs)
        return False

    def __getattr__(self, name):
        if self.__graph is None:
            raise RuntimeError("Graph not created")
        attr = getattr(self.__graph, name)
        if name in {
            "get_workspace_size",
            "get_workspace_size_plan_at_index",
            "serialize",
            "deserialize",
            "query_tensor_attributes_of_uid",
        }:
            return attr
        if name in {
            "build",
            "build_operation_graph",
            "build_plan_at_index",
            "build_plans",
            "check_support",
            "create_execution_plan",
            "create_execution_plans",
            "execute",
            "execute_plan_at_index",
            "get_engine_count",
            "get_execution_plan_count",
            "get_knobs_for_engine",
            "get_plan_name_at_index",
            "populate_cuda_graph",
            "update_cuda_graph",
            "validate",
        }:
            raise RuntimeError(f"Calling {name} via wrapper is not allowed")
        if not inspect.ismethod(attr):
            return attr

        def tensor_capture(*args, **kwargs):
            output = attr(*args, **kwargs)
            self.__tensor_unknown.append(output)
            return output

        if name in {"tensor", "tensor_like"}:
            return tensor_capture

        def wrapper(*args, **kwargs):
            args = list(args)
            node_name = kwargs.get("name")
            if node_name is None:
                count = self.__node_count.get(name, 0)
                self.__node_count[name] = count + 1
                node_name = f"{name}.{count}"
                kwargs["name"] = node_name
            if node_name in self.__node_names:
                raise ValueError(f"Node name {node_name} already used")
            self.__node_names.add(node_name)

            def convert(obj, key):
                if isinstance(obj, tuple):
                    converted = tuple(
                        convert(item, f"{key}::{index}")
                        for index, item in enumerate(obj)
                    )
                    return converted
                if hasattr(obj, "__dlpack__"):
                    obj_id = id(obj)
                    if obj_id not in self.__tensor_map:
                        self.__tensor_map[obj_id] = _graph_tensor(self.__graph, obj)
                    obj = self.__tensor_map[obj_id]
                if isinstance(obj, mudnn.tensor):
                    self.__tensor_in[f"{node_name}::{key}"] = obj
                return obj

            for index, obj in enumerate(args):
                args[index] = convert(obj, str(index))
            for key, obj in list(kwargs.items()):
                kwargs[key] = convert(obj, key)

            output = attr(*args, **kwargs)
            output_list = output if isinstance(output, (list, tuple)) else (output,)
            for index, obj in enumerate(output_list):
                if isinstance(obj, mudnn.tensor):
                    tensor_name = obj.get_name() or f"{node_name}::{index}"
                    self.__tensor_out[tensor_name] = obj
            return output

        return wrapper

    def __call__(self, *args, **kwargs):
        if self.__graph is None:
            raise RuntimeError("Graph not created")
        if not self.__graph.get_execution_plan_count():
            raise RuntimeError(
                "You should not invoke the graph before the context exits"
            )
        if len(args) == 1 and isinstance(args[0], dict):
            return self.__call_with_tensor_dict(args[0], **kwargs)
        if self.__input_tuples is None:
            raise ValueError(
                "You should not invoke the graph with positional arguments before running set_io_tuples()"
            )
        if len(args) != len(self.__input_tuples):
            raise ValueError(
                f"Number of arguments ({len(args)}) does not match number of inputs ({len(self.__input_tuples)})"
            )
        return self.__call_with_positional_args(*args, **kwargs)

    def __execution_kwargs(self, kwargs, tensors):
        kwargs = dict(kwargs)
        if "handle" not in kwargs:
            if self.__handle == "auto":
                kwargs["handle"] = get_default_handle()
            elif self.__handle is not None:
                kwargs["handle"] = self.__handle
        if "workspace" not in kwargs:
            if self.__workspace is False:
                raise RuntimeError("Need to specify workspace to execute graph")
            kwargs["workspace"] = self.__workspace
        return kwargs

    def __call_with_positional_args(self, *args, **kwargs):
        variant_pack = {}
        for graph_tensor, user_tensor in zip(self.__input_tuples, args):
            variant_pack[graph_tensor.get_uid()] = user_tensor

        output_device = _first_tensor_device(args)
        outputs = tuple(
            _tensor_like(graph_tensor, output_device)
            for graph_tensor in self.__output_tuples
        )
        for graph_tensor, user_tensor in zip(self.__output_tuples, outputs):
            variant_pack[graph_tensor.get_uid()] = user_tensor

        self.__graph.execute(
            variant_pack,
            **self.__execution_kwargs(kwargs, itertools.chain(args, outputs)),
        )
        return outputs[0] if len(outputs) == 1 else outputs

    def __call_with_tensor_dict(self, tensor_dict, **kwargs):
        variant_pack = {}
        missing_inputs = []
        output_device = _first_tensor_device(tensor_dict.values())
        for name, tensor in itertools.chain(
            self.__tensor_in.items(), self.__tensor_out.items()
        ):
            if tensor.get_uid() in variant_pack or tensor.get_is_virtual():
                continue
            user_tensor = _extract_tensor(name, tensor, tensor_dict)
            if user_tensor is None:
                if name in self.__tensor_out:
                    user_tensor = _tensor_like(tensor, output_device)
                    tensor_dict[name] = user_tensor
                else:
                    missing_inputs.append(name)
                    continue
            if not hasattr(user_tensor, "__dlpack__"):
                raise RuntimeError(f"Tensor {name} is not provided as a dlpack tensor")
            variant_pack[tensor.get_uid()] = user_tensor
        if missing_inputs:
            raise RuntimeError(
                f"Non-virtual input tensors not found in variant pack: {missing_inputs}"
            )
        self.__graph.execute(
            variant_pack, **self.__execution_kwargs(kwargs, tensor_dict.values())
        )
        return tensor_dict

    def set_io_tuples(self, inputs, outputs):
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("inputs must be a list or tuple")
        if not isinstance(outputs, (list, tuple)):
            raise ValueError("outputs must be a list or tuple")
        if not self.__graph.get_execution_plan_count():
            self.__inputs = list(inputs)
            self.__outputs = list(outputs)
            return None
        input_tensors = []
        seen = set()
        for index, value in enumerate(inputs):
            try:
                if _contains_identity(self.__tensor_unknown, value):
                    tensor = value
                else:
                    tensor = self.__tensor_in[
                        _find_tensor(value, self.__tensor_in, self.__tensor_map)
                    ]
            except ValueError:
                raise ValueError(
                    f"Input at index {index} ({value}) not found in tensor map"
                ) from None
            if tensor.get_uid() in seen:
                raise ValueError(f"Input at index {index} ({value}) is a duplicate")
            if _tensor_is_virtual(tensor):
                raise ValueError(f"Input at index {index} is a virtual tensor")
            seen.add(tensor.get_uid())
            input_tensors.append(tensor)

        output_tensors = []
        output_uids = set()
        for index, value in enumerate(outputs):
            try:
                if _contains_identity(self.__tensor_unknown, value):
                    tensor = value
                else:
                    tensor = self.__tensor_out[
                        _find_tensor(value, self.__tensor_out, self.__tensor_map)
                    ]
            except ValueError:
                raise ValueError(
                    f"Output at index {index} ({value}) not found in tensor map"
                ) from None
            if tensor.get_uid() in seen:
                raise ValueError(f"Output at index {index} ({value}) is a duplicate")
            seen.add(tensor.get_uid())
            output_uids.add(tensor.get_uid())
            output_tensors.append(tensor)
        for name, tensor in self.__tensor_out.items():
            if not _tensor_is_virtual(tensor) and tensor.get_uid() not in output_uids:
                raise ValueError(
                    f"Node output {name} is a non-virtual tensor but not specified as output"
                )
        for name, tensor in self.__tensor_in.items():
            if not _tensor_is_virtual(tensor) and tensor.get_uid() not in seen:
                raise ValueError(
                    f"Node input {name} is a non-virtual tensor but not specified as input or output"
                )
        self.__input_tuples = tuple(input_tensors)
        self.__output_tuples = tuple(output_tensors)
        self.__graph.set_io_tuples(self.__input_tuples, self.__output_tuples)
        return None


__all__ = [
    "Graph",
    "data_type",
    "heur_mode",
    "mudnn",
    "CudnnHandle",
    "logger",
    "get_default_handle",
    "destroy_default_handle",
]
