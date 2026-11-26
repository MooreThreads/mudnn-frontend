"""Minimal API-base compatibility helpers for optional cuDNN FE APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Optional, Tuple

from mudnn._experimental_warnings import (
    _reset_experimental_api_warning_registry,
    warn_experimental_api_once,
)

try:
    import torch
except Exception:  # pragma: no cover - torch is expected in upstream tests
    torch = None


def ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def is_power_of_2(n: int) -> bool:
    n = int(n)
    return n > 0 and (n & (n - 1)) == 0


@dataclass(frozen=True)
class TensorDesc:
    dtype: object
    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    stride_order: Tuple[int, ...]
    device: object
    interpret_uint8_as_fp4x2: bool = False
    name: str = ""
    ndim: int = field(init=False)

    def __post_init__(self):
        shape = tuple(int(dim) for dim in self.shape)
        stride = tuple(int(value) for value in self.stride)
        stride_order = tuple(int(dim) for dim in self.stride_order)
        if len(stride) != len(shape):
            raise ValueError(
                f"Stride rank mismatch: expected {len(shape)}, got {len(stride)}"
            )
        if len(stride_order) != len(shape):
            raise ValueError(
                f"Stride order rank mismatch: expected {len(shape)}, got {len(stride_order)}"
            )
        if tuple(sorted(stride_order)) != tuple(range(len(shape))):
            raise ValueError("Stride order must be a permutation of tensor dimensions")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "stride", stride)
        object.__setattr__(self, "stride_order", stride_order)
        object.__setattr__(self, "ndim", len(shape))

    @staticmethod
    def _normalize_dim(dim: int, ndim: int, *, allow_new_dim: bool = False) -> int:
        min_dim = -ndim - (1 if allow_new_dim else 0)
        max_dim = ndim if allow_new_dim else ndim - 1
        if dim < min_dim or dim > max_dim:
            raise IndexError("dimension out of range")
        if dim < 0:
            dim += ndim + (1 if allow_new_dim else 0)
        return dim

    @staticmethod
    def _compute_stride_order(
        shape: Tuple[int, ...], stride: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        return tuple(
            index
            for index, _ in sorted(
                enumerate(stride), key=lambda item: (item[1], shape[item[0]])
            )
        )

    @staticmethod
    def _numel(shape: Tuple[int, ...]) -> int:
        result = 1
        for dim in shape:
            result *= int(dim)
        return result

    @staticmethod
    def _compute_contiguous_stride(shape: Tuple[int, ...]) -> Tuple[int, ...]:
        stride = [0] * len(shape)
        running = 1
        for index in range(len(shape) - 1, -1, -1):
            stride[index] = running
            running *= max(int(shape[index]), 1)
        return tuple(stride)

    @classmethod
    def _compute_view_stride(
        cls,
        old_shape: Tuple[int, ...],
        old_stride: Tuple[int, ...],
        new_shape: Tuple[int, ...],
    ) -> Optional[Tuple[int, ...]]:
        if cls._numel(old_shape) == 0 or len(old_shape) == 0:
            return cls._compute_contiguous_stride(new_shape)

        new_stride = [0] * len(new_shape)
        view_dim = len(new_shape) - 1
        chunk_base_stride = old_stride[-1]
        tensor_numel = 1
        view_numel = 1

        for tensor_dim in range(len(old_shape) - 1, -1, -1):
            tensor_numel *= old_shape[tensor_dim]
            starts_new_chunk = tensor_dim == 0 or (
                old_shape[tensor_dim - 1] != 1
                and old_stride[tensor_dim - 1] != tensor_numel * chunk_base_stride
            )
            if not starts_new_chunk:
                continue

            while view_dim >= 0 and (
                view_numel < tensor_numel or new_shape[view_dim] == 1
            ):
                new_stride[view_dim] = view_numel * chunk_base_stride
                view_numel *= new_shape[view_dim]
                view_dim -= 1

            if view_numel != tensor_numel:
                return None

            if tensor_dim > 0:
                chunk_base_stride = old_stride[tensor_dim - 1]
                tensor_numel = 1
                view_numel = 1

        if view_dim >= 0:
            return None
        return tuple(new_stride)

    @staticmethod
    def _is_contiguous_with_order(
        shape: Tuple[int, ...], stride: Tuple[int, ...], order: Tuple[int, ...]
    ) -> bool:
        expected = 1
        for dim in order:
            size = int(shape[dim])
            if size == 1:
                continue
            if int(stride[dim]) != expected:
                return False
            expected *= size
        return True

    def _with_layout(self, shape: Tuple[int, ...], stride: Tuple[int, ...]):
        return TensorDesc(
            dtype=self.dtype,
            shape=shape,
            stride=stride,
            stride_order=self._compute_stride_order(shape, stride),
            device=self.device,
            interpret_uint8_as_fp4x2=self.interpret_uint8_as_fp4x2,
            name=self.name,
        )

    @classmethod
    def from_tensor(
        cls, tensor, *, name: str = "", interpret_uint8_as_fp4x2: bool = False
    ):
        return cls(
            dtype=tensor.dtype,
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            stride_order=cls._compute_stride_order(
                tuple(tensor.shape), tuple(tensor.stride())
            ),
            device=tensor.device,
            interpret_uint8_as_fp4x2=interpret_uint8_as_fp4x2,
            name=name,
        )

    def size(self, dim: Optional[int] = None):
        if dim is None:
            return torch.Size(self.shape) if torch is not None else self.shape
        return self.shape[self._normalize_dim(int(dim), self.ndim)]

    def __len__(self):
        if self.ndim == 0:
            raise TypeError("len() of a 0-d tensor")
        return self.shape[0]

    def permute(self, *dims):
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        dims = tuple(self._normalize_dim(int(dim), self.ndim) for dim in dims)
        if len(dims) != self.ndim:
            raise RuntimeError(f"permute(): expected {self.ndim} dims")
        if len(set(dims)) != self.ndim:
            raise RuntimeError("permute(): dims must be unique")
        return self._with_layout(
            tuple(self.shape[dim] for dim in dims),
            tuple(self.stride[dim] for dim in dims),
        )

    def transpose(self, dim0: int, dim1: int):
        dim0 = self._normalize_dim(int(dim0), self.ndim)
        dim1 = self._normalize_dim(int(dim1), self.ndim)
        if dim0 == dim1:
            return self
        dims = list(range(self.ndim))
        dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
        return self.permute(dims)

    def squeeze(self, dim=None):
        if dim is None:
            keep = [idx for idx, size in enumerate(self.shape) if size != 1]
        elif isinstance(dim, (tuple, list)):
            squeeze_dims = tuple(self._normalize_dim(int(d), self.ndim) for d in dim)
            if len(set(squeeze_dims)) != len(squeeze_dims):
                raise RuntimeError("squeeze(): dims must be unique")
            squeeze_dims = {d for d in squeeze_dims if self.shape[d] == 1}
            keep = [idx for idx in range(self.ndim) if idx not in squeeze_dims]
        else:
            squeeze_dim = self._normalize_dim(int(dim), self.ndim)
            if self.shape[squeeze_dim] != 1:
                return self
            keep = [idx for idx in range(self.ndim) if idx != squeeze_dim]
        if len(keep) == self.ndim:
            return self
        return self._with_layout(
            tuple(self.shape[idx] for idx in keep),
            tuple(self.stride[idx] for idx in keep),
        )

    def unsqueeze(self, dim: int):
        dim = self._normalize_dim(int(dim), self.ndim, allow_new_dim=True)
        inserted_stride = 1 if dim >= self.ndim else self.stride[dim] * self.shape[dim]
        return self._with_layout(
            self.shape[:dim] + (1,) + self.shape[dim:],
            self.stride[:dim] + (inserted_stride,) + self.stride[dim:],
        )

    def is_contiguous(self, memory_format=None) -> bool:
        if torch is not None and memory_format is None:
            memory_format = torch.contiguous_format
        if torch is None or memory_format in {
            torch.contiguous_format,
            torch.preserve_format,
        }:
            if self._numel(self.shape) == 0:
                return True
            return self._is_contiguous_with_order(
                self.shape, self.stride, tuple(range(self.ndim - 1, -1, -1))
            )
        if memory_format == torch.channels_last:
            return self.ndim == 4 and self._is_contiguous_with_order(
                self.shape, self.stride, (1, 3, 2, 0)
            )
        if memory_format == torch.channels_last_3d:
            return self.ndim == 5 and self._is_contiguous_with_order(
                self.shape, self.stride, (1, 4, 3, 2, 0)
            )
        raise ValueError(f"Unsupported memory format: {memory_format}")

    def contiguous(self):
        if self.is_contiguous():
            return self
        return self._with_layout(
            self.shape, self._compute_contiguous_stride(self.shape)
        )

    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        new_shape = tuple(int(dim) for dim in shape)
        old_numel = self._numel(self.shape)
        infer_dim = None
        known_numel = 1
        for index, dim in enumerate(new_shape):
            if dim == -1:
                if infer_dim is not None:
                    raise RuntimeError("only one dimension can be inferred")
                infer_dim = index
                continue
            if dim < 0:
                raise RuntimeError(f"invalid shape dimension {dim}")
            known_numel *= dim
        if infer_dim is not None:
            if known_numel == 0 or old_numel % known_numel != 0:
                raise RuntimeError("shape is invalid for input size")
            inferred = old_numel // known_numel
            new_shape = new_shape[:infer_dim] + (inferred,) + new_shape[infer_dim + 1 :]
            known_numel *= inferred
        if known_numel != old_numel:
            raise RuntimeError("shape is invalid for input size")
        new_stride = self._compute_view_stride(self.shape, self.stride, new_shape)
        if new_stride is None:
            raise RuntimeError(
                "view size is not compatible with input tensor's size and stride"
            )
        return self._with_layout(new_shape, new_stride)

    def as_strided(self, size, stride, storage_offset: int = 0):
        if int(storage_offset) != 0:
            raise RuntimeError("TensorDesc.as_strided(): non-zero storage_offset")
        size = tuple(int(dim) for dim in size)
        stride = tuple(int(value) for value in stride)
        if len(size) != len(stride):
            raise RuntimeError("TensorDesc.as_strided(): size/stride rank mismatch")
        if any(dim < 0 for dim in size) or any(value < 0 for value in stride):
            raise RuntimeError("TensorDesc.as_strided(): negative size/stride")
        return self._with_layout(size, stride)


class APIBase:
    def __init__(self):
        self._is_supported = False
        self._kernel = None
        self._compiled_kernel = None
        self._interpret_uint8_as_fp4x2 = False
        self._logger = logging.getLogger(self.__class__.__name__)

    def _warn_experimental_api(self) -> None:
        warn_experimental_api_once(self._logger, self.__class__.__name__)

    def check_support(self) -> bool:
        raise NotImplementedError(
            f"{self.__class__.__name__} is not supported by muDNN frontend yet"
        )

    def compile(self) -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__} compilation is not supported by muDNN frontend yet"
        )

    def execute(self, *args, **kwargs) -> Any:
        raise NotImplementedError(
            f"{self.__class__.__name__} execution is not supported by muDNN frontend yet"
        )

    def __call__(self, *args, **kwargs) -> Any:
        if self._compiled_kernel is None:
            self.compile()
        return self.execute(*args, **kwargs)

    def _ensure_support_checked(self):
        if not self._is_supported:
            self.check_support()

    def _get_default_stream(self, stream=None):
        if stream is not None:
            return stream
        try:
            import cutlass

            return cutlass.cuda.default_stream()
        except Exception as exc:
            raise NotImplementedError(
                "default CUDA stream lookup is not available in muDNN frontend"
            ) from exc

    def _pad_tensor_to_ndim(self, tensor, ndim: int, name: str = ""):
        if tensor is None:
            return None
        if tensor.ndim < int(ndim):
            self._logger.info("Padding %s to %dD from %s", name, ndim, tensor.shape)
            for _ in range(int(ndim) - tensor.ndim):
                tensor = tensor.unsqueeze(-1)
        return tensor

    def _unpad_tensor_to_ndim(self, tensor, ndim: int, name: str = ""):
        if tensor is None:
            return None
        if tensor.ndim > int(ndim):
            self._logger.info("Unpadding %s from %s to %dD", name, tensor.shape, ndim)
            for _ in range(tensor.ndim - int(ndim)):
                if tensor.shape and tensor.shape[-1] == 1:
                    tensor = tensor.squeeze(-1)
                else:
                    break
            if tensor.ndim != int(ndim):
                self._logger.critical(
                    "Unpadding %s resulted in shape %s, expected %dD",
                    name,
                    tensor.shape,
                    ndim,
                )
        return tensor

    def _is_fp4x2(self, tensor_or_dtype) -> bool:
        if tensor_or_dtype is None or torch is None:
            return False
        if isinstance(tensor_or_dtype, TensorDesc):
            dtype = tensor_or_dtype.dtype
            interpret_uint8_as_fp4x2 = tensor_or_dtype.interpret_uint8_as_fp4x2
        elif isinstance(tensor_or_dtype, torch.Tensor):
            dtype = tensor_or_dtype.dtype
            interpret_uint8_as_fp4x2 = self._interpret_uint8_as_fp4x2
        else:
            dtype = tensor_or_dtype
            interpret_uint8_as_fp4x2 = self._interpret_uint8_as_fp4x2
        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", object())
        return dtype == fp4_dtype or (interpret_uint8_as_fp4x2 and dtype == torch.uint8)

    def _is_fp8(self, tensor_or_dtype) -> bool:
        if tensor_or_dtype is None or torch is None:
            return False
        dtype = (
            tensor_or_dtype.dtype
            if isinstance(tensor_or_dtype, (torch.Tensor, TensorDesc))
            else tensor_or_dtype
        )
        fp8_dtypes = {
            dtype
            for dtype in (
                getattr(torch, "float8_e5m2", None),
                getattr(torch, "float8_e4m3fn", None),
            )
            if dtype is not None
        }
        return dtype in fp8_dtypes

    def _is_f16(self, tensor_or_dtype) -> bool:
        if tensor_or_dtype is None or torch is None:
            return False
        dtype = (
            tensor_or_dtype.dtype
            if isinstance(tensor_or_dtype, (torch.Tensor, TensorDesc))
            else tensor_or_dtype
        )
        return dtype in {torch.float16, torch.bfloat16}

    def _get_innermost_stride_dim(self, tensor, name: str = "") -> int:
        stride = tensor.stride if isinstance(tensor, TensorDesc) else tensor.stride()
        index = next((i for i, value in enumerate(stride) if value == 1), None)
        if index is None:
            shape = tensor.shape
            message = (
                f"tensor {name} has shape: {shape} stride {stride} - "
                "innermost contiguous (stride == 1) dimension not found."
            )
            self._logger.critical(message)
            raise RuntimeError(message)
        return index

    def _tensor_shape(self, tensor, name: str = ""):
        if tensor is None:
            return None
        if isinstance(tensor, TensorDesc):
            return tensor.shape
        if self._is_fp4x2(tensor):
            innermost_dim = self._get_innermost_stride_dim(tensor, name=name)
            return tuple(
                dim * 2 if index == innermost_dim else dim
                for index, dim in enumerate(tensor.shape)
            )
        return tuple(tensor.shape)

    def _tensor_stride(self, tensor, name: str = ""):
        if tensor is None:
            return None
        if isinstance(tensor, TensorDesc):
            return tensor.stride
        stride = tuple(tensor.stride())
        if self._is_fp4x2(tensor):
            innermost_dim = self._get_innermost_stride_dim(tensor, name=name)
            return tuple(
                value if index == innermost_dim else value * 2
                for index, value in enumerate(stride)
            )
        return stride

    def _check_tensor_shape(self, tensor_or_shape, shape, name: str = ""):
        if tensor_or_shape is None:
            return None
        tensor_shape = (
            self._tensor_shape(tensor_or_shape, name=name)
            if torch is not None
            and isinstance(tensor_or_shape, (torch.Tensor, TensorDesc))
            else tuple(tensor_or_shape)
        )
        if isinstance(shape, tuple):
            expected = tuple(shape)
            if tensor_shape != expected:
                raise ValueError(
                    f"{name} tensor shape mismatch: expected {expected}, got {tensor_shape}"
                )
        elif isinstance(shape, list):
            expected = [tuple(item) for item in shape]
            if tensor_shape not in expected:
                raise ValueError(
                    f"{name} tensor shape mismatch: expected one of {expected}, got {tensor_shape}"
                )
        else:
            raise ValueError(f"Expected shape to be a tuple or list, got {type(shape)}")
        return tensor_shape

    def _check_tensor_stride(
        self,
        tensor_or_stride,
        stride=None,
        stride_order=None,
        name: str = "",
        extra_error_msg: str = "",
    ):
        if tensor_or_stride is None:
            return None, None
        if isinstance(tensor_or_stride, TensorDesc):
            tensor_stride = tensor_or_stride.stride
            tensor_stride_order = tensor_or_stride.stride_order
        elif torch is not None and isinstance(tensor_or_stride, torch.Tensor):
            tensor_stride = self._tensor_stride(tensor_or_stride, name=name)
            tensor_shape = self._tensor_shape(tensor_or_stride, name=name)
            tensor_stride_order = TensorDesc._compute_stride_order(
                tensor_shape, tensor_stride
            )
        else:
            tensor_stride = tuple(tensor_or_stride)
            tensor_stride_order = tuple(
                index
                for index, _ in sorted(
                    enumerate(tensor_stride), key=lambda item: item[1]
                )
            )

        def with_extra(message: str) -> str:
            return f"{message}: {extra_error_msg}" if extra_error_msg else message

        if stride is not None:
            if isinstance(stride, tuple):
                expected = tuple(stride)
                if tensor_stride != expected:
                    raise ValueError(
                        with_extra(
                            f"{name} tensor stride mismatch: expected {expected}, got {tensor_stride}"
                        )
                    )
            elif isinstance(stride, list):
                expected = [tuple(item) for item in stride]
                if tensor_stride not in expected:
                    raise ValueError(
                        with_extra(
                            f"{name} tensor stride mismatch: expected one of {expected}, got {tensor_stride}"
                        )
                    )
            else:
                raise ValueError(
                    with_extra(
                        f"Expected stride to be a tuple or list, got {type(stride)}"
                    )
                )
        if stride_order is not None:
            if isinstance(stride_order, tuple):
                expected = tuple(stride_order)
                if tensor_stride_order != expected:
                    raise ValueError(
                        with_extra(
                            f"{name} tensor stride order mismatch: expected {expected}, got {tensor_stride_order}"
                        )
                    )
            elif isinstance(stride_order, list):
                expected = [tuple(item) for item in stride_order]
                if tensor_stride_order not in expected:
                    raise ValueError(
                        with_extra(
                            f"{name} tensor stride order mismatch: expected one of {expected}, got {tensor_stride_order}"
                        )
                    )
            else:
                raise ValueError(
                    with_extra(
                        f"Expected stride order to be a tuple or list, got {type(stride_order)}"
                    )
                )
        return tensor_stride, tensor_stride_order

    def _check_dtype(
        self, tensor_or_dtype, dtype, name: str = "", extra_error_msg: str = ""
    ):
        if tensor_or_dtype is None:
            return None
        tensor_dtype = (
            tensor_or_dtype.dtype
            if torch is not None
            and isinstance(tensor_or_dtype, (torch.Tensor, TensorDesc))
            else tensor_or_dtype
        )

        def with_extra(message: str) -> str:
            return f"{message}: {extra_error_msg}" if extra_error_msg else message

        if torch is not None and isinstance(dtype, torch.dtype):
            if tensor_dtype != dtype:
                raise ValueError(
                    with_extra(
                        f"{name} dtype mismatch: expected {dtype}, got {tensor_dtype}"
                    )
                )
        elif isinstance(dtype, list):
            if tensor_dtype not in dtype:
                raise ValueError(
                    with_extra(
                        f"{name} dtype mismatch: expected one of {dtype}, got {tensor_dtype}"
                    )
                )
        else:
            raise ValueError(
                f"Expected dtype to be a torch.dtype or list, got {type(dtype)}"
            )
        return tensor_dtype

    def _value_error_if(self, condition: bool, error_msg: str) -> None:
        if condition:
            raise ValueError(error_msg)

    def _not_implemented_error_if(self, condition: bool, error_msg: str) -> None:
        if condition:
            raise NotImplementedError(error_msg)

    def _runtime_error_if(self, condition: bool, error_msg: str) -> None:
        if condition:
            raise RuntimeError(error_msg)

    def _make_tensor_desc(
        self, tensor, name: str = "", interpret_uint8_as_fp4x2: Optional[bool] = None
    ):
        if tensor is None:
            return None
        if torch is None:
            raise NotImplementedError("TensorDesc creation requires torch")
        if isinstance(tensor, TensorDesc):
            return tensor
        if interpret_uint8_as_fp4x2 is None:
            interpret_uint8_as_fp4x2 = self._interpret_uint8_as_fp4x2
        previous = self._interpret_uint8_as_fp4x2
        self._interpret_uint8_as_fp4x2 = interpret_uint8_as_fp4x2
        try:
            tensor_shape = self._tensor_shape(tensor, name=name)
            tensor_stride = self._tensor_stride(tensor, name=name)
        finally:
            self._interpret_uint8_as_fp4x2 = previous
        return TensorDesc(
            dtype=tensor.dtype,
            shape=tensor_shape,
            stride=tensor_stride,
            stride_order=TensorDesc._compute_stride_order(tensor_shape, tensor_stride),
            device=tensor.device,
            interpret_uint8_as_fp4x2=interpret_uint8_as_fp4x2,
            name=name,
        )

    def _make_fake_cute_tensor_like(self, *args, **kwargs):
        raise NotImplementedError(
            "cute tensor metadata is not available in muDNN frontend"
        )

    def _make_fake_cute_tensor_from_desc(self, *args, **kwargs):
        raise NotImplementedError(
            "cute tensor metadata is not available in muDNN frontend"
        )

    def _make_fake_cute_tensor(self, *args, **kwargs):
        raise NotImplementedError(
            "cute tensor metadata is not available in muDNN frontend"
        )

    def _make_fake_cute_compact_tensor(self, *args, **kwargs):
        raise NotImplementedError(
            "cute tensor metadata is not available in muDNN frontend"
        )


class TupleDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._keys = list(self.keys())

    def __iter__(self):
        return (self[key] for key in self._keys)

    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0 or key >= len(self._keys):
                raise IndexError(
                    f"index {key} out of range for TupleDict with {len(self._keys)} items"
                )
            return super().__getitem__(self._keys[key])
        return super().__getitem__(key)


__all__ = [
    "APIBase",
    "TensorDesc",
    "TupleDict",
    "ceil_div",
    "is_power_of_2",
    "_reset_experimental_api_warning_registry",
    "warn_experimental_api_once",
]
