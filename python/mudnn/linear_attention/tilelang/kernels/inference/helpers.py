"""Small runtime helpers for the migrated TileLang kernels."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Arch:
    major: int
    minor: int


MP31_ARCH = Arch(3, 1)
MP32_ARCH = Arch(3, 2)


def get_arch(device):
    properties = torch.musa.get_device_properties(device)
    return Arch(properties.major, properties.minor)


def cosize(shape, strides=None):
    if strides is None:
        stride = 1
        reversed_strides = []
        for extent in reversed(shape):
            reversed_strides.append(stride)
            stride *= extent
        strides = tuple(reversed(reversed_strides))
    return 1 + sum((extent - 1) * stride for extent, stride in zip(shape, strides))
