# Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
# rights reserved.

from common.split_loader import export_tests


export_tests(
    globals(),
    include=(
        "matmul",
        "gemm",
        "grouped_matmul",
        "moe_grouped",
    ),
)
