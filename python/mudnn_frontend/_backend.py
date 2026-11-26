from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import os
import shutil
import subprocess
from pathlib import Path


MUDNN_STATUS_SUCCESS = 0
MUDNN_STATUS_NOT_INITIALIZED = 1001
MUDNN_STATUS_BAD_PARAM = 2000
MUDNN_STATUS_NOT_SUPPORTED = 3000
MUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE = 3003
MUDNN_STATUS_INTERNAL_ERROR = 4000
MUDNN_STATUS_ALLOC_FAILED = 4003

MUDNN_TYPE_HANDLE = 0
MUDNN_TYPE_DATA_TYPE = 1
MUDNN_TYPE_BOOLEAN = 2
MUDNN_TYPE_INT64 = 3
MUDNN_TYPE_FLOAT = 4
MUDNN_TYPE_DOUBLE = 5
MUDNN_TYPE_VOID_PTR = 6
MUDNN_TYPE_CONVOLUTION_MODE = 7
MUDNN_TYPE_HEUR_MODE = 8
MUDNN_TYPE_KNOB_TYPE = 9
MUDNN_TYPE_NUMERICAL_NOTE = 11
MUDNN_TYPE_POINTWISE_MODE = 14
MUDNN_TYPE_BACKEND_DESCRIPTOR = 15
MUDNN_TYPE_GENSTATS_MODE = 16
MUDNN_TYPE_BN_FINALIZE_STATS_MODE = 17
MUDNN_TYPE_REDUCTION_OPERATOR_TYPE = 18
MUDNN_TYPE_BEHAVIOR_NOTE = 19
MUDNN_TYPE_TENSOR_REORDERING_MODE = 20
MUDNN_TYPE_RESAMPLE_MODE = 21
MUDNN_TYPE_NORM_MODE = 27
MUDNN_TYPE_NORM_FWD_PHASE = 28
MUDNN_TYPE_RNG_DISTRIBUTION = 29
MUDNN_TYPE_MOE_GROUPED_MATMUL_MODE = 30
MUDNN_TYPE_RESHAPE_MODE = 31
MUDNN_TYPE_OPERATIONGRAPH_MODE = 32
MUDNN_TYPE_SIGNAL_MODE = 25

MUDNN_BACKEND_POINTWISE_DESCRIPTOR = 0
MUDNN_BACKEND_CONVOLUTION_DESCRIPTOR = 1
MUDNN_BACKEND_ENGINE_DESCRIPTOR = 2
MUDNN_BACKEND_ENGINECFG_DESCRIPTOR = 3
MUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR = 4
MUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR = 5
MUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR = 13
MUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR = 15
MUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR = 16
MUDNN_BACKEND_TENSOR_DESCRIPTOR = 17
MUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR = 10
MUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR = 11
MUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR = 12
MUDNN_BACKEND_MATMUL_DESCRIPTOR = 18
MUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR = 19
MUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR = 20
MUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR = 14
MUDNN_BACKEND_REDUCTION_DESCRIPTOR = 21
MUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR = 22
MUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR = 23
MUDNN_BACKEND_RESAMPLE_DESCRIPTOR = 24
MUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR = 25
MUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR = 26
MUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR = 27
MUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR = 29
MUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR = 30
MUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR = 31
MUDNN_BACKEND_RNG_DESCRIPTOR = 32
MUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR = 33
MUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR = 28
MUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR = 35
MUDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR = 36
MUDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR = 37
MUDNN_BACKEND_OPERATION_EXPAND_BAND_MATRIX_DESCRIPTOR = 39
MUDNN_BACKEND_OPERATION_CONTRACT_BAND_MATRIX_DESCRIPTOR = 40
MUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR = 41
MUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR = 42
MUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR = 43
MUDNN_BACKEND_OPERATION_DIAGONAL_BAND_MASK_DESCRIPTOR = 44
MUDNN_BACKEND_OPERATION_SOFTMAX_DESCRIPTOR = 45
MUDNN_BACKEND_OPERATION_TRANSPOSE_DESCRIPTOR = 46
MUDNN_BACKEND_OPERATION_SLICE_DESCRIPTOR = 47
MUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR = 48
MUDNN_BACKEND_OPERATION_ROPE_FWD_DESCRIPTOR = 49
MUDNN_BACKEND_OPERATION_ROPE_BWD_DESCRIPTOR = 50

MUDNN_ATTR_POINTWISE_MODE = 0
MUDNN_ATTR_POINTWISE_MATH_PREC = 1
MUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP = 3
MUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP = 4
MUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE = 5
MUDNN_ATTR_POINTWISE_ELU_ALPHA = 6
MUDNN_ATTR_POINTWISE_SOFTPLUS_BETA = 7
MUDNN_ATTR_POINTWISE_SWISH_BETA = 8
MUDNN_ATTR_POINTWISE_AXIS = 9
MUDNN_ATTR_CONVOLUTION_COMP_TYPE = 100
MUDNN_ATTR_CONVOLUTION_CONV_MODE = 101
MUDNN_ATTR_CONVOLUTION_DILATIONS = 102
MUDNN_ATTR_CONVOLUTION_FILTER_STRIDES = 103
MUDNN_ATTR_CONVOLUTION_POST_PADDINGS = 104
MUDNN_ATTR_CONVOLUTION_PRE_PADDINGS = 105
MUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS = 106
# muDNN-private graph attributes intentionally live outside the cuDNN enum
# range. Keep these values synchronized with mudnn_graph.h.
MUDNN_ATTR_CONVOLUTION_GROUP_COUNT = 9000
MUDNN_ATTR_ENGINEHEUR_MODE = 200
MUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH = 201
MUDNN_ATTR_ENGINEHEUR_RESULTS = 202
MUDNN_ATTR_ENGINECFG_ENGINE = 300
MUDNN_ATTR_ENGINECFG_KNOB_CHOICES = 302
MUDNN_ATTR_ENGINECFG_WORKSPACE_SIZE = 303
MUDNN_ATTR_ENGINECFG_SHARED_MEMORY_USED = 304
MUDNN_ATTR_EXECUTION_PLAN_HANDLE = 400
MUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG = 401
MUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE = 402
MUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA = 700
MUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA = 701
MUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC = 702
MUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W = 703
MUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X = 704
MUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y = 705
MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA = 706
MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA = 707
MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC = 708
MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W = 709
MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX = 710
MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY = 711
MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA = 712
MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA = 713
MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC = 714
MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW = 715
MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X = 716
MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY = 717
MUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR = 750
MUDNN_ATTR_OPERATION_POINTWISE_XDESC = 751
MUDNN_ATTR_OPERATION_POINTWISE_BDESC = 752
MUDNN_ATTR_OPERATION_POINTWISE_YDESC = 753
MUDNN_ATTR_OPERATION_POINTWISE_ALPHA1 = 754
MUDNN_ATTR_OPERATION_POINTWISE_ALPHA2 = 755
MUDNN_ATTR_OPERATION_POINTWISE_DXDESC = 756
MUDNN_ATTR_OPERATION_POINTWISE_DYDESC = 757
MUDNN_ATTR_OPERATION_POINTWISE_TDESC = 758
MUDNN_ATTR_OPERATION_GENSTATS_MODE = 770
MUDNN_ATTR_OPERATION_GENSTATS_MATH_PREC = 771
MUDNN_ATTR_OPERATION_GENSTATS_XDESC = 772
MUDNN_ATTR_OPERATION_GENSTATS_SUMDESC = 773
MUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC = 774
MUDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE = 780
MUDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC = 781
MUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC = 782
MUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC = 783
MUDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC = 784
MUDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC = 785
MUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC = 786
MUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC = 787
MUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC = 788
MUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC = 789
MUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC = 790
MUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC = 791
MUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC = 792
MUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC = 793
MUDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC = 794
MUDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC = 795
MUDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC = 796
MUDNN_ATTR_OPERATIONGRAPH_HANDLE = 800
MUDNN_ATTR_OPERATIONGRAPH_OPS = 801
MUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT = 802
MUDNN_ATTR_OPERATIONGRAPH_IS_DYNAMIC_SHAPE_ENABLED = 803
MUDNN_ATTR_OPERATIONGRAPH_IS_SAME_TOPOLOGY = 804
MUDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED = 805
MUDNN_ATTR_OPERATIONGRAPH_MODE = 806
MUDNN_ATTR_TENSOR_BYTE_ALIGNMENT = 900
MUDNN_ATTR_TENSOR_DATA_TYPE = 901
MUDNN_ATTR_TENSOR_DIMENSIONS = 902
MUDNN_ATTR_TENSOR_STRIDES = 903
MUDNN_ATTR_TENSOR_VECTOR_COUNT = 904
MUDNN_ATTR_TENSOR_VECTORIZED_DIMENSION = 905
MUDNN_ATTR_TENSOR_UNIQUE_ID = 906
MUDNN_ATTR_TENSOR_IS_VIRTUAL = 907
MUDNN_ATTR_TENSOR_IS_BY_VALUE = 908
MUDNN_ATTR_TENSOR_REORDERING_MODE = 909
MUDNN_ATTR_TENSOR_CONSTANT_VALUE = 910
MUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC = 913
MUDNN_ATTR_TENSOR_RAGGED_OFFSET_MULTIPLIER = 914
MUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS = 1000
MUDNN_ATTR_VARIANT_PACK_DATA_POINTERS = 1001
MUDNN_ATTR_VARIANT_PACK_WORKSPACE = 1003
MUDNN_ATTR_KNOB_INFO_TYPE = 1200
MUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE = 1201
MUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE = 1202
MUDNN_ATTR_KNOB_INFO_STRIDE = 1203
MUDNN_ATTR_ENGINE_OPERATION_GRAPH = 1300
MUDNN_ATTR_ENGINE_GLOBAL_INDEX = 1301
MUDNN_ATTR_ENGINE_KNOB_INFO = 1302
MUDNN_ATTR_ENGINE_NUMERICAL_NOTE = 1303
MUDNN_ATTR_ENGINE_BEHAVIOR_NOTE = 1305
MUDNN_ATTR_MATMUL_COMP_TYPE = 1500
MUDNN_ATTR_MATMUL_PADDING_VALUE = 1503
MUDNN_ATTR_OPERATION_MATMUL_ADESC = 1520
MUDNN_ATTR_OPERATION_MATMUL_BDESC = 1521
MUDNN_ATTR_OPERATION_MATMUL_CDESC = 1522
MUDNN_ATTR_OPERATION_MATMUL_DESC = 1523
MUDNN_ATTR_REDUCTION_OPERATOR = 1600
MUDNN_ATTR_REDUCTION_COMP_TYPE = 1601
MUDNN_ATTR_REDUCTION_IS_DETERMINISTIC = 1602
MUDNN_ATTR_OPERATION_REDUCTION_XDESC = 1610
MUDNN_ATTR_OPERATION_REDUCTION_YDESC = 1611
MUDNN_ATTR_OPERATION_REDUCTION_DESC = 1612
MUDNN_ATTR_OPERATION_REDUCTION_GROUP_OFFSET_DESC = 1613
MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC = 1620
MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC = 1621
MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC = 1622
MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC = 1623
MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC = 1624
MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC = 1625
MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC = 1626
MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC = 1627
MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC = 1628
MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC = 1629
MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS = 1630
MUDNN_ATTR_RESAMPLE_MODE = 1700
MUDNN_ATTR_RESAMPLE_COMP_TYPE = 1701
MUDNN_ATTR_RESAMPLE_SPATIAL_DIMS = 1702
MUDNN_ATTR_RESAMPLE_POST_PADDINGS = 1703
MUDNN_ATTR_RESAMPLE_PRE_PADDINGS = 1704
MUDNN_ATTR_RESAMPLE_STRIDES = 1705
MUDNN_ATTR_RESAMPLE_WINDOW_DIMS = 1706
MUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC = 1710
MUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC = 1711
MUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC = 1712
MUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA = 1713
MUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA = 1714
MUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC = 1716
MUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC = 1720
MUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC = 1721
MUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC = 1722
MUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA = 1723
MUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA = 1724
MUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC = 1725
MUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC = 1726
MUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC = 1727
MUDNN_ATTR_OPERATION_SDPA_FWD_CU_SEQ_LEN_QDESC = 2820
MUDNN_ATTR_OPERATION_SDPA_FWD_CU_SEQ_LEN_KVDESC = 2821
MUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_CU_SEQ_LEN_KVDESC = 3008
MUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_CU_SEQ_LEN_QDESC = 3009
MUDNN_ATTR_OPERATION_ROPE_FWD_XDESC = 3400
MUDNN_ATTR_OPERATION_ROPE_FWD_YDESC = 3401
MUDNN_ATTR_OPERATION_ROPE_FWD_FREQSDESC = 3402
MUDNN_ATTR_OPERATION_ROPE_FWD_OUTPUT_SCALE = 3403
MUDNN_ATTR_OPERATION_ROPE_FWD_ROPE_DIM = 3404
MUDNN_ATTR_OPERATION_ROPE_BWD_DYDESC = 3410
MUDNN_ATTR_OPERATION_ROPE_BWD_DXDESC = 3411
MUDNN_ATTR_OPERATION_ROPE_BWD_FREQSDESC = 3412
MUDNN_ATTR_OPERATION_ROPE_BWD_OUTPUT_SCALE = 3413
MUDNN_ATTR_OPERATION_ROPE_BWD_ROPE_DIM = 3414
MUDNN_ATTR_OPERATION_CONCAT_AXIS = 1800
MUDNN_ATTR_OPERATION_CONCAT_INPUT_DESCS = 1801
MUDNN_ATTR_OPERATION_CONCAT_INPLACE_INDEX = 1802
MUDNN_ATTR_OPERATION_CONCAT_OUTPUT_DESC = 1803
MUDNN_ATTR_OPERATION_SLICE_XDESC = 3300
MUDNN_ATTR_OPERATION_SLICE_YDESC = 3301
MUDNN_ATTR_OPERATION_SLICE_START_INDICES = 3302
MUDNN_ATTR_OPERATION_SLICE_LIMIT_INDICES = 3303
MUDNN_ATTR_OPERATION_SLICE_STRIDES = 3304
MUDNN_ATTR_OPERATION_TRANSPOSE_XDESC = 3200
MUDNN_ATTR_OPERATION_TRANSPOSE_YDESC = 3201
MUDNN_ATTR_OPERATION_TRANSPOSE_PERMUTATION = 3202
MUDNN_ATTR_OPERATION_NORM_FWD_MODE = 2000
MUDNN_ATTR_OPERATION_NORM_FWD_PHASE = 2001
MUDNN_ATTR_OPERATION_NORM_FWD_XDESC = 2002
MUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC = 2003
MUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC = 2004
MUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC = 2005
MUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC = 2006
MUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC = 2007
MUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC = 2008
MUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC = 2009
MUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC = 2010
MUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC = 2011
MUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC = 2012
MUDNN_ATTR_OPERATION_NORM_FWD_YDESC = 2013
MUDNN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS = 2014
MUDNN_ATTR_OPERATION_NORM_BWD_MODE = 2100
MUDNN_ATTR_OPERATION_NORM_BWD_XDESC = 2101
MUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC = 2102
MUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC = 2103
MUDNN_ATTR_OPERATION_NORM_BWD_DYDESC = 2104
MUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC = 2105
MUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC = 2106
MUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC = 2107
MUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC = 2108
MUDNN_ATTR_OPERATION_NORM_BWD_DXDESC = 2109
MUDNN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS = 2110
MUDNN_ATTR_OPERATION_RESHAPE_XDESC = 2200
MUDNN_ATTR_OPERATION_RESHAPE_YDESC = 2201
MUDNN_ATTR_OPERATION_RESHAPE_MODE = 2202
MUDNN_ATTR_RNG_DISTRIBUTION = 2300
MUDNN_ATTR_RNG_NORMAL_DIST_MEAN = 2301
MUDNN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION = 2302
MUDNN_ATTR_RNG_UNIFORM_DIST_MAXIMUM = 2303
MUDNN_ATTR_RNG_UNIFORM_DIST_MINIMUM = 2304
MUDNN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY = 2305
MUDNN_ATTR_OPERATION_RNG_YDESC = 2310
MUDNN_ATTR_OPERATION_RNG_SEED = 2311
MUDNN_ATTR_OPERATION_RNG_DESC = 2312
MUDNN_ATTR_OPERATION_RNG_OFFSET_DESC = 2313
MUDNN_ATTR_OPERATION_SIGNAL_MODE = 1900
MUDNN_ATTR_OPERATION_SIGNAL_FLAGDESC = 1901
MUDNN_ATTR_OPERATION_SIGNAL_VALUE = 1902
MUDNN_ATTR_OPERATION_SIGNAL_XDESC = 1903
MUDNN_ATTR_OPERATION_SIGNAL_YDESC = 1904
MUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_CONTAINER_DESC = 1950
MUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_YDESC = 1951
MUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_SEQUENCE_DESC = 1952
MUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_PAGE_TABLE_DESC = 1953
MUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_XDESC = 2500
MUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_YDESC = 2501
MUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_SCALE_DESC = 2502
MUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_MATH_PREC = 2503
MUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_BLOCK_SIZE = 2504
MUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_XDESC = 2600
MUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_SCALE_DESC = 2601
MUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_YDESC = 2602
MUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_MATH_PREC = 2603
MUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE = 2604
MUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_NEG_SCALE = 2605
MUDNN_ATTR_OPERATION_SDPA_FWD_QDESC = 2800
MUDNN_ATTR_OPERATION_SDPA_FWD_KDESC = 2801
MUDNN_ATTR_OPERATION_SDPA_FWD_VDESC = 2802
MUDNN_ATTR_OPERATION_SDPA_FWD_ODESC = 2803
MUDNN_ATTR_OPERATION_SDPA_FWD_STATSDESC = 2804
MUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC = 2805
MUDNN_ATTR_OPERATION_SDPA_FWD_BLOCK_MASK_DESC = 2806
MUDNN_ATTR_OPERATION_SDPA_FWD_PAGE_TABLE_KDESC = 2807
MUDNN_ATTR_OPERATION_SDPA_FWD_PAGE_TABLE_VDESC = 2808
MUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_QDESC = 2809
MUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_KVDESC = 2810
MUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH = 2811
MUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH_INPUT_UID = 2812
MUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH_OUTPUT_UID = 2813
MUDNN_ATTR_OPERATION_SDPA_FWD_SOFTMAX_DESC = 2814
MUDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_SEED_DESC = 2815
MUDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_OFFSET_DESC = 2816
MUDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_RNG_DUMP_DESC = 2817
MUDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_PROBABILITY = 2818
MUDNN_ATTR_OPERATION_SDPA_FWD_UNFUSE_FMA = 2819
MUDNN_ATTR_OPERATION_SDPA_FWD_USE_PADDING_MASK = 9001
MUDNN_ATTR_OPERATION_SDPA_FWD_BIAS_DESC = 9002
MUDNN_ATTR_OPERATION_SDPA_FWD_CAUSAL_MASK = 9003
MUDNN_ATTR_OPERATION_SDPA_BWD_QDESC = 2851
MUDNN_ATTR_OPERATION_SDPA_BWD_KDESC = 2852
MUDNN_ATTR_OPERATION_SDPA_BWD_VDESC = 2853
MUDNN_ATTR_OPERATION_SDPA_BWD_ODESC = 2854
MUDNN_ATTR_OPERATION_SDPA_BWD_STATSDESC = 2855
MUDNN_ATTR_OPERATION_SDPA_BWD_SCALEDESC = 2856
MUDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_QDESC = 2857
MUDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_KVDESC = 2858
MUDNN_ATTR_OPERATION_SDPA_BWD_DQDESC = 2859
MUDNN_ATTR_OPERATION_SDPA_BWD_DKDESC = 2860
MUDNN_ATTR_OPERATION_SDPA_BWD_DVDESC = 2861
MUDNN_ATTR_OPERATION_SDPA_BWD_DODDESC = 2862
MUDNN_ATTR_OPERATION_SDPA_BWD_SINK_DESC = 2863
MUDNN_ATTR_OPERATION_SDPA_BWD_DSINK_DESC = 2864
MUDNN_ATTR_OPERATION_SDPA_BWD_MAX_TOTAL_SEQ_LEN_Q = 2865
MUDNN_ATTR_OPERATION_SDPA_BWD_MAX_TOTAL_SEQ_LEN_KV = 2866
MUDNN_ATTR_OPERATION_SDPA_BWD_SUBGRAPH = 2867
MUDNN_ATTR_OPERATION_SDPA_BWD_SUBGRAPH_INPUT_UID = 2868
MUDNN_ATTR_OPERATION_SDPA_BWD_SUBGRAPH_OUTPUT_UID = 2869
MUDNN_ATTR_OPERATION_SDPA_BWD_CAUSAL_MASK = 9004
MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MODE = 2900
MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MATH_PREC = 2901
MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC = 2902
MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_WEIGHT_DESC = 2903
MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_FIRST_TOKEN_OFFSET_DESC = 2904
MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_OUTPUT_DESC = 2905
MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC = 2906
MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_KS_DESC = 2907
MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOP_K = 2908
MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_MATH_PREC = 2951
MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_TOKEN_DESC = 2952
MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DWEIGHT_DESC = 2953
MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_FIRST_TOKEN_OFFSET_DESC = 2954
MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DOUTPUT_DESC = 2955

MUDNN_MOE_GROUPED_MATMUL_MODE_NONE = 0
MUDNN_MOE_GROUPED_MATMUL_MODE_GATHER = 1
MUDNN_MOE_GROUPED_MATMUL_MODE_SCATTER = 2


class BackendError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _installed_library_dir():
    package_file = Path(__file__).resolve()
    prefix = package_file.parents[2]
    lib_dir = prefix / "lib"
    if (lib_dir / "libmudnn.so").exists():
        return lib_dir
    return None


def _candidate_library_paths():
    env_lib = os.environ.get("MUDNN_LIB")
    if env_lib:
        yield str(Path(env_lib) / "libmudnn.so")
    installed_lib = _installed_library_dir()
    if installed_lib is not None:
        yield str(installed_lib / "libmudnn.so")
    found = ctypes.util.find_library("mudnn")
    if found:
        yield found
    yield "libmudnn.so"


def _rpath_anchor(lib_dir: Path):
    """Build a tiny global object whose RPATH is ``lib_dir``.

    ``libmudnn.so`` has no RUNPATH, and its siblings are ``-Wl,-z,now`` with
    cycles (tensor and reduce, ops and xmma). Opening one sibling first fails
    before ld.so can see the other side of the cycle. Opening this anchor makes
    ld.so map the whole graph, then bind it.
    """
    gcc = shutil.which("gcc")
    if gcc is None or not (lib_dir / "libmudnn.so").is_file():
        return None
    digest = hashlib.sha256(str(lib_dir.resolve()).encode()).hexdigest()[:16]
    cache_dir = Path.home() / ".cache" / "mudnn-frontend"
    cache = cache_dir / f"rpath-anchor-{digest}.so"
    src = cache_dir / f"rpath-anchor-{digest}.c"
    tmp = cache_dir / f"rpath-anchor-{digest}.so.tmp"
    try:
        if cache.is_file() and cache.stat().st_size > 0:
            return cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        src.write_text(
            "extern const char* mudnnGetErrorString(int status);\n"
            "void mudnn_frontend_anchor(void) { (void)mudnnGetErrorString; }\n"
        )
        subprocess.run(
            [
                gcc,
                "-shared",
                "-fPIC",
                "-o",
                str(tmp),
                str(src),
                "-Wl,--disable-new-dtags",
                f"-Wl,-rpath,{lib_dir}",
                "-Wl,--no-as-needed",
                f"-L{lib_dir}",
                "-lmudnn",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        os.replace(tmp, cache)
    except (OSError, subprocess.CalledProcessError):
        return None
    finally:
        src.unlink(missing_ok=True)
        tmp.unlink(missing_ok=True)
    return cache if cache.is_file() else None


def _open_library(path: str):
    try:
        return [], ctypes.CDLL(path)
    except OSError as direct_error:
        lib_path = Path(path)
        anchor = _rpath_anchor(lib_path.parent) if lib_path.is_absolute() else None
        if anchor is None:
            raise
        try:
            handle = ctypes.CDLL(str(anchor), mode=ctypes.RTLD_GLOBAL)
            return [handle], ctypes.CDLL(path)
        except OSError:
            raise direct_error from None


class BackendLibrary:
    def __init__(self):
        errors = []
        for path in _candidate_library_paths():
            try:
                self._dependency_handles, self.lib = _open_library(path)
                break
            except OSError as exc:
                errors.append(f"{path}: {exc}")
        else:
            raise BackendError("failed to load libmudnn.so: " + "; ".join(errors))
        self._configure()

    def _configure(self):
        self.lib.mudnnGetErrorString.argtypes = [ctypes.c_int]
        self.lib.mudnnGetErrorString.restype = ctypes.c_char_p

        self.lib.mudnnCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.mudnnCreate.restype = ctypes.c_int
        self.lib.mudnnDestroy.argtypes = [ctypes.c_void_p]
        self.lib.mudnnDestroy.restype = ctypes.c_int

        self.lib.mudnnBackendCreateDescriptor.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.mudnnBackendCreateDescriptor.restype = ctypes.c_int
        self.lib.mudnnBackendDestroyDescriptor.argtypes = [ctypes.c_void_p]
        self.lib.mudnnBackendDestroyDescriptor.restype = ctypes.c_int
        self.lib.mudnnBackendFinalize.argtypes = [ctypes.c_void_p]
        self.lib.mudnnBackendFinalize.restype = ctypes.c_int
        self.lib.mudnnBackendSetAttribute.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_longlong,
            ctypes.c_void_p,
        ]
        self.lib.mudnnBackendSetAttribute.restype = ctypes.c_int
        self.lib.mudnnBackendGetAttribute.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_longlong,
            ctypes.POINTER(ctypes.c_longlong),
            ctypes.c_void_p,
        ]
        self.lib.mudnnBackendGetAttribute.restype = ctypes.c_int
        self.lib.mudnnBackendExecute.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.mudnnBackendExecute.restype = ctypes.c_int

        # Direct subquadratic entrypoints. Pointers stay c_void_p so a raw
        # device address round-trips instead of being truncated to a 32-bit int.
        # Bind them only when this libmudnn exports them so older backends still load.
        _causal_forward = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        _causal_backward = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        for name, argtypes in (
            ("mudnnCausalConv1dForward", _causal_forward),
            ("mudnnCausalConv1dNwhForward", _causal_forward),
            ("mudnnCausalConv1dBackward", _causal_backward),
            ("mudnnCausalConv1dNwhBackward", _causal_backward),
        ):
            symbol = getattr(self.lib, name, None)
            if symbol is None:
                continue
            symbol.argtypes = argtypes
            symbol.restype = ctypes.c_int

        # B2B entrypoints take a raw MUSA stream, not a graph handle.
        forward = getattr(self.lib, "mudnnB2BCausalConv1dForward", None)
        if forward is not None:
            forward.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            forward.restype = ctypes.c_int
        backward = getattr(self.lib, "mudnnB2BCausalConv1dBackward", None)
        if backward is not None:
            backward.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            backward.restype = ctypes.c_int

    def check(self, status: int, message: str):
        if status == MUDNN_STATUS_SUCCESS:
            return
        raw = self.lib.mudnnGetErrorString(status)
        error = raw.decode("utf-8", errors="replace") if raw else str(status)
        raise BackendError(f"{message}: {error} ({status})", status=status)


_LIBRARY = None


def get_library() -> BackendLibrary:
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = BackendLibrary()
    return _LIBRARY


class Handle:
    def __init__(self, lib: BackendLibrary | None = None):
        self.lib = lib or get_library()
        self.ptr = ctypes.c_void_p()
        self.lib.check(self.lib.lib.mudnnCreate(ctypes.byref(self.ptr)), "mudnnCreate")

    def close(self):
        if self.ptr.value:
            self.lib.check(self.lib.lib.mudnnDestroy(self.ptr), "mudnnDestroy")
            self.ptr = ctypes.c_void_p()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class Descriptor:
    def __init__(
        self,
        descriptor_type: int | None = None,
        *,
        ptr: ctypes.c_void_p | None = None,
        owned: bool = True,
        lib: BackendLibrary | None = None,
    ):
        self.lib = lib or get_library()
        self.owned = owned
        if ptr is not None:
            self.ptr = ctypes.c_void_p(
                ptr.value if isinstance(ptr, ctypes.c_void_p) else ptr
            )
        else:
            self.ptr = ctypes.c_void_p()
            self.lib.check(
                self.lib.lib.mudnnBackendCreateDescriptor(
                    int(descriptor_type), ctypes.byref(self.ptr)
                ),
                "mudnnBackendCreateDescriptor",
            )

    def close(self):
        if self.owned and self.ptr.value:
            self.lib.check(
                self.lib.lib.mudnnBackendDestroyDescriptor(self.ptr),
                "mudnnBackendDestroyDescriptor",
            )
            self.ptr = ctypes.c_void_p()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def finalize(self):
        self.lib.check(
            self.lib.lib.mudnnBackendFinalize(self.ptr), "mudnnBackendFinalize"
        )

    def set_int64(self, attr: int, values):
        array = (ctypes.c_longlong * len(values))(*[int(v) for v in values])
        self._set(attr, MUDNN_TYPE_INT64, len(values), array)

    def set_double(self, attr: int, value: float):
        data = ctypes.c_double(float(value))
        self._set(attr, MUDNN_TYPE_DOUBLE, 1, ctypes.byref(data))

    def set_bool(self, attr: int, value: bool):
        data = ctypes.c_bool(bool(value))
        self._set(attr, MUDNN_TYPE_BOOLEAN, 1, ctypes.byref(data))

    def set_enum(self, attr: int, attr_type: int, value: int):
        data = ctypes.c_int(int(value))
        self._set(attr, attr_type, 1, ctypes.byref(data))

    def set_handle(self, attr: int, handle: Handle | ctypes.c_void_p):
        ptr = handle.ptr if isinstance(handle, Handle) else handle
        data = ctypes.c_void_p(ptr.value)
        self._set(attr, MUDNN_TYPE_HANDLE, 1, ctypes.byref(data))

    def set_descriptor(self, attr: int, descriptor: "Descriptor"):
        data = ctypes.c_void_p(descriptor.ptr.value)
        self._set(attr, MUDNN_TYPE_BACKEND_DESCRIPTOR, 1, ctypes.byref(data))

    def set_descriptors(self, attr: int, descriptors):
        array = (ctypes.c_void_p * len(descriptors))(
            *[ctypes.c_void_p(desc.ptr.value) for desc in descriptors]
        )
        self._set(attr, MUDNN_TYPE_BACKEND_DESCRIPTOR, len(descriptors), array)

    def set_void_ptrs(self, attr: int, pointers):
        array = (ctypes.c_void_p * len(pointers))(
            *[ctypes.c_void_p(int(pointer)) for pointer in pointers]
        )
        self._set(attr, MUDNN_TYPE_VOID_PTR, len(pointers), array)

    def _set(self, attr: int, attr_type: int, count: int, data):
        self.lib.check(
            self.lib.lib.mudnnBackendSetAttribute(
                self.ptr,
                int(attr),
                int(attr_type),
                int(count),
                ctypes.cast(data, ctypes.c_void_p),
            ),
            "mudnnBackendSetAttribute",
        )

    def get_int64(self, attr: int, requested: int = 1):
        count = ctypes.c_longlong()
        values = (ctypes.c_longlong * requested)()
        self.lib.check(
            self.lib.lib.mudnnBackendGetAttribute(
                self.ptr,
                int(attr),
                MUDNN_TYPE_INT64,
                int(requested),
                ctypes.byref(count),
                values,
            ),
            "mudnnBackendGetAttribute",
        )
        return tuple(int(values[i]) for i in range(int(count.value)))

    def get_descriptors(self, attr: int, requested: int = 1):
        count = ctypes.c_longlong()
        values = (ctypes.c_void_p * requested)()
        self.lib.check(
            self.lib.lib.mudnnBackendGetAttribute(
                self.ptr,
                int(attr),
                MUDNN_TYPE_BACKEND_DESCRIPTOR,
                int(requested),
                ctypes.byref(count),
                values,
            ),
            "mudnnBackendGetAttribute",
        )
        return [
            Descriptor(ptr=ctypes.c_void_p(values[i]), owned=False, lib=self.lib)
            for i in range(int(count.value))
        ]

    def get_descriptor_count(self, attr: int) -> int:
        count = ctypes.c_longlong()
        self.lib.check(
            self.lib.lib.mudnnBackendGetAttribute(
                self.ptr,
                int(attr),
                MUDNN_TYPE_BACKEND_DESCRIPTOR,
                0,
                ctypes.byref(count),
                None,
            ),
            "mudnnBackendGetAttribute",
        )
        return int(count.value)

    def get_enums(self, attr: int, attr_type: int):
        count = ctypes.c_longlong()
        self.lib.check(
            self.lib.lib.mudnnBackendGetAttribute(
                self.ptr,
                int(attr),
                int(attr_type),
                0,
                ctypes.byref(count),
                None,
            ),
            "mudnnBackendGetAttribute",
        )
        values = (ctypes.c_int * int(count.value))()
        if count.value:
            self.lib.check(
                self.lib.lib.mudnnBackendGetAttribute(
                    self.ptr,
                    int(attr),
                    int(attr_type),
                    int(count.value),
                    ctypes.byref(count),
                    values,
                ),
                "mudnnBackendGetAttribute",
            )
        return tuple(int(values[i]) for i in range(int(count.value)))


def create_tensor_desc(
    tensor, *, is_virtual: bool, ragged_offset_desc: Descriptor | None = None
) -> Descriptor:
    desc = Descriptor(MUDNN_BACKEND_TENSOR_DESCRIPTOR)
    desc.set_enum(MUDNN_ATTR_TENSOR_DATA_TYPE, MUDNN_TYPE_DATA_TYPE, tensor.data_type)
    desc.set_int64(MUDNN_ATTR_TENSOR_DIMENSIONS, tensor.dims)
    desc.set_int64(MUDNN_ATTR_TENSOR_STRIDES, tensor.strides)
    desc.set_int64(MUDNN_ATTR_TENSOR_UNIQUE_ID, [tensor.uid])
    desc.set_int64(MUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, [getattr(tensor, "alignment", 16)])
    desc.set_int64(MUDNN_ATTR_TENSOR_VECTOR_COUNT, [getattr(tensor, "vector_count", 1)])
    desc.set_int64(
        MUDNN_ATTR_TENSOR_VECTORIZED_DIMENSION,
        [getattr(tensor, "vector_dimension", -1)],
    )
    desc.set_bool(MUDNN_ATTR_TENSOR_IS_VIRTUAL, is_virtual)
    desc.set_bool(MUDNN_ATTR_TENSOR_IS_BY_VALUE, getattr(tensor, "is_by_value", False))
    reordering_type = getattr(tensor, "reordering_type", None)
    if reordering_type is not None:
        desc.set_enum(
            MUDNN_ATTR_TENSOR_REORDERING_MODE,
            MUDNN_TYPE_TENSOR_REORDERING_MODE,
            reordering_type,
        )
    if ragged_offset_desc is not None:
        desc.set_descriptor(MUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC, ragged_offset_desc)
    desc.finalize()
    return desc


def create_variant_pack(
    uid_to_pointer: dict[int, int], *, workspace_ptr=None
) -> Descriptor:
    uids = tuple(int(uid) for uid in uid_to_pointer.keys())
    pointers = tuple(int(pointer) for pointer in uid_to_pointer.values())
    desc = Descriptor(MUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR)
    desc.set_int64(MUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, uids)
    desc.set_void_ptrs(MUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, pointers)
    if workspace_ptr is not None:
        desc.set_void_ptrs(MUDNN_ATTR_VARIANT_PACK_WORKSPACE, [int(workspace_ptr)])
    desc.finalize()
    return desc


def execute(handle: Handle, plan: Descriptor, variant_pack: Descriptor):
    lib = get_library()
    lib.check(
        lib.lib.mudnnBackendExecute(handle.ptr, plan.ptr, variant_pack.ptr),
        "mudnnBackendExecute",
    )
