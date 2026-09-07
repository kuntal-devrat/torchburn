# FX graph -> structured payload parser (REQ-001 / REQ-002).
#
# Split from the monolithic ``_parser.py``; this package re-exports the
# previous flat-module surface so ``from torchburn._parser import
# parse_graph, payload_json`` keeps working unchanged.

from .graph_walker import (
    _arg_count_ok,
    _dtype_str,
    _extract_kwargs,
    _ref,
    _ref_kwargs,
    _sanitize_nonfinite,
    parse_graph,
    payload_json,
)
from .op_registry import (
    _ATEN_TO_OP,
    _FUNCTION_TO_OP,
    _METHOD_TO_OP,
    _SUPPORTED_OPS,
    _TUPLE_OUTPUT_OPS,
    _getitem_as_select,
    _is_tuple_source,
    _target_key,
    canonical_op,
)

__all__ = [
    "parse_graph",
    "payload_json",
    "canonical_op",
    "_FUNCTION_TO_OP",
    "_ATEN_TO_OP",
    "_METHOD_TO_OP",
    "_SUPPORTED_OPS",
    "_TUPLE_OUTPUT_OPS",
    "_target_key",
    "_is_tuple_source",
    "_getitem_as_select",
    "_dtype_str",
    "_ref_kwargs",
    "_extract_kwargs",
    "_arg_count_ok",
    "_ref",
    "_sanitize_nonfinite",
]
