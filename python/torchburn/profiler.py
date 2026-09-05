"""TorchBurn profiling and diagnostics API.

Provides:
- ``profile()`` context manager for timing compiled vs eager execution
- ``coverage_report()`` showing operator coverage breakdown
- ``memory_stats()`` showing allocation statistics
"""

from __future__ import annotations

import contextlib
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Generator

from . import _torchburn as _native


@dataclass
class ProfileResult:
    """Result from a profiling session."""
    wall_time_ms: float = 0.0
    num_nodes: int = 0
    num_supported: int = 0
    num_unsupported: int = 0
    num_fallbacks: int = 0
    engine: str = ""
    graph_signature: str = ""

    @property
    def native_ratio(self) -> float:
        """Fraction of nodes executed natively."""
        total = self.num_supported + self.num_unsupported
        return self.num_supported / total if total > 0 else 0.0

    def summary(self) -> str:
        lines = [
            f"TorchBurn Profile",
            f"  Engine:      {self.engine}",
            f"  Wall time:   {self.wall_time_ms:.2f} ms",
            f"  Nodes:       {self.num_nodes} total ({self.num_supported} native, "
            f"{self.num_unsupported} eager fallback)",
            f"  Native %:    {self.native_ratio*100:.1f}%",
            f"  Fallbacks:   {self.num_fallbacks}",
        ]
        if self.graph_signature:
            lines.append(f"  Signature:   {self.graph_signature[:32]}...")
        return "\n".join(lines)


@dataclass
class _Stats:
    """Global profiling statistics."""
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _calls: int = 0
    _total_ms: float = 0.0
    _total_nodes: int = 0
    _total_native: int = 0
    _total_fallbacks: int = 0

    def record(self, result: ProfileResult) -> None:
        with self._lock:
            self._calls += 1
            self._total_ms += result.wall_time_ms
            self._total_nodes += result.num_nodes
            self._total_native += result.num_supported
            self._total_fallbacks += result.num_fallbacks

    def reset(self) -> None:
        with self._lock:
            self._calls = 0
            self._total_ms = 0.0
            self._total_nodes = 0
            self._total_native = 0
            self._total_fallbacks = 0

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "calls": self._calls,
                "total_ms": self._total_ms,
                "avg_ms": self._total_ms / self._calls if self._calls > 0 else 0.0,
                "total_nodes": self._total_nodes,
                "total_native": self._total_native,
                "total_fallbacks": self._total_fallbacks,
                "native_ratio": (
                    self._total_native / self._total_nodes
                    if self._total_nodes > 0 else 0.0
                ),
            }


_STATS = _Stats()

# Thread-local current profile for _interpreter to populate
_thread_local = threading.local()


def _set_current_profile(result: ProfileResult | None) -> None:
    _thread_local.current = result


def _get_current_profile() -> ProfileResult | None:
    return getattr(_thread_local, "current", None)


@contextlib.contextmanager
def profile() -> Generator[ProfileResult, None, None]:
    """Context manager for profiling TorchBurn execution.

    Usage::

        with torchburn.profiler.profile() as result:
            output = compiled_model(input_tensor)
        print(result.summary())

    The result is populated after the context manager exits.
    """
    result = ProfileResult()
    result.engine = _native.active_engine()
    _set_current_profile(result)
    start = time.perf_counter()
    try:
        yield result
    finally:
        elapsed = time.perf_counter() - start
        result.wall_time_ms = elapsed * 1000.0
        _set_current_profile(None)
        _STATS.record(result)


def coverage_report() -> dict[str, Any]:
    """Get operator coverage statistics from cached graph compilations.

    Returns a dict with:
        - supported_targets: list of native targets
        - target_count: number of supported targets
        - engine: active engine name
    """
    targets = _native.supported_targets()
    return {
        "supported_targets": sorted(targets),
        "target_count": len(targets),
        "engine": _native.active_engine(),
    }


def memory_stats() -> dict[str, Any]:
    """Get accumulated profiling statistics since last reset.

    Returns:
        calls: number of profiled executions
        total_ms: total wall time across all profiled calls
        avg_ms: average wall time per call
        total_nodes: total graph nodes across all calls
        total_native: total native nodes across all calls
        total_fallbacks: total fallback nodes across all calls
        native_ratio: fraction of nodes executed natively
    """
    return _STATS.stats()


def reset_stats() -> None:
    """Reset accumulated profiling statistics."""
    _STATS.reset()


def supported_ops() -> list[str]:
    """Return a sorted list of all operators the native engine supports."""
    return sorted(_native.supported_targets())


def active_engine() -> str:
    """Return the name of the active execution engine."""
    return _native.active_engine()


def memory_pool_stats() -> dict[str, Any]:
    """Return memory pool diagnostics including allocation count, hit rate, and cached buffers."""
    try:
        return _native.memory_pool_stats()
    except AttributeError:
        return {
            "alloc_count": 0,
            "hit_count": 0,
            "recycle_count": 0,
            "cached_buffers": 0,
            "cached_words": 0,
            "hit_rate": 0.0,
        }


def clear_memory_pool() -> None:
    """Clear all cached buffers from the thread memory pool."""
    try:
        _native.clear_memory_pool()
    except AttributeError:
        pass


def trace(model, example_inputs, **kwargs) -> dict[str, Any]:
    """Generate a Chrome trace JSON for a model (stub for ROADMAP 15.5).

    Returns a dict with `traceEvents` that can be loaded in `chrome://tracing`
    or `perfetto`. Currently profiles the native execution via `profile()`.
    """
    import torch

    if not isinstance(example_inputs, (list, tuple)):
        example_inputs = [example_inputs]
    # Compile and profile
    compiled = torch.compile(model, backend="torchburn", **kwargs)
    with profile() as result:
        # Warmup + timed run
        _ = compiled(*example_inputs)
    # Build minimal Chrome trace format
    return {
        "traceEvents": [
            {
                "name": "torchburn::execute",
                "cat": "torchburn",
                "ph": "X",
                "ts": 0,
                "dur": result.wall_time_ms * 1000,
                "pid": 1,
                "tid": 1,
                "args": {
                    "engine": result.engine,
                    "nodes": result.num_nodes,
                    "native": result.num_supported,
                    "fallback": result.num_unsupported,
                },
            }
        ],
        "displayTimeUnit": "ms",
        "metadata": {
            "engine": result.engine,
            "wall_time_ms": result.wall_time_ms,
        },
    }


def op_coverage(model, example_inputs, **kwargs) -> dict[str, Any]:
    """Analyze operator coverage for a specific model without executing.

    Returns:
        total_nodes: total FX nodes
        native_nodes: count of native ops
        fallback_nodes: count of fallback ops
        native_ratio: fraction native
        unsupported_ops: list of fallback op targets
        engine: active engine
    """
    import torch
    from torch.fx.experimental.proxy_tensor import make_fx
    from ._parser import parse_graph

    if not isinstance(example_inputs, (list, tuple)):
        example_inputs = [example_inputs]
    # Try to get FX graph
    try:
        gm = make_fx(model)(*example_inputs)
    except Exception:
        # Fallback to torch.compile's dynamo capture via a dummy
        gm = model
        return {
            "total_nodes": 0,
            "native_nodes": 0,
            "fallback_nodes": 0,
            "native_ratio": 0.0,
            "unsupported_ops": [],
            "engine": active_engine(),
            "error": "could not capture graph via make_fx; use torch.compile for full trace",
        }
    try:
        plan, _ = parse_graph(gm, list(example_inputs))
        total = len([n for n in plan["nodes"] if n["op"] in ("supported", "unsupported")])
        native = len([n for n in plan["nodes"] if n["op"] == "supported"])
        fallback = len([n for n in plan["nodes"] if n["op"] == "unsupported"])
        unsupported = [n["fx_target"] for n in plan["nodes"] if n["op"] == "unsupported"]
        return {
            "total_nodes": total,
            "native_nodes": native,
            "fallback_nodes": fallback,
            "native_ratio": native / total if total else 0.0,
            "unsupported_ops": unsupported,
            "engine": active_engine(),
        }
    except Exception as exc:
        return {
            "total_nodes": 0,
            "native_nodes": 0,
            "fallback_nodes": 0,
            "native_ratio": 0.0,
            "unsupported_ops": [],
            "engine": active_engine(),
            "error": str(exc),
        }


class GraphVisualization:
    """Interactive visual representation of a TorchBurn compiled execution graph."""

    def __init__(self, plan: dict[str, Any], engine: str, model_name: str = "Model") -> None:
        self.plan = plan
        self.engine = engine
        self.model_name = model_name
        self.nodes = plan.get("nodes", [])
        self.inputs = plan.get("inputs", [])
        self.outputs = plan.get("outputs", [])

        self.native_nodes = [n for n in self.nodes if n.get("op") == "supported"]
        self.fallback_nodes = [n for n in self.nodes if n.get("op") == "unsupported"]
        self.io_nodes = [n for n in self.nodes if n.get("op") in ("placeholder", "output", "get_attr")]

        total = len(self.native_nodes) + len(self.fallback_nodes)
        self.total_compute_nodes = total
        self.native_ratio = len(self.native_nodes) / total if total > 0 else 1.0

    def summary(self) -> str:
        """Return a formatted terminal ASCII dashboard."""
        pct = self.native_ratio * 100.0
        bar_len = int(pct // 5)
        status_bar = f"[{'=' * bar_len}{' ' * (20 - bar_len)}] {pct:.1f}%"
        lines = [
            "+" + "=" * 78 + "+",
            f"| TorchBurn Graph Inspector: {self.model_name:<48} |",
            f"| Engine: {self.engine:<20} | Compute Nodes: {self.total_compute_nodes:<5} | Native: {len(self.native_nodes)} ({pct:.1f}%) |",
            f"| Acceleration Coverage: {status_bar:<50} |",
            "+" + "=" * 5 + "+" + "=" * 15 + "+" + "=" * 18 + "+" + "=" * 21 + "+" + "=" * 15 + "+",
            f"| {'ID':<3} | {'Status':<13} | {'Op Target':<16} | {'FX Target':<19} | {'Inputs':<13} |",
            "+" + "=" * 5 + "+" + "=" * 15 + "+" + "=" * 18 + "+" + "=" * 21 + "+" + "=" * 15 + "+",
        ]
        for n in self.nodes:
            nid = str(n.get("id", "-"))
            op_kind = n.get("op", "")
            if op_kind == "supported":
                status = "[NATIVE]"
            elif op_kind == "unsupported":
                status = "[FALLBACK]"
            elif op_kind == "placeholder":
                status = "[INPUT]"
            elif op_kind == "output":
                status = "[OUTPUT]"
            else:
                status = f"[{op_kind.upper()}]"

            target = str(n.get("target") or n.get("name") or "-")[:16]
            fx_target = str(n.get("fx_target") or "-")[:19]

            args = n.get("args", [])
            in_ids = []
            for a in args:
                if isinstance(a, dict) and "id" in a:
                    in_ids.append(f"#{a['id']}")
                elif isinstance(a, dict) and a.get("kind") == "slot":
                    in_ids.append(f"s{a.get('index')}")
            inputs_str = ", ".join(in_ids)[:13] if in_ids else "-"

            lines.append(
                f"| {nid:<3} | {status:<13} | {target:<16} | {fx_target:<19} | {inputs_str:<13} |"
            )

        lines.append("+" + "=" * 5 + "+" + "=" * 15 + "+" + "=" * 18 + "+" + "=" * 21 + "+" + "=" * 15 + "+")
        if self.fallback_nodes:
            lines.append(f"  * {len(self.fallback_nodes)} node(s) executing via PyTorch eager fallback.")
        else:
            lines.append("  * 100% of compute nodes accelerated natively via TorchBurn.")
        return "\n".join(lines)

    def to_mermaid(self) -> str:
        """Generate a Mermaid.js flowchart representation."""
        lines = [
            "flowchart TD",
            "    classDef native fill:#059669,stroke:#10b981,stroke-width:2px,color:#ffffff,font-family:sans-serif;",
            "    classDef fallback fill:#d97706,stroke:#f59e0b,stroke-width:2px,color:#ffffff,font-family:sans-serif;",
            "    classDef io fill:#1e40af,stroke:#3b82f6,stroke-width:2px,color:#ffffff,font-family:sans-serif;",
        ]
        for n in self.nodes:
            nid = n.get("id", 0)
            op = n.get("op", "")
            target = n.get("target") or n.get("name") or "op"
            fx_target = n.get("fx_target") or ""
            label = f"{target}"
            if fx_target and fx_target != target:
                label += f"<br/><small>{fx_target}</small>"

            if op == "supported":
                cls = "native"
                badge = "(Native)"
            elif op == "unsupported":
                cls = "fallback"
                badge = "(Fallback)"
            else:
                cls = "io"
                badge = f"({op})"

            lines.append(f'    node_{nid}["{label}<br/><b>{badge}</b>"]:::{cls}')

        for n in self.nodes:
            nid = n.get("id", 0)
            for a in n.get("args", []):
                if isinstance(a, dict) and "id" in a:
                    lines.append(f"    node_{a['id']} --> node_{nid}")

        return "\n".join(lines)

    def to_html(self, title: str | None = None) -> str:
        """Generate a rich, standalone interactive HTML visualization report."""
        page_title = title or f"TorchBurn Execution Graph — {self.model_name}"
        mermaid_code = self.to_mermaid()
        pct = self.native_ratio * 100.0

        rows = []
        for n in self.nodes:
            nid = n.get("id", "-")
            op = n.get("op", "")
            target = n.get("target") or n.get("name") or "-"
            fx_target = n.get("fx_target") or "-"
            if op == "supported":
                badge = '<span class="badge badge-native">Native Rust</span>'
                row_cls = "row-native"
            elif op == "unsupported":
                badge = '<span class="badge badge-fallback">Eager Fallback</span>'
                row_cls = "row-fallback"
            else:
                badge = f'<span class="badge badge-io">{op}</span>'
                row_cls = "row-io"

            args = n.get("args", [])
            in_links = []
            for a in args:
                if isinstance(a, dict) and "id" in a:
                    in_links.append(f"#{a['id']}")
            inputs_str = ", ".join(in_links) if in_links else "-"

            rows.append(f"""
            <tr class="{row_cls}">
                <td><code>#{nid}</code></td>
                <td>{badge}</td>
                <td><strong>{target}</strong></td>
                <td><code>{fx_target}</code></td>
                <td>{inputs_str}</td>
            </tr>
            """)

        table_rows = "\n".join(rows)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{page_title}</title>
    <style>
        :root {{
            --bg: #090d16;
            --card-bg: #111827;
            --border: #1f2937;
            --text: #f3f4f6;
            --text-dim: #9ca3af;
            --emerald: #10b981;
            --amber: #f59e0b;
            --blue: #3b82f6;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
            padding: 24px;
            line-height: 1.5;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border);
        }}
        .header h1 {{ font-size: 24px; font-weight: 700; color: #fff; }}
        .header .subtitle {{ color: var(--text-dim); font-size: 14px; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 18px;
        }}
        .card-label {{ font-size: 12px; font-weight: 600; text-transform: uppercase; color: var(--text-dim); margin-bottom: 6px; }}
        .card-val {{ font-size: 28px; font-weight: 700; color: #fff; }}
        .card-val.emerald {{ color: var(--emerald); }}
        .card-val.amber {{ color: var(--amber); }}
        .card-val.blue {{ color: var(--blue); }}
        .progress-bar {{
            background: #1f2937;
            border-radius: 9999px;
            height: 8px;
            overflow: hidden;
            margin-top: 8px;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #10b981, #059669);
            width: {pct:.1f}%;
        }}
        .graph-container {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            overflow-x: auto;
            text-align: center;
        }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
        }}
        .badge-native {{ background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #059669; }}
        .badge-fallback {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #d97706; }}
        .badge-io {{ background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid #2563eb; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        th, td {{
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            background: #161f30;
            color: var(--text-dim);
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
        }}
        code {{
            background: #1f2937;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 12px;
        }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <script>mermaid.initialize({{ startOnLoad: true, theme: 'dark' }});</script>
</head>
<body>
    <div class="header">
        <div>
            <h1>TorchBurn Execution Graph</h1>
            <div class="subtitle">{self.model_name} • Hardware Engine: {self.engine}</div>
        </div>
        <div>
            <span class="badge {'badge-native' if pct >= 90 else 'badge-fallback'}" style="font-size: 14px; padding: 6px 12px;">
                {pct:.1f}% Native Acceleration
            </span>
        </div>
    </div>

    <div class="stats-grid">
        <div class="card">
            <div class="card-label">Hardware Acceleration</div>
            <div class="card-val emerald">{pct:.1f}%</div>
            <div class="progress-bar"><div class="progress-fill"></div></div>
        </div>
        <div class="card">
            <div class="card-label">Compute Nodes</div>
            <div class="card-val blue">{self.total_compute_nodes}</div>
            <div style="font-size: 12px; color: var(--text-dim); margin-top: 6px;">Total operators planned</div>
        </div>
        <div class="card">
            <div class="card-label">Native Fast-Path Nodes</div>
            <div class="card-val emerald">{len(self.native_nodes)}</div>
            <div style="font-size: 12px; color: var(--text-dim); margin-top: 6px;">Zero-overhead Rust kernels</div>
        </div>
        <div class="card">
            <div class="card-label">Eager Fallbacks</div>
            <div class="card-val {'amber' if self.fallback_nodes else 'emerald'}">{len(self.fallback_nodes)}</div>
            <div style="font-size: 12px; color: var(--text-dim); margin-top: 6px;">PyTorch CPU fallback nodes</div>
        </div>
    </div>

    <div class="card graph-container">
        <h2 style="font-size: 16px; margin-bottom: 16px; text-align: left;">Execution DAG</h2>
        <div class="mermaid">
{mermaid_code}
        </div>
    </div>

    <div class="card" style="padding: 0; overflow: hidden;">
        <h2 style="font-size: 16px; padding: 18px; border-bottom: 1px solid var(--border);">Operator Node Registry</h2>
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Status</th>
                    <th>Operator Target</th>
                    <th>FX Target</th>
                    <th>Inputs</th>
                </tr>
            </thead>
            <tbody>
{table_rows}
            </tbody>
        </table>
    </div>
</body>
</html>"""

    def save(self, filepath: str) -> None:
        """Save the HTML report to a local file."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_html())

    def _repr_html_(self) -> str:
        """Jupyter Notebook / IPython rich HTML display representation."""
        return self.to_html()


def visualize(
    model: Any,
    example_inputs: Any,
    output_html: str | None = None,
    print_summary: bool = True,
    **kwargs: Any,
) -> GraphVisualization:
    """Analyze and visually inspect a model's compiled execution graph.

    Args:
        model: PyTorch model or GraphModule to compile and inspect.
        example_inputs: Sample input tensor(s).
        output_html: Optional file path to write the standalone interactive HTML report.
        print_summary: Whether to print the formatted ASCII summary to terminal.

    Returns:
        GraphVisualization instance with ``.summary()``, ``.to_mermaid()``, and ``.to_html()``.
    """
    from torch.fx.experimental.proxy_tensor import make_fx
    from ._parser import parse_graph

    if not isinstance(example_inputs, (list, tuple)):
        example_inputs = [example_inputs]

    model_name = getattr(model, "__class__", type(model)).__name__

    try:
        gm = make_fx(model)(*example_inputs)
    except Exception:
        gm = model

    plan, _ = parse_graph(gm, list(example_inputs))
    viz = GraphVisualization(plan, engine=active_engine(), model_name=model_name)

    if print_summary:
        print(viz.summary())

    if output_html:
        viz.save(output_html)

    return viz

