"""DAG builder + linear scheduler.

For the MVP we run stages strictly in declared order (source -> stages -> sinks)
with async fan-out to sinks. A real DAG topological executor lives in `runner`.
"""
from __future__ import annotations

from collections import defaultdict, deque

from inferops.core.errors import PipelineError
from inferops.pipeline.spec import PipelineSpec


def topo_sort(node_names: list[str], edges: list[tuple[str, str]]) -> list[str]:
    indeg: dict[str, int] = defaultdict(int)
    adj: dict[str, list[str]] = defaultdict(list)
    nodes = set(node_names)
    for a, b in edges:
        adj[a].append(b)
        indeg[b] += 1
    q = deque([n for n in node_names if indeg[n] == 0])
    out: list[str] = []
    while q:
        n = q.popleft()
        out.append(n)
        for m in adj[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                q.append(m)
    if len(out) != len(nodes):
        raise PipelineError("pipeline graph has a cycle or disconnected nodes")
    return out


def linear_order(spec: PipelineSpec) -> list[str]:
    names: list[str] = []
    if spec.source is not None:
        names.append(spec.source.name)
    names.extend(s.name for s in spec.stages)
    if not spec.edges:
        return names
    edges = [(a, b) for a, b in spec.edges]
    return topo_sort(names, edges)
