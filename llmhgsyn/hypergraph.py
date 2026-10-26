from __future__ import annotations

from collections import defaultdict
import hashlib
from typing import Dict, List, Set

import numpy as np


def graph_fingerprint(incidence: np.ndarray) -> str:
    canonical = np.asarray(incidence, dtype=np.int64)
    payload = str(canonical.shape).encode("ascii") + canonical.tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def incidence_members(incidence: np.ndarray) -> Dict[int, List[int]]:
    members: Dict[int, List[int]] = defaultdict(list)
    for node_id, edge_id in incidence.T.tolist():
        members[int(edge_id)].append(int(node_id))
    return dict(members)


def node_neighbors(num_nodes: int, incidence: np.ndarray) -> List[Set[int]]:
    adjacency: List[Set[int]] = [set() for _ in range(num_nodes)]
    for members in incidence_members(incidence).values():
        for node_id in members:
            adjacency[node_id].update(other for other in members if other != node_id)
    return adjacency


def exact_hop_neighborhoods(
    num_nodes: int,
    incidence: np.ndarray,
    max_hops: int,
) -> List[List[List[int]]]:
    """Return exact-distance node neighborhoods for every center node."""
    adjacency = node_neighbors(num_nodes, incidence)
    all_nodes: List[List[List[int]]] = []
    for center in range(num_nodes):
        seen = {center}
        frontier = {center}
        scales: List[List[int]] = [[center]]
        for _ in range(max_hops):
            next_frontier: Set[int] = set()
            for node_id in frontier:
                next_frontier.update(adjacency[node_id])
            next_frontier -= seen
            scales.append(sorted(next_frontier))
            seen.update(next_frontier)
            frontier = next_frontier
        all_nodes.append(scales)
    return all_nodes
