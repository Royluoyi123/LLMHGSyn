from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np

from .data import NodeCatalog, load_descriptions, load_fold
from .hypergraph import graph_fingerprint, incidence_members


SYSTEM_INSTRUCTION = (
    "You are a biomedical graph expert. Given a center node, its biomedical description, "
    "a multi-hop drug-drug-cell-line hypergraph context, and several candidate samples, "
    "classify every candidate as synergistic or non-synergistic. Return only the requested "
    "JSON object. The structural context is built from positive training samples only."
)


Triple = Tuple[int, int, int]


def _shorten(text: str, max_chars: int) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= max_chars else compact[: max_chars - 1] + "…"


def _select(values: List[int], limit: int, seed: int) -> List[int]:
    if len(values) <= limit:
        return values
    generator = random.Random(seed)
    return sorted(generator.sample(values, limit))


def _canonical_triple(drug_a: int, drug_b: int, cell_line: int) -> Triple:
    return min(drug_a, drug_b), max(drug_a, drug_b), cell_line


def _query_pools(train: np.ndarray, num_nodes: int):
    labels_by_triple: Dict[Triple, Set[int]] = defaultdict(set)
    for drug_a, drug_b, cell_line, label in train.tolist():
        labels_by_triple[_canonical_triple(drug_a, drug_b, cell_line)].add(int(label))

    positive: List[List[Triple]] = [[] for _ in range(num_nodes)]
    negative: List[List[Triple]] = [[] for _ in range(num_nodes)]
    ambiguous = 0
    for triple, labels in labels_by_triple.items():
        if len(labels) != 1:
            ambiguous += 1
            continue
        label = next(iter(labels))
        pools = positive if label == 1 else negative
        for node_id in set(triple):
            pools[node_id].append(triple)
    for pools in (positive, negative):
        for node_pool in pools:
            node_pool.sort()
    return positive, negative, ambiguous


def _sample_query_bag(
    positive_pool: List[Triple],
    negative_pool: List[Triple],
    samples_per_prompt: int,
    positive_per_prompt: int,
    seed: int,
) -> List[Tuple[Triple, int]]:
    generator = random.Random(seed)
    positive_count = min(positive_per_prompt, len(positive_pool), samples_per_prompt)
    negative_count = min(samples_per_prompt - positive_count, len(negative_pool))
    remaining = samples_per_prompt - positive_count - negative_count
    if remaining > 0:
        positive_count += min(remaining, len(positive_pool) - positive_count)
        remaining = samples_per_prompt - positive_count - negative_count
    if remaining > 0:
        negative_count += min(remaining, len(negative_pool) - negative_count)
    selected = [
        (triple, 1) for triple in generator.sample(positive_pool, positive_count)
    ] + [
        (triple, 0) for triple in generator.sample(negative_pool, negative_count)
    ]
    generator.shuffle(selected)
    return selected


def _masked_graph_view(
    catalog: NodeCatalog,
    edge_members: Dict[int, List[int]],
    masked_triples: Set[Triple],
    center_node: int,
    max_hops: int,
):
    active_edge_members: Dict[int, List[int]] = {}
    node_edges: List[List[int]] = [[] for _ in range(catalog.num_nodes)]
    adjacency: List[Set[int]] = [set() for _ in range(catalog.num_nodes)]
    for edge_id, members in edge_members.items():
        drugs = sorted(node_id for node_id in members if node_id < catalog.num_drugs)
        cells = [node_id for node_id in members if node_id >= catalog.num_drugs]
        if len(drugs) != 2 or len(cells) != 1:
            raise ValueError(f"Hyperedge {edge_id} is not a drug-drug-cell-line triple")
        triple = _canonical_triple(drugs[0], drugs[1], cells[0])
        if triple in masked_triples:
            continue
        active_edge_members[edge_id] = members
        for node_id in members:
            node_edges[node_id].append(edge_id)
            adjacency[node_id].update(other for other in members if other != node_id)

    neighborhoods: List[List[int]] = [[center_node]]
    seen = {center_node}
    frontier = {center_node}
    for _ in range(max_hops):
        next_frontier: Set[int] = set()
        for node_id in frontier:
            next_frontier.update(adjacency[node_id])
        next_frontier -= seen
        neighborhoods.append(sorted(next_frontier))
        seen.update(next_frontier)
        frontier = next_frontier
    return active_edge_members, node_edges, neighborhoods


def _render_query_samples(
    catalog: NodeCatalog,
    descriptions: List[str],
    center_node: int,
    query_bag: List[Tuple[Triple, int]],
    description_chars: int,
):
    lines = ["Samples to classify:"]
    targets: Dict[str, str] = {}
    triples: List[List[int]] = []
    labels: List[int] = []
    if not query_bag:
        lines.extend(
            [
                "No unambiguous training sample contains this center node.",
                "Query: Return the no-training-samples status.",
            ]
        )
        targets["status"] = "no_training_samples"
        return lines, targets, triples, labels
    for sample_index, (triple, label) in enumerate(query_bag, start=1):
        drug_a, drug_b, cell_line = triple
        sample_key = f"sample_{sample_index}"
        lines.extend(
            [
                f"{sample_key}:",
                f"- drug_a: {catalog.node_name(drug_a)}",
                f"- drug_b: {catalog.node_name(drug_b)}",
                f"- cell_line: {catalog.node_name(cell_line)}",
            ]
        )
        for node_id in (drug_a, drug_b, cell_line):
            if node_id == center_node:
                continue
            lines.append(
                f"- {catalog.node_name(node_id)} description: "
                + _shorten(descriptions[node_id], description_chars)
            )
        targets[sample_key] = "synergistic" if label == 1 else "non-synergistic"
        triples.append(list(triple))
        labels.append(label)
    lines.append(
        "Query: Classify every sample as synergistic or non-synergistic in the same order."
    )
    return lines, targets, triples, labels


def build_instruction_records(
    catalog: NodeCatalog,
    incidence,
    train: np.ndarray,
    descriptions: List[str],
    max_hops: int = 2,
    max_nodes_per_hop: int = 8,
    max_edges_per_hop: int = 8,
    description_chars: int = 180,
    center_description_chars: int = 1000,
    samples_per_prompt: int = 8,
    positive_per_prompt: int = 4,
    bags_per_node: int = 4,
    seed: int = 42,
) -> List[Dict[str, object]]:
    if samples_per_prompt <= 0 or bags_per_node <= 0:
        raise ValueError("samples_per_prompt and bags_per_node must be positive")
    if positive_per_prompt < 0 or positive_per_prompt > samples_per_prompt:
        raise ValueError("positive_per_prompt must be in [0, samples_per_prompt]")
    fingerprint = graph_fingerprint(incidence) #超图指纹 
    edge_members = incidence_members(incidence)  #超边索引 表示一条超边包含了哪些节点
    positive_pools, negative_pools, _ = _query_pools(train, catalog.num_nodes) #按照中心节点分成俩个查询池
    records: List[Dict[str, object]] = []
    for center_node in range(catalog.num_nodes):
        center_name = catalog.node_name(center_node)
        center_type = catalog.node_type(center_node)
        for bag_id in range(bags_per_node):
            query_bag = _sample_query_bag(
                positive_pools[center_node],
                negative_pools[center_node],
                samples_per_prompt,
                positive_per_prompt,
                seed + center_node * 10007 + bag_id * 101,
            )
            masked_triples = {triple for triple, label in query_bag if label == 1}
            active_edge_members, node_edges, neighborhoods = _masked_graph_view(
                catalog,
                edge_members,
                masked_triples,
                center_node,
                max_hops,
            )
            query_lines, targets, query_triples, query_labels = _render_query_samples(
                catalog,
                descriptions,
                center_node,
                query_bag,
                description_chars,
            )
            for hop in range(max_hops + 1):
                context_lines = [
                    f"Center node: {center_name}",
                    f"Center type: {center_type}",
                    "Center description: "
                    + _shorten(descriptions[center_node], center_description_chars),
                    f"Graph scale: {hop}-hop",
                ]
                if hop == 0:
                    context_lines.append("Neighborhood: center identity only.")
                else:
                    any_neighbor = False
                    for distance in range(1, hop + 1):
                        previous_frontier = (
                            [center_node]
                            if distance == 1
                            else neighborhoods[distance - 1]
                        )
                        candidate_edges = sorted(
                            {
                                edge_id
                                for node_id in previous_frontier
                                for edge_id in node_edges[node_id]
                            }
                        )
                        selected_edges = _select(
                            candidate_edges,
                            max_edges_per_hop,
                            seed + center_node * 2017 + bag_id * 211 + distance,
                        )
                        if selected_edges:
                            context_lines.append(
                                f"Positive synergy hyperedges at step {distance}:"
                            )
                            for edge_id in selected_edges:
                                members = active_edge_members[edge_id]
                                drugs = [
                                    node_id
                                    for node_id in members
                                    if catalog.node_type(node_id) == "drug"
                                ]
                                cells = [
                                    node_id
                                    for node_id in members
                                    if catalog.node_type(node_id) == "cell_line"
                                ]
                                if len(drugs) == 2 and len(cells) == 1:
                                    context_lines.append(
                                        "- (drug_a="
                                        + catalog.node_name(drugs[0])
                                        + ", drug_b="
                                        + catalog.node_name(drugs[1])
                                        + ", cell_line="
                                        + catalog.node_name(cells[0])
                                        + ")"
                                    )
                        selected = _select(
                            neighborhoods[distance],
                            max_nodes_per_hop,
                            seed + center_node * 1009 + bag_id * 307 + distance,
                        )
                        if not selected:
                            continue
                        any_neighbor = True
                        context_lines.append(f"Exact {distance}-hop neighbors:")
                        for node_id in selected:
                            context_lines.append(
                                "- "
                                + f"{catalog.node_name(node_id)} [{catalog.node_type(node_id)}]: "
                                + _shorten(descriptions[node_id], description_chars)
                            )
                    if not any_neighbor:
                        context_lines.append(
                            "Neighborhood: no connected positive training hyperedge."
                        )
                context_lines.extend(query_lines)
                records.append(
                    {
                        "id": f"node-{center_node}-hop-{hop}-bag-{bag_id}",
                        "center_node": center_node,
                        "node_name": center_name,
                        "node_type": center_type,
                        "hop": hop,
                        "bag_id": bag_id,
                        "graph_id": fingerprint,
                        "instruction": SYSTEM_INSTRUCTION,
                        "input": "\n".join(context_lines),
                        "target": json.dumps(targets, ensure_ascii=False),
                        "query_triples": query_triples,
                        "query_labels": query_labels,
                        "masked_positive_triples": [
                            list(triple) for triple in sorted(masked_triples)
                        ],
                    }
                )
            if not query_bag:
                break
    return records


def write_jsonl(records: Iterable[Dict[str, object]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> List[Dict[str, object]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare_instruction_data(
    data_root: str | Path,
    split: str,
    fold: int,
    output_dir: str | Path,
    max_hops: int = 2,
    samples_per_prompt: int = 8,
    positive_per_prompt: int = 4,
    bags_per_node: int = 4,
    seed: int = 42,
    dataset: str = "GDSC"
) -> Dict[str, int]:
    
    data_root = Path(data_root)
    fold_data = load_fold(data_root / dataset, split=split, fold=fold)
    descriptions = load_descriptions(data_root, fold_data.catalog)
    records = build_instruction_records(
        fold_data.catalog,
        fold_data.incidence,
        fold_data.train,
        descriptions,
        max_hops=max_hops,
        samples_per_prompt=samples_per_prompt,
        positive_per_prompt=positive_per_prompt,
        bags_per_node=bags_per_node,
        seed=seed,
    ) #所有节点的prompt样本
    for record in records:
        record["split"] = split
        record["fold"] = fold

    output_dir = Path(output_dir)
    write_jsonl(records, output_dir / "all.jsonl")
    write_jsonl(records, output_dir / "train.jsonl")
    catalog_records = [
        {
            "node_id": node_id,
            "name": fold_data.catalog.node_name(node_id),
            "type": fold_data.catalog.node_type(node_id),
        }
        for node_id in range(fold_data.catalog.num_nodes)
    ]
    (output_dir / "node_catalog.json").write_text(
        json.dumps(catalog_records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _, _, ambiguous = _query_pools(fold_data.train, fold_data.catalog.num_nodes)
    return {
        "all": len(records),
        "train": len(records),
        "bags_per_node": bags_per_node,
        "samples_per_prompt": samples_per_prompt,
        "ambiguous_query_triples_excluded": ambiguous,
    }


def format_prompt(record: Dict[str, object], include_target: bool = False) -> str:
    prompt = (
        f"### Instruction:\n{record['instruction']}\n\n"
        f"### Input:\n{record['input']}\n\n"
        "### Response:\n"
    )
    return prompt + (str(record["target"]) if include_target else "")
