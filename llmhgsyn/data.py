from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NodeCatalog:
    drug_names: List[str]
    cell_names: List[str]
    smiles: List[str]
    expression: np.ndarray
    drug_to_id: Dict[str, int]
    cell_to_id: Dict[str, int]

    @property
    def num_drugs(self) -> int:
        return len(self.drug_names)

    @property
    def num_cells(self) -> int:
        return len(self.cell_names)

    @property
    def num_nodes(self) -> int:
        return self.num_drugs + self.num_cells 

    def node_type(self, node_id: int) -> str:
        return "drug" if node_id < self.num_drugs else "cell_line"

    def node_name(self, node_id: int) -> str:
        if node_id < self.num_drugs:
            return self.drug_names[node_id]
        return self.cell_names[node_id - self.num_drugs]


@dataclass(frozen=True)
class SynergyFold:
    catalog: NodeCatalog
    train: np.ndarray
    test: np.ndarray
    incidence: np.ndarray


def load_catalog(data_dir: str | Path) -> NodeCatalog:
    data_dir = Path(data_dir)
    drug_df = pd.read_csv(data_dir / "drug_smiles.csv")
    cell_df = pd.read_csv(data_dir / "cell_line_gene_expression.csv", index_col=0)

    required = {"Drug_name", "Smiles"}
    if not required.issubset(drug_df.columns):
        raise ValueError(f"drug_smiles.csv must contain {sorted(required)}")
    if drug_df["Drug_name"].duplicated().any() or cell_df.index.duplicated().any():
        raise ValueError("Drug and cell-line names must be unique")

    drug_names = drug_df["Drug_name"].astype(str).tolist()
    cell_names = cell_df.index.astype(str).tolist()
    drug_to_id = {name: idx for idx, name in enumerate(drug_names)}
    cell_to_id = {
        name: len(drug_names) + idx for idx, name in enumerate(cell_names)
    }
    expression = cell_df.to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(expression).all():
        raise ValueError("Cell-line expression contains NaN or infinite values")

    return NodeCatalog(
        drug_names=drug_names,
        cell_names=cell_names,
        smiles=drug_df["Smiles"].astype(str).tolist(),
        expression=expression,
        drug_to_id=drug_to_id,
        cell_to_id=cell_to_id,
    )


def load_triples(csv_path: str | Path, catalog: NodeCatalog) -> np.ndarray:
    frame = pd.read_csv(csv_path)
    required = ["drug_a", "drug_b", "cell_line", "synergy"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {missing}")

    unknown_drugs = (
        set(frame["drug_a"].astype(str))
        | set(frame["drug_b"].astype(str))
    ) - set(catalog.drug_to_id)
    unknown_cells = set(frame["cell_line"].astype(str)) - set(catalog.cell_to_id)
    if unknown_drugs or unknown_cells:
        raise ValueError(
            f"Unknown IDs in {csv_path}: drugs={sorted(unknown_drugs)[:5]}, "
            f"cells={sorted(unknown_cells)[:5]}"
        )

    labels = frame["synergy"].astype(int).to_numpy()
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("This pipeline expects binary synergy labels (0/1)")

    triples = np.column_stack(
        [
            frame["drug_a"].astype(str).map(catalog.drug_to_id),
            frame["drug_b"].astype(str).map(catalog.drug_to_id),
            frame["cell_line"].astype(str).map(catalog.cell_to_id),
            labels,
        ]
    )
    return triples.astype(np.int64, copy=False)


def build_positive_incidence(train: np.ndarray, deduplicate: bool = True) -> np.ndarray:
    """Build [node_id, hyperedge_id] incidence entries from positive train triples only."""
    positives = train[train[:, 3] == 1, :3]
    if deduplicate:
        # Drug order does not define a different synergy hyperedge.
        canonical = positives.copy()
        canonical[:, :2] = np.sort(canonical[:, :2], axis=1)
        positives = np.unique(canonical, axis=0)
    if len(positives) == 0:
        raise ValueError("The selected training fold has no positive samples")
    nodes = positives.reshape(-1)
    edge_ids = np.repeat(np.arange(len(positives), dtype=np.int64), 3)
    return np.vstack([nodes, edge_ids]).astype(np.int64, copy=False)


def load_fold(
    data_dir: str | Path,
    split: str = "warm_start",
    fold: int = 1,
) -> SynergyFold:
    if fold not in range(1, 6):
        raise ValueError("fold must be in [1, 5]")
    data_dir = Path(data_dir)
    catalog = load_catalog(data_dir)
    split_dir = data_dir / split
    train = load_triples(split_dir / f"train_{fold}.csv", catalog) #训练集
    test = load_triples(split_dir / f"test_{fold}.csv", catalog) #测试集
    incidence = build_positive_incidence(train) #构建超边 2，超边数量x3 第一行节点编号，第二行超边编号
    return SynergyFold(catalog=catalog, train=train, test=test, incidence=incidence)


def load_descriptions(data_root: str | Path, catalog: NodeCatalog) -> List[str]:
    root = Path(data_root)
    feature_dir = root / "LLM_features"
    with (feature_dir / "ensem_describe_deepsynergydrug_default_all.pickle").open("rb") as f:
        drug_desc = pickle.load(f)
    with (feature_dir / "ensem_describe_deepsynergycellline_default_all.pickle").open("rb") as f:
        cell_desc = pickle.load(f)
    missing_drugs = [name for name in catalog.drug_names if name not in drug_desc]
    missing_cells = [name for name in catalog.cell_names if name not in cell_desc]
    if missing_drugs or missing_cells:
        raise ValueError(
            f"Missing LLM descriptions: drugs={missing_drugs[:5]}, cells={missing_cells[:5]}"
        )
    return [str(drug_desc[name]) for name in catalog.drug_names] + [
        str(cell_desc[name]) for name in catalog.cell_names
    ]


def dataset_summary(fold_data: SynergyFold) -> Dict[str, int]:
    positive_train = int((fold_data.train[:, 3] == 1).sum())
    return {
        "num_drugs": fold_data.catalog.num_drugs,
        "num_cell_lines": fold_data.catalog.num_cells,
        "num_nodes": fold_data.catalog.num_nodes,
        "num_train_samples": len(fold_data.train),
        "num_test_samples": len(fold_data.test),
        "num_positive_train_samples": positive_train,
        "num_positive_hyperedges": int(fold_data.incidence[1].max()) + 1,
    }


def save_summary(summary: Dict[str, int], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
