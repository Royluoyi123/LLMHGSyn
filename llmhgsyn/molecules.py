from __future__ import annotations

import hashlib
import re
import warnings
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


ATOM_DIM = 16
BOND_DIM = 14
MOLECULE_CACHE_VERSION = 2
_TOKEN_PATTERN = re.compile(r"\[[^\]]+\]|Br|Cl|Si|Se|[A-Z][a-z]?|[bcnops]")


def _one_hot_with_other(value, choices) -> List[float]:
    encoded = [float(value == choice) for choice in choices]
    encoded.append(float(sum(encoded) == 0))
    return encoded


def _rdkit_bond_features(bond) -> np.ndarray:
    from rdkit import Chem

    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    stereo_types = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
    ]
    return np.asarray(
        [
            *_one_hot_with_other(bond.GetBondType(), bond_types),
            float(bond.GetIsConjugated()),
            float(bond.IsInRing()),
            *_one_hot_with_other(bond.GetStereo(), stereo_types),
        ],
        dtype=np.float32,
    )


def _rdkit_graph(smiles: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    hybridizations = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    features = []
    for atom in molecule.GetAtoms():
        hybrid = [float(atom.GetHybridization() == value) for value in hybridizations]
        hybrid.append(float(sum(hybrid) == 0))
        features.append(
            [
                atom.GetAtomicNum() / 100.0,
                atom.GetDegree() / 6.0,
                atom.GetFormalCharge() / 5.0,
                atom.GetTotalNumHs() / 4.0,
                float(atom.GetIsAromatic()),
                float(atom.IsInRing()),
                atom.GetMass() / 200.0,
                atom.GetTotalValence() / 8.0,
                atom.GetNumRadicalElectrons() / 4.0,
                float(atom.HasProp("_ChiralityPossible")),
                *hybrid,
            ]
        )

    sources: List[int] = []
    targets: List[int] = []
    edge_features: List[np.ndarray] = []
    for bond in molecule.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_features = _rdkit_bond_features(bond)
        sources.extend([left, right])
        targets.extend([right, left])
        edge_features.extend([bond_features, bond_features])

    edge_index = np.asarray([sources, targets], dtype=np.int64)
    if edge_features:
        edge_attr = np.stack(edge_features).astype(np.float32, copy=False)
    else:
        edge_attr = np.empty((0, BOND_DIM), dtype=np.float32)
    return np.asarray(features, dtype=np.float32), edge_index, edge_attr


def _lexical_graph(smiles: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dependency-free fallback; RDKit is strongly recommended for experiments."""
    tokens = _TOKEN_PATTERN.findall(smiles)
    if not tokens:
        tokens = ["C"]
    features = np.zeros((len(tokens), ATOM_DIM), dtype=np.float32)
    for idx, token in enumerate(tokens):
        symbol = token.strip("[]")
        stable_index = int.from_bytes(
            hashlib.sha256(symbol.encode("utf-8")).digest()[:4], "little"
        )
        features[idx, stable_index % ATOM_DIM] = 1.0
        features[idx, -1] = float(symbol.islower())

    sources: List[int] = []
    targets: List[int] = []
    edge_features: List[np.ndarray] = []
    single_bond = np.zeros(BOND_DIM, dtype=np.float32)
    single_bond[0] = 1.0
    single_bond[7] = 1.0
    for idx in range(len(tokens) - 1):
        sources.extend([idx, idx + 1])
        targets.extend([idx + 1, idx])
        edge_features.extend([single_bond, single_bond])
    edge_index = np.asarray([sources, targets], dtype=np.int64)
    if edge_features:
        edge_attr = np.stack(edge_features).astype(np.float32, copy=False)
    else:
        edge_attr = np.empty((0, BOND_DIM), dtype=np.float32)
    return features, edge_index, edge_attr


def build_molecular_graph_batch(
    smiles_list: Sequence[str],
    allow_lexical_fallback: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not smiles_list:
        raise ValueError("At least one drug SMILES is required")
    graphs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    try:
        import rdkit  # noqa: F401

        featurizer = _rdkit_graph
    except ImportError:
        if not allow_lexical_fallback:
            raise ImportError(
                "RDKit is required to build molecular graphs. Install requirements.txt "
                "or pass allow_lexical_fallback=True only for a smoke test."
            )
        warnings.warn("RDKit is unavailable; using an approximate SMILES-token graph")
        featurizer = _lexical_graph
    for smiles in smiles_list:
        graphs.append(featurizer(smiles))

    node_features: List[np.ndarray] = []
    edge_indices: List[np.ndarray] = []
    edge_attributes: List[np.ndarray] = []
    molecule_batch: List[np.ndarray] = []
    node_offset = 0
    for graph_id, (features, edge_index, edge_attr) in enumerate(graphs):
        node_features.append(features)
        edge_indices.append(edge_index + node_offset)
        edge_attributes.append(edge_attr)
        molecule_batch.append(np.full(len(features), graph_id, dtype=np.int64))
        node_offset += len(features)

    return (
        np.concatenate(node_features, axis=0).astype(np.float32, copy=False),
        np.concatenate(edge_indices, axis=1).astype(np.int64, copy=False),
        np.concatenate(edge_attributes, axis=0).astype(np.float32, copy=False),
        np.concatenate(molecule_batch, axis=0).astype(np.int64, copy=False),
    )


def _smiles_fingerprint(smiles_list: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for smiles in smiles_list:
        digest.update(smiles.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_or_build_molecular_graphs(
    smiles_list: Sequence[str],
    cache_path: str | Path,
    allow_lexical_fallback: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache_path = Path(cache_path)
    fingerprint = _smiles_fingerprint(smiles_list)
    required = {
        "node_features",
        "edge_index",
        "edge_attr",
        "molecule_batch",
        "smiles_fingerprint",
        "cache_version",
    }
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as data:
            if required.issubset(data.files):
                cached_fingerprint = str(data["smiles_fingerprint"].item())
                cached_version = int(data["cache_version"].item())
                if cached_fingerprint == fingerprint and cached_version == MOLECULE_CACHE_VERSION:
                    return (
                        data["node_features"],
                        data["edge_index"],
                        data["edge_attr"],
                        data["molecule_batch"],
                    )
        warnings.warn(
            f"Rebuilding outdated or incompatible molecular graph cache: {cache_path}"
        )

    result = build_molecular_graph_batch(smiles_list, allow_lexical_fallback)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        node_features=result[0],
        edge_index=result[1],
        edge_attr=result[2],
        molecule_batch=result[3],
        smiles_fingerprint=np.asarray(fingerprint),
        cache_version=np.asarray(MOLECULE_CACHE_VERSION, dtype=np.int64),
    )
    return result
