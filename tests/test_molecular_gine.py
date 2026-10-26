import tempfile
import unittest
from pathlib import Path

import numpy as np

from llmhgsyn.molecules import (
    ATOM_DIM,
    BOND_DIM,
    build_molecular_graph_batch,
    load_or_build_molecular_graphs,
)


class MolecularGraphBatchTest(unittest.TestCase):
    def test_graph_batch_contains_directed_bonds_and_bond_features(self):
        atom_x, edge_index, edge_attr, molecule_batch = build_molecular_graph_batch(
            ["CCO", "C"], allow_lexical_fallback=True
        )
        self.assertEqual(atom_x.shape, (4, ATOM_DIM))
        self.assertEqual(edge_index.shape, (2, 4))
        self.assertEqual(edge_attr.shape, (4, BOND_DIM))
        self.assertEqual(molecule_batch.tolist(), [0, 0, 0, 1])
        self.assertTrue(np.all(edge_attr[:, 0] == 1.0))

    def test_old_dense_cache_is_rebuilt_in_sparse_edge_format(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "molecules.npz"
            np.savez_compressed(
                cache,
                node_features=np.zeros((2, 3, ATOM_DIM), dtype=np.float32),
                adjacency=np.zeros((2, 3, 3), dtype=np.float32),
                mask=np.ones((2, 3), dtype=np.float32),
            )
            result = load_or_build_molecular_graphs(
                ["CCO", "C"], cache, allow_lexical_fallback=True
            )
            self.assertEqual(len(result), 4)
            with np.load(cache, allow_pickle=False) as data:
                self.assertIn("edge_attr", data.files)
                self.assertIn("molecule_batch", data.files)


try:
    import torch
    import torch_geometric  # noqa: F401
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch and PyTorch Geometric are not installed")
class MolecularGINEEncoderTest(unittest.TestCase):
    def test_encoder_returns_one_vector_per_drug(self):
        from llmhgsyn.models import MolecularGraphEncoder

        arrays = build_molecular_graph_batch(["CCO", "C"], allow_lexical_fallback=True)
        atom_x, edge_index, edge_attr, molecule_batch = [torch.from_numpy(x) for x in arrays]
        encoder = MolecularGraphEncoder(ATOM_DIM, BOND_DIM, 32, 32, dropout=0.0)
        output = encoder(atom_x, edge_index.long(), edge_attr, molecule_batch.long())
        self.assertEqual(tuple(output.shape), (2, 32))
        output.sum().backward()


if __name__ == "__main__":
    unittest.main()
