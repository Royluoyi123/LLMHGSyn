import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from llmhgsyn.data import load_fold
from llmhgsyn.instructions import prepare_instruction_data, read_jsonl


ROOT = Path(__file__).resolve().parents[1]


class DataPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fold = load_fold(ROOT / "Data/GDSC", "warm_start", 1)

    def test_node_catalog_and_global_cell_offset(self):
        self.assertEqual(self.fold.catalog.num_drugs, 63)
        self.assertEqual(self.fold.catalog.num_cells, 112)
        self.assertEqual(self.fold.catalog.num_nodes, 175)
        self.assertGreaterEqual(self.fold.train[:, 2].min(), self.fold.catalog.num_drugs)

    def test_incidence_uses_positive_samples_only(self):
        positive_canonical = self.fold.train[self.fold.train[:, 3] == 1, :3].copy()
        positive_canonical[:, :2] = np.sort(positive_canonical[:, :2], axis=1)
        positive_canonical = np.unique(positive_canonical, axis=0)
        num_edges = int(self.fold.incidence[1].max()) + 1
        self.assertEqual(num_edges, len(positive_canonical))
        self.assertEqual(self.fold.incidence.shape[1], num_edges * 3)

    def test_multiscale_instructions_cover_every_node(self):
        with tempfile.TemporaryDirectory() as directory:
            counts = prepare_instruction_data(
                ROOT / "Data",
                "warm_start",
                1,
                directory,
                max_hops=2,
                bags_per_node=2,
            )
            records = read_jsonl(Path(directory) / "all.jsonl")
            self.assertGreaterEqual(counts["all"], self.fold.catalog.num_nodes * 3)
            self.assertLessEqual(counts["all"], self.fold.catalog.num_nodes * 3 * 2)
            self.assertEqual(
                {(int(item["center_node"]), int(item["hop"])) for item in records},
                {(node, hop) for node in range(self.fold.catalog.num_nodes) for hop in range(3)},
            )
            self.assertEqual(len({item["id"] for item in records}), len(records))
            train_records = read_jsonl(Path(directory) / "train.jsonl")
            self.assertEqual(train_records, records)

    def test_descriptions_are_inputs_and_targets_are_synergy_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            prepare_instruction_data(
                ROOT / "Data",
                "warm_start",
                1,
                directory,
                max_hops=2,
                bags_per_node=1,
            )
            records = read_jsonl(Path(directory) / "all.jsonl")
            record = next(
                item
                for item in records
                if item["center_node"] == 0 and item["hop"] == 1
            )
            self.assertIn("Center description: Dasatinib", record["input"])
            target = json.loads(record["target"])
            self.assertEqual(len(target), 8)
            self.assertEqual(set(target.values()), {"synergistic", "non-synergistic"})
            self.assertEqual(record["query_labels"].count(1), 4)
            self.assertEqual(record["query_labels"].count(0), 4)

    def test_positive_query_hyperedges_are_masked_from_structural_context(self):
        with tempfile.TemporaryDirectory() as directory:
            prepare_instruction_data(
                ROOT / "Data",
                "warm_start",
                1,
                directory,
                max_hops=2,
                bags_per_node=1,
            )
            records = read_jsonl(Path(directory) / "all.jsonl")
            record = next(
                item
                for item in records
                if item["center_node"] == 0 and item["hop"] == 2
            )
            structural_context = record["input"].split("Samples to classify:", 1)[0]
            for triple, label in zip(record["query_triples"], record["query_labels"]):
                if label != 1:
                    continue
                drug_a, drug_b, cell_line = triple
                rendered_edge = (
                    f"(drug_a={self.fold.catalog.node_name(drug_a)}, "
                    f"drug_b={self.fold.catalog.node_name(drug_b)}, "
                    f"cell_line={self.fold.catalog.node_name(cell_line)})"
                )
                self.assertNotIn(rendered_edge, structural_context)


if __name__ == "__main__":
    unittest.main()
