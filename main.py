from __future__ import annotations

import argparse

from llmhgsyn.training import train_synergy_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the multi-scale LLM-distilled HGNN drug-synergy model"
    )
    parser.add_argument("--data-root", default="LLMHGSyn/Data")
    parser.add_argument(
        "--dataset",
        choices=["GDSC", "DrugCombDB"],
        default="GDSC",
    )
    parser.add_argument("--split", default="warm_start")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--teacher-features", default="LLMHGSyn/outputs/teacher/warm_start/fold_1.npz")
    parser.add_argument("--molecule-cache", default="LLMHGSyn/outputs/cache/gdsc_molecular_graphs.npz")
    parser.add_argument("--output-dir", default="LLMHGSyn/outputs/train/warm_start/fold_1_gine")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hgnn-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--distill-weight", type=float, default=0.4)
    parser.add_argument("--cosine-weight", type=float, default=0.1)
    parser.add_argument("--distill-warmup-ratio", type=float, default=0.1)
    parser.add_argument("--distill-ramp-epochs", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument(
        "--distill-scales",
        default="all",
        help="Comma-separated teacher/student scale indices, or all",
    )
    parser.add_argument(
        "--distill-loss-mode",
        choices=["both", "cosine_only", "infonce_only"],
        default="both",
    )
    parser.add_argument(
        "--student-fusion",
        choices=["learned", "last"],
        default="learned",
        help=(
            "Use learnable multi-scale fusion or only the last HGNN state "
            "for prediction"
        ),
    )
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--early-stop-metric",
        choices=["roc_auc", "aupr", "f1", "mcc", "kappa"],
        default="roc_auc",
        help="Metric used for checkpoint selection and early stopping",
    )
    parser.add_argument(
        "--early-stop-source",
        choices=["train", "test"],
        default="test",
        help="Dataset split used for checkpoint selection and early stopping",
    )
    parser.add_argument("--early-stop-patience", type=int, default=50)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=3)
    parser.add_argument(
        "--allow-lexical-fallback",
        action="store_true",
        help="Smoke-test only: approximate molecular graphs when RDKit is unavailable",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train_synergy_model(parse_args())
