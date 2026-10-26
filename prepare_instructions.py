from __future__ import annotations

import argparse
import json
from pathlib import Path
from llmhgsyn.data import dataset_summary, load_fold, save_summary
from llmhgsyn.instructions import prepare_instruction_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build multi-scale node graph instructions")
    parser.add_argument("--data-root", default="LLMHGSyn/Data")
    parser.add_argument("--split", default="warm_start")
    parser.add_argument("--fold", type=int, default=5)
    parser.add_argument("--hops", type=int, default=2) #几跳子图
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--positive-per-prompt", type=int, default=4)
    parser.add_argument("--bags-per-node", type=int, default=4) #每个节点有几组边 分类预测
    parser.add_argument("--output-dir", default="LLMHGSyn/outputs_DrugCombDB/instructions/warm_start/fold_5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dataset",
        choices=["GDSC", "DrugCombDB"],
        default="DrugCombDB",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fold_data = load_fold(Path(args.data_root) / args.dataset, args.split, args.fold)
    summary = dataset_summary(fold_data)
    counts = prepare_instruction_data(
        args.data_root,
        args.split,
        args.fold,
        args.output_dir,
        max_hops=args.hops,
        samples_per_prompt=args.samples_per_prompt,
        positive_per_prompt=args.positive_per_prompt,
        bags_per_node=args.bags_per_node,
        seed=args.seed,
        dataset=args.dataset
    )
    save_summary(summary, f"{args.output_dir}/dataset_summary.json")
    print(json.dumps({**summary, "instructions": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
