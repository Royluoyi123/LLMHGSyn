from __future__ import annotations

import argparse
import csv
import itertools
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parent
METRIC_NAMES = [
    "roc_auc",
    "aupr",
    "accuracy",
    "balanced_accuracy",
    "f1",
    "mcc",
    "kappa",
]


def _float_list(value: str) -> List[float]:
    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid float list: {value!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError("The search list cannot be empty")
    return values


def _int_list(value: str) -> List[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer list: {value!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError("The search list cannot be empty")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Five-fold grid search for the LLM-distilled GINE-HGNN model"
    )
    parser.add_argument("--data-root", default="LLMHGSyn/Data")
    parser.add_argument(
        "--dataset",
        choices=["GDSC", "DrugCombDB"],
        default="GDSC",
    )
    parser.add_argument("--split", default="drug_pairs")
    parser.add_argument("--folds", type=_int_list, default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--teacher-root",
        default= "LLMHGSyn/outputs/teacher/drug_pairs",
        help="Directory containing fold_1.npz through fold_5.npz",
    )
    parser.add_argument(
        "--molecule-cache",
        default="LLMHGSyn/outputs/cache/gdsc_molecular_graphs.npz")
    parser.add_argument("--results-csv",
        default= "LLMHGSyn/outputs/grid_search/drug_pairs/five_fold_results.csv")
    parser.add_argument(
        "--best-model-dir",
        default= "LLMHGSyn/outputs/grid_search/drug_pairs/best_models",
        help="Directory used to retain only the best configuration's fold models",
    )
    parser.add_argument(
        "--device",
        default="cuda:3",
        help="PyTorch device, for example 3 or cuda:3",
    )  
    parser.add_argument("--hgnn-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--early-stop-patience", type=int, default=200)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--early-stop-metric",
        choices=["roc_auc", "aupr", "f1", "mcc", "kappa"],
        default="roc_auc",
    )
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-lexical-fallback", action="store_true")
    parser.add_argument(
        "--temp-dir",
        default=None,
        help="Optional parent directory for temporary files",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore an existing results CSV and run every configuration again",
    )
    parser.add_argument("--hidden-dims", type=_int_list, default=[128,256,512])
    parser.add_argument(
        "--learning-rates",
        type=_float_list,
        default=[1e-3,1e-4],
    )
    parser.add_argument(
        "--weight-decays",
        type=_float_list,
        default=[1e-5,0],
    )
    parser.add_argument(
        "--dropouts",
        type=_float_list,
        default=[0,0.1,0.2,0.3,0.4],
    )
    parser.add_argument(
        "--distill-weights",
        type=_float_list,
        default=[0.01,0.02, 0.1],
    )
    parser.add_argument(
        "--temperatures",
        type=_float_list,
        default=[0.05,0.1,0.2],
    )
    parser.add_argument("--cosine-weight", type=float, default=0.1)
    parser.add_argument("--distill-warmup-ratio", type=float, default=0.1)
    parser.add_argument("--distill-ramp-epochs", type=int, default=50)
    parser.add_argument("--grad-clips", type=_float_list, default=[5.0])
    return parser.parse_args()


def _fieldnames(folds: List[int]) -> List[str]:
    fields = [
        "selection_rank",
        "config_id",
        "status",
        "hidden_dim",
        "learning_rate",
        "weight_decay",
        "dropout",
        "distill_weight",
        "cosine_weight",
        "distill_warmup_ratio",
        "distill_ramp_epochs",
        "temperature",
        "grad_clip",
    ]

    for name in METRIC_NAMES:
        fields.extend([
            f"{name}_mean",
            f"{name}_std",
        ])

    return fields


def _write_results(
    rows: Iterable[Dict[str, object]], output: Path, folds: List[int]
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.parent / f".{output.name}.tmp"
    with temporary_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=_fieldnames(folds),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary_output.replace(output)


def _read_existing_results(output: Path) -> Dict[int, Dict[str, object]]:
    if not output.exists():
        return {}
    with output.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "config_id" not in reader.fieldnames:
            raise ValueError(f"Existing results CSV has no config_id column: {output}")
        rows: Dict[int, Dict[str, object]] = {}
        for row in reader:
            raw_config_id = str(row.get("config_id", "")).strip()
            if not raw_config_id:
                continue
            rows[int(float(raw_config_id))] = dict(row)
    return rows


def _same_value(left: object, right: object) -> bool:
    try:
        left_number = float(left)
        right_number = float(right)
    except (TypeError, ValueError):
        return str(left) == str(right)
    tolerance = 1e-12 * max(1.0, abs(left_number), abs(right_number))
    return abs(left_number - right_number) <= tolerance


def _validate_existing_results(
    rows_by_id: Dict[int, Dict[str, object]],
    configurations: List[Dict[str, object]],
    parameter_names: List[str],
) -> None:
    for config_id, row in rows_by_id.items():
        if config_id < 1 or config_id > len(configurations):
            raise ValueError(
                f"Existing CSV contains config_id={config_id}, but the current grid only "
                f"contains 1..{len(configurations)}. Use another --results-csv or --restart."
            )
        expected = configurations[config_id - 1]
        mismatched = [
            name
            for name in parameter_names
            if name not in row or not _same_value(row[name], expected[name])
        ]
        if mismatched:
            raise ValueError(
                f"Existing CSV config_id={config_id} does not match the current parameter "
                f"grid for: {', '.join(mismatched)}. Use the original grid or a new CSV."
            )


def _row_succeeded(row: Dict[str, object], ranking_field: str) -> bool:
    if str(row.get("status", "")).strip().lower() != "ok":
        return False
    try:
        return str(row.get(ranking_field, "")).strip() != "" and float(
            row[ranking_field]
        ) == float(row[ranking_field])
    except (TypeError, ValueError):
        return False


def _ranked_rows(
    rows_by_id: Dict[int, Dict[str, object]], ranking_field: str
) -> List[Dict[str, object]]:
    successful = [
        row for row in rows_by_id.values() if _row_succeeded(row, ranking_field)
    ]
    successful.sort(key=lambda row: float(row[ranking_field]), reverse=True)
    for rank, row in enumerate(successful, start=1):
        row["selection_rank"] = rank
    failed = [
        row for row in rows_by_id.values() if not _row_succeeded(row, ranking_field)
    ]
    failed.sort(key=lambda row: int(float(row["config_id"])))
    return successful + failed


def _mean_std(values: List[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return float(mean), float(std)


def _aggregate_metrics(
    config_id: int,
    config: Dict[str, object],
    args: argparse.Namespace,
    folds: List[int],
    metrics_by_fold: Dict[int, Dict[str, object]],
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "selection_rank": "",
        "config_id": config_id,
        "status": "ok",
        **config,
    }

    for name in METRIC_NAMES:
        values = [
            float(metrics_by_fold[fold]["test"][name])
            for fold in folds
        ]

        row[f"{name}_mean"], row[f"{name}_std"] = _mean_std(values)

    return row


def _trial_command(
    args: argparse.Namespace,
    fold: int,
    teacher_features: Path,
    trial_output: Path,
    config: Dict[str, object],
) -> List[str]:
    command = [
        sys.executable,
        str(ROOT / "main.py"),
        "--data-root",
        str(Path(args.data_root)),
        "--dataset",
        args.dataset,
        "--split",
        args.split,
        "--fold",
        str(fold),
        "--teacher-features",
        str(teacher_features),
        "--molecule-cache",
        str(Path(args.molecule_cache)),
        "--output-dir",
        str(trial_output),
        "--hidden-dim",
        str(config["hidden_dim"]),
        "--hgnn-layers",
        str(args.hgnn_layers),
        "--dropout",
        str(config["dropout"]),
        "--epochs",
        str(args.epochs),
        "--learning-rate",
        str(config["learning_rate"]),
        "--weight-decay",
        str(config["weight_decay"]),
        "--distill-weight",
        str(config["distill_weight"]),
        "--cosine-weight",
        str(config["cosine_weight"]),
        "--distill-warmup-ratio",
        str(config["distill_warmup_ratio"]),
        "--distill-ramp-epochs",
        str(config["distill_ramp_epochs"]),
        "--temperature",
        str(config["temperature"]),
        "--grad-clip",
        str(config["grad_clip"]),
        "--log-every",
        str(args.log_every),
        "--early-stop-metric",
        args.early_stop_metric,
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--early-stop-min-delta",
        str(args.early_stop_min_delta),
        "--seed",
        str(args.seed),
    ]
    if args.device:
        device = str(args.device)
        if device.isdigit():
            device = f"cuda:{device}"
    command.extend(["--device", device])
    if args.allow_lexical_fallback:
        command.append("--allow-lexical-fallback")
    return command


def _best_models_complete(
    best_model_dir: Path, folds: List[int], expected_config_id: int
) -> bool:
    if not all((best_model_dir / f"fold_{fold}.pt").exists() for fold in folds):
        return False
    best_config_path = best_model_dir / "best_config.json"
    if not best_config_path.exists():
        return False
    try:
        saved = json.loads(best_config_path.read_text(encoding="utf-8"))
        return int(saved["config_id"]) == expected_config_id
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _save_best_models(
    best_row: Dict[str, object],
    config_output: Path,
    best_model_dir: Path,
    folds: List[int],
    parameter_names: List[str],
    ranking_field: str,
) -> None:
    best_config_id = int(best_row["config_id"])
    best_model_dir.mkdir(parents=True, exist_ok=True)
    staged_models = []
    for fold in folds:
        source_model = config_output / f"fold_{fold}" / "model.pt"
        if not source_model.exists():
            raise FileNotFoundError(
                f"Best configuration is missing its fold model: {source_model}"
            )
        staged_model = best_model_dir / f".fold_{fold}.pt.tmp"
        shutil.copy2(source_model, staged_model)
        staged_models.append((staged_model, best_model_dir / f"fold_{fold}.pt"))

    best_config = {
        "config_id": best_config_id,
        "selection_metric": ranking_field,
        "selection_value": float(best_row[ranking_field]),
        "folds": folds,
        "parameters": {name: best_row[name] for name in parameter_names},
    }
    staged_config = best_model_dir / ".best_config.json.tmp"
    staged_config.write_text(
        json.dumps(best_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for staged_model, final_model in staged_models:
        staged_model.replace(final_model)
    staged_config.replace(best_model_dir / "best_config.json")


def main() -> None:
    args = parse_args()
    folds = list(dict.fromkeys(args.folds))
    if any(fold not in range(1, 6) for fold in folds):
        raise SystemExit("--folds can only contain fold numbers in [1, 5]")

    teacher_root = Path(args.teacher_root) if args.teacher_root else (
        ROOT / "outputs" / "teacher" / args.split
    )
    teacher_files = {
        fold: teacher_root / f"fold_{fold}.npz" for fold in folds
    }
    missing_teacher_files = [
        str(path) for path in teacher_files.values() if not path.exists()
    ]
    if missing_teacher_files:
        raise SystemExit(
            "Missing teacher feature files:\n" + "\n".join(missing_teacher_files)
        )

    results_csv = Path(args.results_csv) if args.results_csv else (
        ROOT / "outputs" / "grid_search" / args.split / "five_fold_results.csv"
    )
    best_model_dir = Path(args.best_model_dir) if args.best_model_dir else (
        results_csv.parent / "best_models"
    )

    keys = [
        "hidden_dim",
        "learning_rate",
        "weight_decay",
        "dropout",
        "distill_weight",
        "temperature",
        "grad_clip",
    ]
    values = [
        args.hidden_dims,
        args.learning_rates,
        args.weight_decays,
        args.dropouts,
        args.distill_weights,
        args.temperatures,
        args.grad_clips,
    ]
    configurations = [
        {
            **dict(zip(keys, combination)),
            "cosine_weight": args.cosine_weight,
            "distill_warmup_ratio": args.distill_warmup_ratio,
            "distill_ramp_epochs": args.distill_ramp_epochs,
        }
        for combination in itertools.product(*values)
    ]
    parameter_names = list(configurations[0])
    total_runs = len(configurations) * len(folds)
    print(
        f"Grid configurations: {len(configurations)}, folds per configuration: "
        f"{len(folds)}, total training runs: {total_runs}"
    )
    print(f"All configuration results will be retained in: {results_csv}")
    print(
        "Only the best configuration's fold models will be retained in: "
        f"{best_model_dir}"
    )
    print(
        "WARNING: This search selects configurations using test-set metrics, so the reported "
        "test results are selection-biased."
    )

    ranking_field = f"{args.early_stop_metric}_mean"
    if args.restart:
        rows_by_id: Dict[int, Dict[str, object]] = {}
    else:
        rows_by_id = _read_existing_results(results_csv)
        _validate_existing_results(rows_by_id, configurations, parameter_names)

    successful_before_resume = [
        row for row in rows_by_id.values() if _row_succeeded(row, ranking_field)
    ]
    successful_before_resume.sort(
        key=lambda row: float(row[ranking_field]), reverse=True
    )
    current_best: Optional[Dict[str, object]] = (
        successful_before_resume[0] if successful_before_resume else None
    )
    rebuild_best_config_id: Optional[int] = None
    if current_best is not None:
        current_best_config_id = int(float(current_best["config_id"]))
    else:
        current_best_config_id = -1
    if current_best is not None and not _best_models_complete(
        best_model_dir, folds, current_best_config_id
    ):
        rebuild_best_config_id = current_best_config_id
        print(
            "Existing best-model files are incomplete; the best successful "
            f"configuration {rebuild_best_config_id} will be rerun to rebuild them."
        )

    skipped_count = sum(
        1
        for config_id, row in rows_by_id.items()
        if _row_succeeded(row, ranking_field) and config_id != rebuild_best_config_id
    )
    retry_count = sum(
        1 for row in rows_by_id.values() if not _row_succeeded(row, ranking_field)
    )
    missing_count = len(configurations) - len(rows_by_id)
    print(
        f"Resume summary: skip {skipped_count} successful configurations, "
        f"retry {retry_count} failed/incomplete configurations, "
        f"run {missing_count} configurations not yet recorded."
    )

    temporary_parent = None
    if args.temp_dir:
        temporary_parent = Path(args.temp_dir)
        temporary_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="llmhgsyn_grid_", dir=temporary_parent
    ) as temporary_root:
        temporary_root_path = Path(temporary_root)
        for config_id, config in enumerate(configurations, start=1):
            existing_row = rows_by_id.get(config_id)
            if (
                existing_row is not None
                and _row_succeeded(existing_row, ranking_field)
                and config_id != rebuild_best_config_id
            ):
                continue
            print(
                f"\nConfiguration [{config_id}/{len(configurations)}] "
                + " ".join(f"{key}={value}" for key, value in config.items()),
                flush=True,
            )
            metrics_by_fold: Dict[int, Dict[str, object]] = {}
            failed_exit_code = 0
            for fold in folds:
                trial_output = (
                    temporary_root_path
                    / f"config_{config_id:04d}"
                    / f"fold_{fold}"
                )
                print(f"  Running fold {fold}", flush=True)
                completed = subprocess.run(
                    _trial_command(
                        args,
                        fold,
                        teacher_files[fold],
                        trial_output,
                        config,
                    )
                )
                metrics_path = trial_output / "metrics.json"
                if completed.returncode != 0 or not metrics_path.exists():
                    failed_exit_code = completed.returncode or 1
                    print(
                        f"  Fold {fold} failed with exit code {failed_exit_code}; "
                        "skipping the remaining folds for this configuration.",
                        flush=True,
                    )
                    break
                metrics_by_fold[fold] = json.loads(
                    metrics_path.read_text(encoding="utf-8")
                )

            if len(metrics_by_fold) == len(folds):
                row = _aggregate_metrics(
                    config_id,
                    config,
                    args,
                    folds,
                    metrics_by_fold,
                )
            else:
                row = {
                    "selection_rank": "",
                    "config_id": config_id,
                    "status": "failed",
                    "exit_code": failed_exit_code,
                    "completed_folds": len(metrics_by_fold),
                    "split": args.split,
                    "folds": ",".join(map(str, folds)),
                    "seed": args.seed,
                    **config,
                    "early_stop_metric": args.early_stop_metric,
                }
            rows_by_id[config_id] = row

            if _row_succeeded(row, ranking_field):
                is_new_best = current_best is None or float(row[ranking_field]) > float(
                    current_best[ranking_field]
                )
                needs_rebuild = config_id == rebuild_best_config_id
                if is_new_best or needs_rebuild:
                    _save_best_models(
                        row,
                        temporary_root_path / f"config_{config_id:04d}",
                        best_model_dir,
                        folds,
                        parameter_names,
                        ranking_field,
                    )
                    if is_new_best:
                        current_best = row
                    rebuild_best_config_id = None

            _write_results(
                _ranked_rows(rows_by_id, ranking_field), results_csv, folds
            )
            shutil.rmtree(
                temporary_root_path / f"config_{config_id:04d}",
                ignore_errors=True,
            )

    ordered_rows = _ranked_rows(rows_by_id, ranking_field)
    successful = [row for row in ordered_rows if _row_succeeded(row, ranking_field)]
    failed = [row for row in ordered_rows if not _row_succeeded(row, ranking_field)]
    _write_results(ordered_rows, results_csv, folds)
    print(
        f"Grid search finished: {len(successful)} configurations succeeded, "
        f"{len(failed)} failed. Results: {results_csv}"
    )
    if successful:
        print(
            f"Best configuration: config_id={successful[0]['config_id']} "
            f"{ranking_field}={float(successful[0][ranking_field]):.6f}. "
            f"Models: {best_model_dir}"
        )


if __name__ == "__main__":
    main()
