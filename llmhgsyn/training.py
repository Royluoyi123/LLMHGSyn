from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict

import numpy as np

from utils import get_roc_score


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    auc, aupr, acc, kappa, bacc, f1, mcc, _, _ = get_roc_score(labels, probabilities)
    return {
        "roc_auc": float(auc),
        "aupr": float(aupr),
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "f1": float(f1),
        "mcc": float(mcc),
        "kappa": float(kappa),
    }


def _active_distill_scales(value: str, num_scales: int):
    if str(value).strip().lower() == "all":
        return list(range(num_scales))
    try:
        scales = sorted(
            {int(item.strip()) for item in str(value).split(",") if item.strip()}
        )
    except ValueError as exc:
        raise ValueError(f"Invalid distill_scales value: {value!r}") from exc
    if not scales or any(scale < 0 or scale >= num_scales for scale in scales):
        raise ValueError(
            f"distill_scales must contain indices in [0, {num_scales - 1}]"
        )
    return scales


def _prediction_states(model, states, fusion_mode: str):
    if fusion_mode == "last":
        return states[-1]
    return model.fuse_states(states)


def train_synergy_model(args) -> Dict[str, object]:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise SystemExit(f"PyTorch is required for training: {exc}. Install requirements.txt")

    from .data import dataset_summary, load_fold
    from .hypergraph import graph_fingerprint
    from .losses import multi_scale_info_nce
    from .models import LLMHGNSynergy, MultiScaleProjector
    from .molecules import ATOM_DIM, BOND_DIM, load_or_build_molecular_graphs

    seed_everything(args.seed)
    early_stop_source = getattr(args, "early_stop_source", "test")
    if early_stop_source not in {"train", "test"}:
        raise ValueError("early_stop_source must be either 'train' or 'test'")
    if args.early_stop_patience < 1:
        raise ValueError("early_stop_patience must be at least 1")
    if args.early_stop_min_delta < 0:
        raise ValueError("early_stop_min_delta must be non-negative")
    if args.log_every < 1:
        raise ValueError("log_every must be at least 1")
    if not 0.0 <= args.distill_warmup_ratio < 1.0:
        raise ValueError("distill_warmup_ratio must be in [0, 1)")
    if args.distill_ramp_epochs < 1:
        raise ValueError("distill_ramp_epochs must be at least 1")
    if args.cosine_weight < 0:
        raise ValueError("cosine_weight must be non-negative")
    if args.distill_loss_mode == "cosine_only" and args.cosine_weight <= 0:
        raise ValueError("cosine_only requires cosine_weight to be greater than zero")
    if early_stop_source == "test":
        print(
            "WARNING: Test-set metrics are used for checkpoint selection and early stopping. "
            "The resulting test scores are selection-biased and are not an unbiased final evaluation."
        )
    else:
        print(
            "Checkpoint selection and early stopping use training-set metrics. "
            "Test metrics are monitored but do not affect model selection."
        )
    fold_data = load_fold(
        Path(args.data_root) / args.dataset,
        args.split,
        args.fold,
    )
    summary = dataset_summary(fold_data)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    teacher_file = np.load(args.teacher_features, allow_pickle=False)
    teacher_np = teacher_file["teacher_features"].astype(np.float32)
    if "layer_pooling" in teacher_file.files:
        print(f"Teacher layer pooling: {str(teacher_file['layer_pooling'].item())}")
    else:
        print(
            "WARNING: Teacher file has no layer_pooling metadata; rerun extract_teacher.py to use last4_mean."
        )
    if "graph_id" in teacher_file.files:
        expected_graph_id = graph_fingerprint(fold_data.incidence)
        supplied_graph_id = str(teacher_file["graph_id"].item())
        if supplied_graph_id != expected_graph_id:
            raise ValueError(
                "Teacher features were extracted from a different fold/split hypergraph"
            )
    if "node_names" in teacher_file.files:
        expected_names = fold_data.catalog.drug_names + fold_data.catalog.cell_names
        if teacher_file["node_names"].astype(str).tolist() != expected_names:
            raise ValueError("Teacher node order does not match the current node catalog")
    expected_scales = args.hgnn_layers + 1
    active_distill_scales = _active_distill_scales(
        args.distill_scales, expected_scales
    )
    contrastive_weight = (
        0.0 if args.distill_loss_mode == "cosine_only" else 1.0
    )
    active_cosine_weight = (
        0.0 if args.distill_loss_mode == "infonce_only" else args.cosine_weight
    )
    print(
        f"Ablation controls: distill_scales={active_distill_scales}, "
        f"loss_mode={args.distill_loss_mode}, student_fusion={args.student_fusion}"
    )
    expected_shape = (expected_scales, fold_data.catalog.num_nodes)
    if teacher_np.shape[:2] != expected_shape:
        raise ValueError(
            f"Teacher shape {teacher_np.shape} is incompatible with expected "
            f"[scales={expected_scales}, nodes={fold_data.catalog.num_nodes}, dim]"
        )

    atom_x, molecule_edge_index, molecule_edge_attr, molecule_batch = (
        load_or_build_molecular_graphs(
            fold_data.catalog.smiles,
            args.molecule_cache,
            allow_lexical_fallback=args.allow_lexical_fallback,
        )
    ) #字符串的
    expression = fold_data.catalog.expression.copy()
    expression = (expression - expression.mean(axis=0, keepdims=True)) / (
        expression.std(axis=0, keepdims=True) + 1e-6
    ) #标准化

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    tensors = {
        "atom_x": torch.from_numpy(atom_x).to(device),
        "molecule_edge_index": torch.from_numpy(molecule_edge_index).long().to(device),
        "molecule_edge_attr": torch.from_numpy(molecule_edge_attr).to(device),
        "molecule_batch": torch.from_numpy(molecule_batch).long().to(device),
        "expression": torch.from_numpy(expression).to(device),
        "incidence": torch.from_numpy(fold_data.incidence).long().to(device),
        "train": torch.from_numpy(fold_data.train[:, :3]).long().to(device),
        "train_labels": torch.from_numpy(fold_data.train[:, 3]).float().to(device),
        "test": torch.from_numpy(fold_data.test[:, :3]).long().to(device),
        "teacher": torch.from_numpy(teacher_np).to(device),
    }
    node_types = torch.cat(
        [
            torch.zeros(fold_data.catalog.num_drugs, dtype=torch.long),
            torch.ones(fold_data.catalog.num_cells, dtype=torch.long),
        ]
    ).to(device)

    node_degree = torch.bincount(
        tensors["incidence"][0],
        minlength=fold_data.catalog.num_nodes,
    )
    distill_node_mask = torch.ones(
        (expected_scales, fold_data.catalog.num_nodes),
        dtype=torch.bool,
        device=device,
    )
    if expected_scales > 1:
        distill_node_mask[1:] = node_degree.gt(0).unsqueeze(0).expand(
            expected_scales - 1, -1
        )

    model = LLMHGNSynergy(
        atom_dim=ATOM_DIM,
        bond_dim=BOND_DIM,
        expression_dim=expression.shape[1],
        hidden_dim=args.hidden_dim,
        hgnn_layers=args.hgnn_layers,
        dropout=args.dropout,
    ).to(device)
    projector = MultiScaleProjector(
        teacher_dim=teacher_np.shape[-1],
        student_dim=args.hidden_dim,
        num_scales=expected_scales,
        num_drugs=fold_data.catalog.num_drugs,
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(projector.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    positives = tensors["train_labels"].sum().clamp_min(1.0)
    negatives = tensors["train_labels"].numel() - positives
    pos_weight = (negatives / positives).detach()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "best_model.pt"
    history_path = output_dir / "history.json"
    best_metric = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    final_loss = 0.0
    epochs_ran = 0
    if args.distill_weight > 0:
        warmup_epochs = min(args.epochs - 1, int(round(args.epochs * args.distill_warmup_ratio)))
        ramp_epochs = max(1, min(args.distill_ramp_epochs, args.epochs - warmup_epochs))
        selection_start_epoch = warmup_epochs + ramp_epochs
    else:
        warmup_epochs = 0
        ramp_epochs = 0
        selection_start_epoch = 1
    print(f"Distillation schedule: warmup={warmup_epochs}, ramp={ramp_epochs}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        projector.train()
        optimizer.zero_grad(set_to_none=True)
        student_states = model.encode_nodes(
            tensors["atom_x"],
            tensors["molecule_edge_index"],
            tensors["molecule_edge_attr"],
            tensors["molecule_batch"],
            tensors["expression"],
            tensors["incidence"],
        )
        fused_student_states = _prediction_states(
            model, student_states, args.student_fusion
        )
        logits = model.predict(fused_student_states, tensors["train"])
        prediction_loss = F.binary_cross_entropy_with_logits(
            logits, tensors["train_labels"], pos_weight=pos_weight
        )
        if args.distill_weight <= 0 or epoch <= warmup_epochs:
            current_distill_weight = 0.0
        else:
            ramp_progress = (epoch - warmup_epochs) / max(1, ramp_epochs)
            current_distill_weight = args.distill_weight * min(ramp_progress, 1.0)
        if current_distill_weight > 0:
            projected_teacher = projector(tensors["teacher"])
            distill_loss = multi_scale_info_nce(
                [student_states[index] for index in active_distill_scales],
                [projected_teacher[index] for index in active_distill_scales],
                node_types,
                projector.scale_logits[active_distill_scales],
                contrastive_weight=contrastive_weight,
                cosine_weight=active_cosine_weight,
                temperature=args.temperature,
                scale_node_mask=distill_node_mask[active_distill_scales],
            )
        else:
            distill_loss = prediction_loss.new_zeros(())
        loss = prediction_loss + current_distill_weight * distill_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(projector.parameters()), args.grad_clip
        )
        optimizer.step()
        final_loss = float(loss.detach().cpu())

        model.eval()
        projector.eval()
        with torch.inference_mode():
            evaluation_states = model.encode_nodes(
                tensors["atom_x"],
                tensors["molecule_edge_index"],
                tensors["molecule_edge_attr"],
                tensors["molecule_batch"],
                tensors["expression"],
                tensors["incidence"],
            )
            fused_evaluation_states = _prediction_states(
                model, evaluation_states, args.student_fusion
            )
            train_probabilities = torch.sigmoid(
                model.predict(fused_evaluation_states, tensors["train"])
            ).cpu().numpy()
            test_probabilities = torch.sigmoid(
                model.predict(fused_evaluation_states, tensors["test"])
            ).cpu().numpy()
        train_metrics = binary_metrics(fold_data.train[:, 3], train_probabilities)
        test_metrics = binary_metrics(fold_data.test[:, 3], test_probabilities)
        active_scale_weights = torch.softmax(
            projector.scale_logits.detach()[active_distill_scales], dim=0
        ).cpu().tolist()
        distill_scale_weights = [0.0] * expected_scales
        for scale, weight in zip(active_distill_scales, active_scale_weights):
            distill_scale_weights[scale] = float(weight)
        if args.student_fusion == "last":
            fusion_weights = [0.0] * (expected_scales - 1) + [1.0]
        else:
            fusion_weights = torch.softmax(
                model.fusion_logits.detach(), dim=0
            ).cpu().tolist()
        selection_metrics = (
            train_metrics if early_stop_source == "train" else test_metrics
        )
        selected_value = float(selection_metrics[args.early_stop_metric])
        comparable_value = selected_value if np.isfinite(selected_value) else float("-inf")
        selection_active = (
            args.distill_weight <= 0 or epoch >= selection_start_epoch
        )
        improved = selection_active and comparable_value > best_metric + args.early_stop_min_delta
        if improved:
            best_metric = comparable_value
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model": {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    },
                    "projector": {
                        name: value.detach().cpu().clone()
                        for name, value in projector.state_dict().items()
                    },
                    "epoch": epoch,
                    "selection_source": early_stop_source,
                    "selection_metric": args.early_stop_metric,
                    "selection_value": selected_value,
                    "train_objective": final_loss,
                    "config": vars(args),
                    "dataset_summary": summary,
                },
                best_model_path,
            )
        elif selection_active:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "total_loss": final_loss,
                "prediction_loss": float(prediction_loss.detach().cpu()),
                "distill_loss": float(distill_loss.detach().cpu()),
                "current_distill_weight": float(current_distill_weight),
                "distill_scale_weights": [float(value) for value in distill_scale_weights],
                "fusion_weights": [float(value) for value in fusion_weights],
                "selection_active": selection_active,
                "selection_source": early_stop_source,
                "train": train_metrics,
                "test": test_metrics,
                "selected_as_best": improved,
            }
        )
        history_path.write_text(
            json.dumps(history, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        epochs_ran = epoch
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:04d} total={final_loss:.5f} "
                f"prediction={prediction_loss.item():.5f} distill={distill_loss.item():.5f} "
                f"kd_weight={current_distill_weight:.5f} "
                f"distill_scales={[round(value, 4) for value in distill_scale_weights]} "
                f"fusion_scales={[round(value, 4) for value in fusion_weights]}"
            )
            print(
                "  train "
                f"auc={train_metrics['roc_auc']:.5f} aupr={train_metrics['aupr']:.5f} "
                f"f1={train_metrics['f1']:.5f} mcc={train_metrics['mcc']:.5f} "
                f"kappa={train_metrics['kappa']:.5f}"
            )
            print(
                "  test  "
                f"auc={test_metrics['roc_auc']:.5f} aupr={test_metrics['aupr']:.5f} "
                f"f1={test_metrics['f1']:.5f} mcc={test_metrics['mcc']:.5f} "
                f"kappa={test_metrics['kappa']:.5f} "
                f"best_{early_stop_source}_{args.early_stop_metric}="
                f"{best_metric:.5f}@{best_epoch}"
            )
        if epochs_without_improvement >= args.early_stop_patience:
            print(
                f"Early stopping at epoch {epoch}: {early_stop_source} "
                f"{args.early_stop_metric} "
                f"did not improve for {args.early_stop_patience} epochs."
            )
            break

    if best_epoch == 0:
        raise RuntimeError(
            f"No finite {early_stop_source} {args.early_stop_metric} was available "
            "for checkpoint selection"
        )
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    projector.load_state_dict(checkpoint["projector"])
    model.eval()
    projector.eval()
    with torch.inference_mode():
        states = model.encode_nodes(
            tensors["atom_x"],
            tensors["molecule_edge_index"],
            tensors["molecule_edge_attr"],
            tensors["molecule_batch"],
            tensors["expression"],
            tensors["incidence"],
        )
        fused_states = _prediction_states(model, states, args.student_fusion)
        train_probabilities = torch.sigmoid(
            model.predict(fused_states, tensors["train"])
        ).cpu().numpy()
        test_probabilities = torch.sigmoid(
            model.predict(fused_states, tensors["test"])
        ).cpu().numpy()
    final_train_metrics = binary_metrics(fold_data.train[:, 3], train_probabilities)
    final_test_metrics = binary_metrics(fold_data.test[:, 3], test_probabilities)
    metrics = {
        "selection_protocol": (
            f"{early_stop_source}_set_used_for_early_stopping_and_checkpoint_selection"
        ),
        "selection_warning": (
            "Test metrics are selection-biased because the test set was used for model selection."
            if early_stop_source == "test"
            else "Training metrics were used for model selection; no validation set was used."
        ),
        "early_stop_source": early_stop_source,
        "early_stop_metric": args.early_stop_metric,
        "best_epoch": best_epoch,
        "epochs_ran": epochs_ran,
        "best_selection_metric": best_metric,
        "best_test_metric": float(final_test_metrics[args.early_stop_metric]),
        "best_epoch_train_objective": float(checkpoint["train_objective"]),
        "last_epoch_train_objective": final_loss,
        "train": final_train_metrics,
        "test": final_test_metrics,
    }

    best_metrics_row = {
        "split": args.split,
        "fold": args.fold,
        "best_epoch": best_epoch,
        "epochs_ran": epochs_ran,
        "early_stop_source": early_stop_source,
        "early_stop_metric": args.early_stop_metric,
        "best_selection_metric": best_metric,
        "best_test_metric": float(final_test_metrics[args.early_stop_metric]),
        **{
            f"train_{name}": value
            for name, value in metrics["train"].items()
        },
        **{
            f"test_{name}": value
            for name, value in metrics["test"].items()
        },
    }

    torch.save(
        {
            "model": model.state_dict(),
            "projector": projector.state_dict(),
            "config": vars(args),
            "dataset_summary": summary,
            "metrics": metrics,
        },
        output_dir / "model.pt",
    )
    np.save(output_dir / "train_probabilities.npy", train_probabilities)
    np.save(output_dir / "test_probabilities.npy", test_probabilities)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    with (output_dir / "best_metrics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(best_metrics_row),
        )
        writer.writeheader()
        writer.writerow(best_metrics_row)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics
