from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F


def multi_scale_info_nce(
    student_states: List[torch.Tensor],
    projected_teacher_states: List[torch.Tensor],
    node_types: torch.Tensor,
    scale_logits: torch.Tensor,
    contrastive_weight: float = 1.0,
    cosine_weight: float = 0.1,
    temperature: float = 0.07,
    scale_node_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if contrastive_weight < 0 or cosine_weight < 0:
        raise ValueError("Distillation loss weights must be non-negative")
    if contrastive_weight == 0 and cosine_weight == 0:
        raise ValueError("At least one distillation loss component must be active")
    if len(student_states) != len(projected_teacher_states):
        raise ValueError("Student HGNN states and teacher scales must have the same length")
    if scale_node_mask is not None and scale_node_mask.shape != (
        len(student_states),
        node_types.numel(),
    ):
        raise ValueError(
            "scale_node_mask must have shape [num_scales, num_nodes]"
        )
    weights = F.softmax(scale_logits, dim=0)
    total = student_states[0].new_zeros(())
    for scale, (student, teacher) in enumerate(zip(student_states, projected_teacher_states)):
        if scale_node_mask is None:
            valid_nodes = torch.ones_like(node_types, dtype=torch.bool)
        else:
            valid_nodes = scale_node_mask[scale].bool()
        if not valid_nodes.any():
            continue

        student = student[valid_nodes]
        teacher = teacher[valid_nodes]
        current_node_types = node_types[valid_nodes]
        student = F.normalize(student, dim=-1)
        teacher = F.normalize(teacher, dim=-1)
        cosine_loss = (1.0 - (student * teacher).sum(dim=-1)).mean()
        loss = cosine_weight * cosine_loss
        if contrastive_weight > 0:
            logits = student @ teacher.T / temperature
            same_type = current_node_types[:, None].eq(current_node_types[None, :])
            logits = logits.masked_fill(~same_type, torch.finfo(logits.dtype).min)
            labels = torch.arange(student.size(0), device=student.device)
            # Bidirectional alignment prevents the projector from only fitting one direction.
            contrastive_loss = 0.5 * (
                F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
            )
            loss = loss + contrastive_weight * contrastive_loss
        total = total + weights[scale] * loss
    return total
