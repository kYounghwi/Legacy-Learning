from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from legacy_learning.schedules import get_beta


def to_unit_range(images: torch.Tensor) -> torch.Tensor:
    if images.detach().min() < -0.05:
        return (images + 1.0) / 2.0
    return images.clamp(0.0, 1.0)


def to_gray(images: torch.Tensor) -> torch.Tensor:
    images = to_unit_range(images)
    if images.shape[1] == 1:
        return images
    r, g, b = images[:, 0:1], images[:, 1:2], images[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


class SobelEdgeExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        kernel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)

    def gradients(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gray = to_gray(images)
        grad_x = F.conv2d(gray, self.kernel_x, padding=1)
        grad_y = F.conv2d(gray, self.kernel_y, padding=1)
        return grad_x, grad_y

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        grad_x, grad_y = self.gradients(images)
        magnitude = torch.sqrt(grad_x.square() + grad_y.square() + 1e-8)
        return magnitude / (magnitude.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6))


class ContentPreservationLoss(nn.Module):
    """Edge-map implementation of the paper's content preservation loss."""

    def __init__(
        self,
        alpha: float = 1.0,
        beta_schedule: str = "cosine",
        fixed_beta: float | None = None,
        use_distance_term: bool = True,
        use_direction_term: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta_schedule = beta_schedule
        self.fixed_beta = fixed_beta
        self.use_distance_term = use_distance_term
        self.use_direction_term = use_direction_term
        self.edge = SobelEdgeExtractor()

    def forward(
        self,
        generated: torch.Tensor,
        reference: torch.Tensor,
        legacy: torch.Tensor,
        step: int,
        max_steps: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        beta = get_beta(
            step=step,
            max_steps=max_steps,
            schedule=self.beta_schedule,
            fixed_beta=self.fixed_beta,
        )

        reference_edge = self.edge(reference)
        legacy_edge = self.edge(legacy)
        mixed_edge = (1.0 - beta) * reference_edge + beta * legacy_edge
        generated_edge = self.edge(generated)

        loss = generated_edge.new_tensor(0.0)
        distance_loss = generated_edge.new_tensor(0.0)
        direction_loss = generated_edge.new_tensor(0.0)

        if self.use_distance_term:
            distance_loss = F.l1_loss(generated_edge, mixed_edge)
            loss = loss + distance_loss

        if self.use_direction_term:
            gen_dx, gen_dy = self.edge.gradients(generated_edge)
            mix_dx, mix_dy = self.edge.gradients(mixed_edge)
            gen_vec = torch.cat([gen_dx, gen_dy], dim=1)
            mix_vec = torch.cat([mix_dx, mix_dy], dim=1)
            cosine = F.cosine_similarity(gen_vec, mix_vec, dim=1, eps=1e-6)
            direction_loss = (1.0 - cosine).mean()
            loss = loss + self.alpha * direction_loss

        details = {
            "beta": float(beta),
            "content_distance": float(distance_loss.detach().cpu()),
            "content_direction": float(direction_loss.detach().cpu()),
        }
        return loss, details
