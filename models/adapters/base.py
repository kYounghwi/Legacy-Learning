from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn


class BaseFFGAdapter(nn.Module):
    """Interface between the common legacy-learning trainer and an FFG model."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

    def build(self) -> None:
        raise NotImplementedError

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        raise NotImplementedError

    def save_checkpoint(self, checkpoint_dir: str | Path) -> None:
        raise NotImplementedError

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return self.parameters()

    def forward_train(self, batch: Dict[str, torch.Tensor], global_step: int, stage: str) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def task_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        raise NotImplementedError

    def reconstruct_image(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def build_sampler(self, checkpoint_path: str | Path, device: torch.device | str) -> None:
        raise NotImplementedError

    def sample_image(
        self,
        content_image_path: str | Path,
        style_image_path: str | Path,
        sampling_cfg: dict,
        seed: int | None = None,
    ):
        raise NotImplementedError

    @contextmanager
    def content_preservation_scope(self):
        yield
