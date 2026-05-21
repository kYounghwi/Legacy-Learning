from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Type

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from legacy_learning.datasets import LegacyManifestDataset
from legacy_learning.losses import ContentPreservationLoss
from models.adapters.base import BaseFFGAdapter


class LegacyLearningTrainer:
    def __init__(self, cfg: dict, adapter_cls: Type[BaseFFGAdapter]):
        self.cfg = cfg
        self.stage = cfg["run"]["stage"]
        self.device = self._resolve_device(cfg["run"].get("device", "auto"))
        self.output_dir = Path(cfg["run"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._set_seed(cfg["run"].get("seed", 123))
        self._save_config()

        self.dataset = LegacyManifestDataset(
            manifest_path=cfg["data"]["manifest"],
            image_size=int(cfg["data"].get("image_size", cfg["model"].get("image_size", 96))),
        )
        if len(self.dataset) == 0:
            raise ValueError(
                f"Dataset is empty: {cfg['data']['manifest']}. "
                "Fill the manifest before running training."
            )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=int(cfg["data"]["batch_size"]),
            shuffle=bool(cfg["data"].get("shuffle", True)),
            num_workers=int(cfg["data"].get("num_workers", 0)),
            drop_last=True,
        )

        self.adapter = adapter_cls(cfg)
        self.adapter.build()
        base_checkpoint = cfg.get("checkpoint", {}).get("base_checkpoint")
        if base_checkpoint:
            self.adapter.load_checkpoint(base_checkpoint)
        self.adapter.to(self.device)

        optim_cfg = cfg["optimization"]
        self.optimizer = torch.optim.AdamW(
            self.adapter.trainable_parameters(),
            lr=float(optim_cfg["learning_rate"]),
            betas=(float(optim_cfg["adam_beta1"]), float(optim_cfg["adam_beta2"])),
            eps=float(optim_cfg["adam_epsilon"]),
            weight_decay=float(optim_cfg["weight_decay"]),
        )

        legacy_cfg = cfg["legacy_learning"]
        self.content_loss = ContentPreservationLoss(
            alpha=float(legacy_cfg.get("alpha", 1.0)),
            beta_schedule=legacy_cfg.get("beta_schedule", "cosine"),
            fixed_beta=legacy_cfg.get("fixed_beta"),
            use_distance_term=bool(legacy_cfg.get("use_distance_term", True)),
            use_direction_term=bool(legacy_cfg.get("use_direction_term", True)),
        ).to(self.device)

    def fit(self) -> None:
        max_steps = int(self.cfg["run"]["max_steps"])
        log_interval = int(self.cfg["run"].get("log_interval", 100))
        ckpt_interval = int(self.cfg["run"].get("ckpt_interval", 2500))
        max_grad_norm = float(self.cfg["optimization"].get("max_grad_norm", 1.0))

        data_iter = iter(self.dataloader)
        progress = tqdm(range(1, max_steps + 1), desc=f"{self.cfg['model']['name']}:{self.stage}")

        for global_step in progress:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            batch = self._move_to_device(batch)
            self.adapter.train()
            self.optimizer.zero_grad(set_to_none=True)

            outputs = self.adapter.forward_train(batch=batch, global_step=global_step, stage=self.stage)
            task_loss, task_metrics = self.adapter.task_loss(outputs=outputs, batch=batch)

            preservation_loss = None
            preservation_metrics = {}
            use_preservation = (
                self.stage == "legacy"
                and bool(self.cfg["legacy_learning"].get("enabled", True))
                and float(self.cfg["legacy_learning"].get("preservation_weight", 0.0)) > 0.0
            )

            if use_preservation:
                generated = self.adapter.reconstruct_image(outputs=outputs, batch=batch)
                preservation_loss, preservation_metrics = self.content_loss(
                    generated=generated,
                    reference=batch["reference_image"],
                    legacy=batch["legacy_image"],
                    step=global_step,
                    max_steps=max_steps,
                )

            if preservation_loss is not None and bool(
                self.cfg["legacy_learning"].get("content_preservation_content_encoder_only", True)
            ):
                task_loss.backward(retain_graph=True)
                with self.adapter.content_preservation_scope():
                    weighted_preservation = (
                        float(self.cfg["legacy_learning"]["preservation_weight"]) * preservation_loss
                    )
                    weighted_preservation.backward()
                total_loss_value = float((task_loss.detach() + weighted_preservation.detach()).cpu())
            else:
                total_loss = task_loss
                if preservation_loss is not None:
                    total_loss = total_loss + float(self.cfg["legacy_learning"]["preservation_weight"]) * preservation_loss
                total_loss.backward()
                total_loss_value = float(total_loss.detach().cpu())

            torch.nn.utils.clip_grad_norm_(self.adapter.trainable_parameters(), max_grad_norm)
            self.optimizer.step()

            metrics = {
                "loss": total_loss_value,
                **task_metrics,
                **preservation_metrics,
            }
            progress.set_postfix({k: f"{v:.4f}" for k, v in metrics.items() if isinstance(v, float)})

            if global_step % log_interval == 0:
                self._append_jsonl("train_metrics.jsonl", {"step": global_step, **metrics})

            if global_step % ckpt_interval == 0 or global_step == max_steps:
                ckpt_dir = self.output_dir / f"global_step_{global_step}"
                self.adapter.save_checkpoint(ckpt_dir)

    def _resolve_device(self, value: str) -> torch.device:
        if value == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(value)

    def _move_to_device(self, batch):
        if torch.is_tensor(batch):
            return batch.to(self.device)
        if isinstance(batch, dict):
            return {key: self._move_to_device(value) for key, value in batch.items()}
        if isinstance(batch, list):
            return [self._move_to_device(value) for value in batch]
        return batch

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _save_config(self) -> None:
        with (self.output_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
            json.dump(self.cfg, f, indent=2, ensure_ascii=False)

    def _append_jsonl(self, filename: str, payload: dict) -> None:
        with (self.output_dir / filename).open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
