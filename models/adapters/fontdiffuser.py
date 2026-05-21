from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.adapters.base import BaseFFGAdapter


class FontDiffuserAdapter(BaseFFGAdapter):
    """Adapter for the original FontDiffuser repository.

    This class keeps FontDiffuser-specific diffusion logic out of the common
    legacy-learning trainer. Point `model.repo_path` to a FontDiffuser checkout.
    """

    def build(self) -> None:
        self._setup_repo()
        from src import FontDiffuserModel, build_content_encoder, build_ddpm_scheduler, build_style_encoder, build_unet

        self.fd_args = self._make_fontdiffuser_args()
        unet = build_unet(self.fd_args)
        style_encoder = build_style_encoder(self.fd_args)
        content_encoder = build_content_encoder(self.fd_args)
        self.noise_scheduler = build_ddpm_scheduler(self.fd_args)
        self.model = FontDiffuserModel(
            unet=unet,
            style_encoder=style_encoder,
            content_encoder=content_encoder,
        )

    def _setup_repo(self) -> None:
        repo_path = self.cfg["model"].get("repo_path")
        if not repo_path:
            raise ValueError("model.repo_path must point to a FontDiffuser repository.")

        repo = Path(repo_path).resolve()
        if not repo.exists():
            raise FileNotFoundError(f"FontDiffuser repo not found: {repo}")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

    def _make_fontdiffuser_args(self) -> SimpleNamespace:
        model_cfg = self.cfg["model"]
        image_size = int(model_cfg.get("image_size", 96))
        return SimpleNamespace(
            resolution=image_size,
            unet_channels=tuple(model_cfg.get("unet_channels", [64, 128, 256, 512])),
            style_image_size=(image_size, image_size),
            content_image_size=(image_size, image_size),
            content_encoder_downsample_size=int(model_cfg.get("content_encoder_downsample_size", 3)),
            channel_attn=bool(model_cfg.get("channel_attn", True)),
            content_start_channel=int(model_cfg.get("content_start_channel", 64)),
            style_start_channel=int(model_cfg.get("style_start_channel", 64)),
            beta_scheduler=model_cfg.get("beta_scheduler", "scaled_linear"),
        )

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint = Path(checkpoint_path)
        map_location = "cpu"
        if checkpoint.is_dir():
            self._load_component_checkpoint(
                checkpoint=checkpoint,
                unet=self.model.unet,
                style_encoder=self.model.style_encoder,
                content_encoder=self.model.content_encoder,
                map_location=map_location,
            )
            return

        loaded = torch.load(checkpoint, map_location=map_location)
        if hasattr(loaded, "state_dict"):
            self.model.load_state_dict(loaded.state_dict())
        elif isinstance(loaded, dict):
            self.model.load_state_dict(loaded)
        else:
            raise ValueError(f"Unsupported checkpoint format: {checkpoint}")

    def save_checkpoint(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.unet.state_dict(), checkpoint_dir / "unet.pth")
        torch.save(self.model.style_encoder.state_dict(), checkpoint_dir / "style_encoder.pth")
        torch.save(self.model.content_encoder.state_dict(), checkpoint_dir / "content_encoder.pth")
        torch.save(self.model.state_dict(), checkpoint_dir / "fontdiffuser_state_dict.pth")

    def build_sampler(self, checkpoint_path: str | Path, device: torch.device | str) -> None:
        self._setup_repo()
        from src import (
            FontDiffuserDPMPipeline,
            FontDiffuserModelDPM,
            build_content_encoder,
            build_ddpm_scheduler,
            build_style_encoder,
            build_unet,
        )

        self.fd_args = self._make_fontdiffuser_args()
        unet = build_unet(self.fd_args)
        style_encoder = build_style_encoder(self.fd_args)
        content_encoder = build_content_encoder(self.fd_args)
        self._load_component_checkpoint(
            checkpoint=Path(checkpoint_path),
            unet=unet,
            style_encoder=style_encoder,
            content_encoder=content_encoder,
            map_location="cpu",
        )

        self.sampling_device = torch.device(device)
        model = FontDiffuserModelDPM(
            unet=unet,
            style_encoder=style_encoder,
            content_encoder=content_encoder,
        )
        model.to(self.sampling_device)
        model.eval()

        sampling_cfg = self.cfg.get("sampling", {})
        train_scheduler = build_ddpm_scheduler(self.fd_args)
        self.sampling_pipe = FontDiffuserDPMPipeline(
            model=model,
            ddpm_train_scheduler=train_scheduler,
            model_type=sampling_cfg.get("model_type", "noise"),
            guidance_type=sampling_cfg.get("guidance_type", "classifier-free"),
            guidance_scale=float(sampling_cfg.get("guidance_scale", 7.5)),
        )

    def sample_image(
        self,
        content_image_path: str | Path,
        style_image_path: str | Path,
        sampling_cfg: dict,
        seed: int | None = None,
    ):
        if not hasattr(self, "sampling_pipe"):
            raise RuntimeError("Call build_sampler() before sample_image().")

        image_size = int(self.cfg["model"].get("image_size", 96))
        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        content_image = transform(Image.open(content_image_path).convert("RGB"))[None, :].to(self.sampling_device)
        style_image = transform(Image.open(style_image_path).convert("RGB"))[None, :].to(self.sampling_device)

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))

        with torch.no_grad():
            images = self.sampling_pipe.generate(
                content_images=content_image,
                style_images=style_image,
                batch_size=1,
                order=int(sampling_cfg.get("order", 2)),
                num_inference_step=int(sampling_cfg.get("num_inference_steps", 20)),
                content_encoder_downsample_size=self.fd_args.content_encoder_downsample_size,
                t_start=sampling_cfg.get("t_start"),
                t_end=sampling_cfg.get("t_end"),
                dm_size=(image_size, image_size),
                algorithm_type=sampling_cfg.get("algorithm_type", "dpmsolver++"),
                skip_type=sampling_cfg.get("skip_type", "time_uniform"),
                method=sampling_cfg.get("method", "multistep"),
                correcting_x0_fn=sampling_cfg.get("correcting_x0_fn"),
                generator=generator,
            )
        return images[0]

    def _load_component_checkpoint(self, checkpoint: Path, unet, style_encoder, content_encoder, map_location) -> None:
        if not checkpoint.is_dir():
            raise ValueError(
                "FontDiffuser sampling/training expects a checkpoint directory containing "
                "unet.pth, style_encoder.pth, and content_encoder.pth."
            )
        unet.load_state_dict(torch.load(checkpoint / "unet.pth", map_location=map_location))
        style_encoder.load_state_dict(torch.load(checkpoint / "style_encoder.pth", map_location=map_location))
        content_encoder.load_state_dict(torch.load(checkpoint / "content_encoder.pth", map_location=map_location))

    def forward_train(self, batch: Dict[str, torch.Tensor], global_step: int, stage: str) -> Dict[str, torch.Tensor]:
        target_images = batch["target_image"]
        content_images = batch["content_image"]
        style_images = batch["style_image"]

        noise = torch.randn_like(target_images)
        batch_size = target_images.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.num_train_timesteps,
            (batch_size,),
            device=target_images.device,
        ).long()
        noisy_target_images = self.noise_scheduler.add_noise(target_images, noise, timesteps)

        drop_prob = float(self.cfg["model"].get("drop_prob", 0.0))
        if drop_prob > 0:
            mask = torch.rand(batch_size, device=target_images.device) < drop_prob
            if mask.any():
                content_images = content_images.clone()
                style_images = style_images.clone()
                content_images[mask] = 1.0
                style_images[mask] = 1.0

        noise_pred, offset_out_sum = self.model(
            x_t=noisy_target_images,
            timesteps=timesteps,
            style_images=style_images,
            content_images=content_images,
            content_encoder_downsample_size=self.fd_args.content_encoder_downsample_size,
        )

        pred_original_sample = self._x0_from_epsilon(
            noise_pred=noise_pred,
            x_t=noisy_target_images,
            timesteps=timesteps,
        )

        return {
            "noise_pred": noise_pred,
            "noise": noise,
            "offset_out_sum": offset_out_sum,
            "pred_original_sample": pred_original_sample,
            "timesteps": timesteps,
        }

    def task_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        diff_loss = F.mse_loss(outputs["noise_pred"].float(), outputs["noise"].float(), reduction="mean")
        offset = outputs["offset_out_sum"]
        offset_loss = offset / 2.0 if torch.is_tensor(offset) else diff_loss.new_tensor(float(offset) / 2.0)
        offset_weight = float(self.cfg["model"].get("offset_weight", 0.5))
        loss = diff_loss + offset_weight * offset_loss
        metrics = {
            "loss_task": float(loss.detach().cpu()),
            "loss_diff": float(diff_loss.detach().cpu()),
            "loss_offset": float(offset_loss.detach().cpu()),
        }
        return loss, metrics

    def reconstruct_image(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return outputs["pred_original_sample"].clamp(-1.0, 1.0)

    def _x0_from_epsilon(self, noise_pred: torch.Tensor, x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        pred_original = []
        for i in range(noise_pred.shape[0]):
            pred_i = self.noise_scheduler.step(
                model_output=noise_pred[i : i + 1],
                timestep=timesteps[i],
                sample=x_t[i : i + 1],
                generator=None,
                return_dict=True,
            ).pred_original_sample
            pred_original.append(pred_i)
        return torch.cat(pred_original, dim=0)

    @contextmanager
    def content_preservation_scope(self):
        previous = {param: param.requires_grad for param in self.parameters()}
        for param in self.parameters():
            param.requires_grad_(False)
        for param in self.model.content_encoder.parameters():
            param.requires_grad_(True)
        try:
            yield
        finally:
            for param, requires_grad in previous.items():
                param.requires_grad_(requires_grad)
