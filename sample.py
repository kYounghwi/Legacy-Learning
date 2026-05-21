from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw

from legacy_learning.config import load_yaml_config
from models.adapters import get_adapter_class


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Sample glyphs from a legacy-learning FFG checkpoint.")
    parser.add_argument("--config", type=str, default="configs/fontdiffuser_legacy.yaml")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model_repo", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--content_image", type=str, default=None)
    parser.add_argument("--content_dir", type=str, default=None)
    parser.add_argument("--style_image", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/samples")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--save_triptych", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def iter_content_images(content_image: str | None, content_dir: str | None) -> Iterable[Path]:
    if content_image:
        yield Path(content_image)
        return
    if not content_dir:
        raise ValueError("Provide either --content_image or --content_dir.")

    root = Path(content_dir)
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def resolve_device(value: str) -> str:
    if value == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def save_triptych(content_path: Path, style_path: Path, output_image: Image.Image, save_path: Path, image_size: int) -> None:
    content = Image.open(content_path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    style = Image.open(style_path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    output = output_image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)

    label_h = 18
    canvas = Image.new("RGB", (image_size * 3, image_size + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, image) in enumerate((("content", content), ("style", style), ("output", output))):
        x = idx * image_size
        canvas.paste(image, (x, label_h))
        draw.text((x + 4, 2), label, fill=(0, 0, 0))
    canvas.save(save_path)


def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)

    if args.model is not None:
        cfg["model"]["name"] = args.model
    if args.model_repo is not None:
        cfg["model"]["repo_path"] = args.model_repo
    if args.num_inference_steps is not None:
        cfg.setdefault("sampling", {})["num_inference_steps"] = args.num_inference_steps
    if args.guidance_scale is not None:
        cfg.setdefault("sampling", {})["guidance_scale"] = args.guidance_scale

    content_paths = list(iter_content_images(args.content_image, args.content_dir))
    if not content_paths:
        raise ValueError("No content images found.")

    if args.dry_run:
        print("Dry run OK.")
        print(f"model={cfg['model']['name']}")
        print(f"checkpoint={args.checkpoint}")
        print(f"style_image={args.style_image}")
        print(f"num_content_images={len(content_paths)}")
        print(f"output_dir={args.output_dir}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_cls = get_adapter_class(cfg["model"]["name"])
    adapter = adapter_cls(cfg)
    adapter.build_sampler(checkpoint_path=args.checkpoint, device=resolve_device(args.device))

    style_path = Path(args.style_image)
    image_size = int(cfg["model"].get("image_size", 96))

    for index, content_path in enumerate(content_paths, start=1):
        seed = None if args.seed is None else args.seed + index - 1
        output_image = adapter.sample_image(
            content_image_path=content_path,
            style_image_path=style_path,
            sampling_cfg=cfg.get("sampling", {}),
            seed=seed,
        )
        output_path = output_dir / f"{content_path.stem}__x__{style_path.stem}.png"
        output_image.save(output_path)
        print(f"[{index}/{len(content_paths)}] saved {output_path}")

        if args.save_triptych:
            triptych_path = output_dir / f"{content_path.stem}__x__{style_path.stem}__triptych.png"
            save_triptych(content_path, style_path, output_image, triptych_path, image_size)


if __name__ == "__main__":
    main()
