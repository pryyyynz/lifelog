"""Download configured local model weights into the project model directory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from app.config import get_config


def download_sentence_transformer(model_name: str, model_dir: Path, dry_run: bool) -> None:
    target = model_dir / "sentence-transformers"
    print(f"text embedding: {model_name} -> {target}")
    if dry_run:
        return
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(model_name, cache_folder=str(target))


def download_cross_encoder(model_name: str, model_dir: Path, dry_run: bool) -> None:
    target = model_dir / "cross-encoders"
    print(f"cross encoder: {model_name} -> {target}")
    if dry_run:
        return
    from sentence_transformers import CrossEncoder

    CrossEncoder(model_name, cache_folder=str(target))


def download_open_clip(model_name: str, pretrained: str, model_dir: Path, dry_run: bool) -> None:
    target = model_dir / "open_clip"
    print(f"image model: {model_name}/{pretrained} -> {target}")
    if dry_run:
        return
    target.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(target / "huggingface")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(target / "huggingface" / "hub")
    os.environ["TORCH_HOME"] = str(target / "torch")
    import open_clip

    open_clip.create_model_and_transforms(model_name, pretrained=pretrained)


def download_transcription(engine: str, model_name: str, model_dir: Path, dry_run: bool) -> None:
    target = model_dir / "transcription"
    print(f"transcription: {engine}/{model_name} -> {target}")
    if dry_run:
        return
    target.mkdir(parents=True, exist_ok=True)
    if engine == "whisperx":
        import whisperx

        whisperx.load_model(model_name, device="cpu", download_root=str(target))
    elif engine == "openai-whisper":
        import whisper

        whisper.load_model(model_name, download_root=str(target))
    else:
        raise ValueError(f"Unsupported transcription engine for automatic download: {engine}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print planned downloads only.")
    parser.add_argument(
        "--component",
        choices=("all", "text", "image", "transcription", "cross-encoder"),
        default="all",
        help="Limit download to one model component.",
    )
    args = parser.parse_args()

    config = get_config()
    config.ensure_directories()
    config.activate_tool_paths()
    model_dir = config.paths.model_dir
    if args.component in {"all", "text"}:
        download_sentence_transformer(config.models.text_embedding_model, model_dir, args.dry_run)
    if args.component in {"all", "image"}:
        download_open_clip(config.models.image_model, config.models.image_pretrained, model_dir, args.dry_run)
    if args.component in {"all", "transcription"}:
        download_transcription(
            config.models.transcription_engine,
            config.models.transcription_model,
            model_dir,
            args.dry_run,
        )
    if args.component in {"all", "cross-encoder"}:
        download_cross_encoder(config.models.cross_encoder_model, model_dir, args.dry_run)
    print("model download plan complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
