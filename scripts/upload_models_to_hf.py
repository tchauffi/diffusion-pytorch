"""Upload exported ONNX models to a Hugging Face Hub repository.

Usage:
    HF_TOKEN=xxx HF_REPO_ID=username/diffusion-onnx \
    poetry run python scripts/upload_models_to_hf.py

The script looks for ONNX files inside data/models_onnx by default and uploads
(or updates) them under the /models folder of the given Hub repo.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi


def _iter_model_files(source_dir: Path) -> Iterable[Path]:
    for path in sorted(source_dir.glob("*.onnx")):
        if path.is_file():
            yield path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload ONNX models to Hugging Face Hub")
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("HF_REPO_ID"),
        help="Hugging Face repo id (user/repo). Falls back to HF_REPO_ID env var.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face access token. Falls back to HF_TOKEN env var.",
    )
    parser.add_argument(
        "--source-dir",
        default="data/models_onnx",
        help="Folder containing *.onnx files to upload.",
    )
    parser.add_argument(
        "--target-prefix",
        default="models",
        help="Subfolder inside the Hub repo to store the files.",
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="Target branch (revision) inside the Hub repo.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private if it does not exist yet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.repo_id:
        raise SystemExit("HF repo id missing. Pass --repo-id or set HF_REPO_ID env var.")
    if not args.token:
        raise SystemExit("HF access token missing. Set HF_TOKEN env var with write access.")

    source_dir = Path(args.source_dir).expanduser().resolve()
    if not source_dir.exists():
        raise SystemExit(f"Source directory not found: {source_dir}")

    files = list(_iter_model_files(source_dir))
    if not files:
        raise SystemExit(f"No .onnx files found in {source_dir}")

    api = HfApi(token=args.token)
    api.create_repo(
        repo_id=args.repo_id,
        token=args.token,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    for file_path in files:
        target_path = f"{args.target_prefix.strip('/')}/{file_path.name}"
        print(f"Uploading {file_path.name} -> {args.repo_id}:{target_path}")
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=target_path,
            repo_id=args.repo_id,
            repo_type="model",
            revision=args.branch,
            token=args.token,
        )

    print("Upload complete.")


if __name__ == "__main__":
    main()
