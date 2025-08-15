#!/usr/bin/env python
"""
Upload a single checkpoint file to the Hugging Face Hub.

EXAMPLE
    python push_checkpoint.py /path/to/pytorch_model.bin your-handle/my-model --private
"""

import argparse
from pathlib import Path
from huggingface_hub import create_repo, upload_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push one local checkpoint file to a Hub repo."
    )
    parser.add_argument("ckpt_path", help="Path to the checkpoint (e.g. .bin or .safetensors)")
    parser.add_argument("repo_id", help="Destination repo: <user>/<repo> or <org>/<repo>")
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private (default: public)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token (optional – falls back to cached login if omitted)",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt_path).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"❌ checkpoint not found: {ckpt_path}")

    # 1-liner each: repo creation (idempotent) + file upload
    create_repo(args.repo_id, private=args.private, exist_ok=True, token=args.token)
    upload_file(
        repo_id=args.repo_id,
        path_or_fileobj=str(ckpt_path),
        path_in_repo=ckpt_path.name,
        token=args.token,
    )

    print(f"✅ Uploaded {ckpt_path.name} → https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()