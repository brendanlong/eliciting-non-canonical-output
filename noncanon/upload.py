"""Upload a run directory to the HuggingFace dataset that holds all artifacts.

    uv run python -m noncanon.upload out/pilot pilot --repo brendanlong/noncanonical-post-training
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("path_in_repo")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--public", action="store_true", help="create the dataset public if it does not exist")
    args = ap.parse_args()
    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=not args.public, exist_ok=True)
    api.upload_folder(folder_path=str(args.folder), path_in_repo=args.path_in_repo, repo_id=args.repo, repo_type="dataset")
    print(f"uploaded {args.folder} -> {args.repo}/{args.path_in_repo}")


if __name__ == "__main__":
    main()
