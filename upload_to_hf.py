#!/usr/bin/env python3
"""
Upload model checkpoint to HuggingFace Hub
"""
import os
from huggingface_hub import HfApi, upload_folder
from pathlib import Path

# Configuration
REPO_ID = "JinyeopSong/KG-R1"
CHECKPOINT_PATH = "/home/yeopjin/orcd/pool/workspace/RL_KG/verl_checkpoints/cwq-KG-r1-grpo-qwen2.5-3b-it_Aug11_f1_turn5-merged-step400"
TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")

def main():
    print(f"Starting upload to {REPO_ID}")
    print(f"Checkpoint path: {CHECKPOINT_PATH}")
    print(f"Total size: ~13GB")
    print()

    # Verify checkpoint exists
    if not Path(CHECKPOINT_PATH).exists():
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")

    # Initialize API
    api = HfApi(token=TOKEN)

    print("Uploading folder to HuggingFace Hub...")
    print("This may take a while for 13GB of data...")
    print()

    # Upload the entire folder
    # This will handle large files with Git LFS automatically
    try:
        result = upload_folder(
            folder_path=CHECKPOINT_PATH,
            repo_id=REPO_ID,
            repo_type="model",
            token=TOKEN,
            commit_message="Upload cwq-KG-r1-grpo-qwen2.5-3b-it_Aug11_f1_turn5-merged-step400 checkpoint",
            create_pr=False,
        )
        print(f"\n✓ Upload successful!")
        print(f"Commit URL: {result}")
        print(f"\nModel available at: https://huggingface.co/{REPO_ID}")

    except Exception as e:
        print(f"\n✗ Upload failed: {e}")
        raise

if __name__ == "__main__":
    main()
