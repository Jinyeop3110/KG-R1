#!/usr/bin/env python3
"""
KG-R1 Inference Initialization Script

Downloads required data from HuggingFace and sets up the environment.
This is a mock/template version - update with actual HuggingFace dataset paths.
"""

import os
import sys
from pathlib import Path
from typing import Optional

def print_header(text: str):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_status(text: str, status: str = "info"):
    """Print status message"""
    symbols = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
        "progress": "⏳"
    }
    print(f"{symbols.get(status, 'ℹ️')} {text}")

def check_dependencies():
    """Check if required packages are installed"""
    print_header("Checking Dependencies")

    required_packages = [
        'huggingface_hub',
        'datasets',
        'tqdm'
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print_status(f"{package} is installed", "success")
        except ImportError:
            print_status(f"{package} is missing", "error")
            missing.append(package)

    if missing:
        print_status(f"\nPlease install missing packages:", "error")
        print(f"  pip install {' '.join(missing)}")
        return False

    return True

def download_kg_data(dataset_id: str, output_dir: Path, force: bool = False):
    """
    Download KG data from HuggingFace

    Args:
        dataset_id: HuggingFace dataset ID (e.g., 'JinyeopSong/KG-R1-data')
        output_dir: Output directory for data
        force: Force re-download even if data exists
    """
    from huggingface_hub import snapshot_download
    from tqdm import tqdm

    print_header(f"Downloading KG Data: {dataset_id}")

    # Check if data already exists
    if output_dir.exists() and not force:
        print_status(f"Data already exists at {output_dir}", "warning")
        response = input("Do you want to re-download? (y/N): ").strip().lower()
        if response != 'y':
            print_status("Skipping download", "info")
            return True

    try:
        print_status("Downloading from HuggingFace Hub...", "progress")

        # Download dataset
        snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
            resume_download=True
        )

        print_status(f"Data downloaded to {output_dir}", "success")
        return True

    except Exception as e:
        print_status(f"Download failed: {e}", "error")
        return False

def verify_data_structure(data_dir: Path):
    """Verify that downloaded data has correct structure"""
    print_header("Verifying Data Structure")

    # Expected structure for KG data
    required_dirs = [
        'cwq_search_augmented_initial_entities',
        'webqsp_search_augmented_initial_entities'
    ]

    required_files = {
        'cwq_search_augmented_initial_entities': ['test.parquet'],
        'webqsp_search_augmented_initial_entities': ['test.parquet']
    }

    all_ok = True

    for dir_name in required_dirs:
        dir_path = data_dir / dir_name
        if dir_path.exists():
            print_status(f"Found: {dir_name}/", "success")

            # Check required files
            for file_name in required_files.get(dir_name, []):
                file_path = dir_path / file_name
                if file_path.exists():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    print_status(f"  - {file_name} ({size_mb:.1f} MB)", "success")
                else:
                    print_status(f"  - {file_name} (missing)", "error")
                    all_ok = False
        else:
            print_status(f"Missing: {dir_name}/", "error")
            all_ok = False

    return all_ok

def setup_environment():
    """Set up environment variables and configuration"""
    print_header("Setting Up Environment")

    project_root = Path(__file__).parent.resolve()
    data_kg_dir = project_root / "data_kg"

    # Create .env file if it doesn't exist
    env_file = project_root / ".env"
    if not env_file.exists():
        with open(env_file, 'w') as f:
            f.write(f"# KG-R1 Inference Environment Configuration\n")
            f.write(f"DATA_DIR={data_kg_dir}\n")
            f.write(f"HF_HOME={project_root / '.cache' / 'huggingface'}\n")
            f.write(f"\n# KG Server Configuration\n")
            f.write(f"KG_SERVER_HOST=0.0.0.0\n")
            f.write(f"KG_SERVER_PORT=8001\n")

        print_status(f"Created .env file", "success")
    else:
        print_status(f".env file already exists", "info")

    return True

def main():
    """Main initialization function"""
    print_header("KG-R1 Inference Initialization")
    print("This script will download and set up required data for KG-R1 inference.\n")

    # Get project root
    project_root = Path(__file__).parent.resolve()
    data_kg_dir = project_root / "data_kg"

    print(f"Project root: {project_root}")
    print(f"Data directory: {data_kg_dir}\n")

    # Step 1: Check dependencies
    if not check_dependencies():
        print_status("\nPlease install missing dependencies and try again.", "error")
        sys.exit(1)

    # Step 2: Download data (MOCK - replace with actual HF dataset ID)
    # TODO: Replace 'JinyeopSong/KG-R1-data' with actual dataset ID when available
    dataset_id = "JinyeopSong/KG-R1-data"  # MOCK - Update this!

    print_status("\n⚠️  MOCK MODE: Using placeholder dataset ID", "warning")
    print_status(f"Current dataset ID: {dataset_id}", "info")
    print_status("Please update this in initialize.py when HF dataset is ready", "warning")

    response = input("\nDo you want to proceed with download? (y/N): ").strip().lower()
    if response != 'y':
        print_status("Skipping data download", "info")
        print_status("You can manually link data: ln -s /path/to/data_kg ./data_kg", "info")
    else:
        success = download_kg_data(dataset_id, data_kg_dir)
        if not success:
            print_status("\nData download failed. You can:", "error")
            print("  1. Manually copy data: cp -r /path/to/data_kg ./")
            print("  2. Create symlink: ln -s /path/to/data_kg ./data_kg")
            print("  3. Update dataset_id in initialize.py and try again")
            # Don't exit - continue with setup

    # Step 3: Verify data structure
    if data_kg_dir.exists():
        if verify_data_structure(data_kg_dir):
            print_status("\nData structure verified!", "success")
        else:
            print_status("\nData structure incomplete", "warning")
            print_status("Some files may be missing. Evaluation may not work.", "warning")
    else:
        print_status("\nNo data_kg directory found", "warning")
        print_status("For simple inference, this is OK (uses HF model)", "info")
        print_status("For full evaluation, you need to set up data_kg/", "info")

    # Step 4: Set up environment
    setup_environment()

    # Final summary
    print_header("Initialization Complete!")

    print("\n✅ Next Steps:")
    print("\n1. Install package:")
    print("   pip install -e .")

    print("\n2. Run simple inference (no data needed):")
    print("   python simple_inference.py --question 'Who directed Inception?'")

    if data_kg_dir.exists():
        print("\n3. Launch KG server (for full evaluation):")
        print("   bash scripts/kg_retrieval_launch_cwq.sh")

        print("\n4. Run full evaluation:")
        print("   cd eval_scripts/kg_r1_eval_main")
        print("   bash eval_qwen_3b_cwq_f1_turn5.sh")
    else:
        print("\n3. Set up data for full evaluation:")
        print("   ln -s /path/to/RL_KG/data_kg ./data_kg")
        print("   OR update dataset_id in initialize.py and run again")

    print("\n📚 Documentation:")
    print("   - README.md - Complete guide")
    print("   - QUICKSTART.md - 3-minute setup")
    print("   - MODEL_INFO.md - HuggingFace model details")

    print("\n" + "="*60)
    print("Happy reasoning! 🧠💡")
    print("="*60 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInitialization cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Initialization failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
