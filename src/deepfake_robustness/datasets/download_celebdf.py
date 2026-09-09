#!/usr/bin/env python3
"""
Celeb-DF v2 Dataset Downloader and Automated Manifest Setup Script.
Downloads Celeb-DF v2 from Google Drive, unpacks files into datasets/Celeb-DF-v2/,
and automatically builds the celebdf_manifest.csv manifest file.
"""

import argparse
import sys
import zipfile
import subprocess
import pandas as pd

from deepfake_robustness.configs import config
from deepfake_robustness.datasets.dataset import generate_celebdf_manifest, validate_celebdf_manifest

GDRIVE_FILE_ID = "1iLx76wsbi9itnkxSqz9BVBl4ZvnbIazj"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

def check_gdown():
    """Check if gdown is installed, or offer installation."""
    try:
        import gdown
        return True
    except ImportError:
        print("[+] 'gdown' package not found. Installing gdown via pip...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            return True
        except Exception as e:
            print(f"[-] Could not install gdown automatically: {e}")
            return False

def download_via_gdown(target_zip):
    """Download Celeb-DF v2 zip file using gdown python API or subprocess with resume capability."""
    print(f"[+] Downloading Celeb-DF v2 from Google Drive ID: {GDRIVE_FILE_ID}...")
    try:
        import gdown
        gdown.download(id=GDRIVE_FILE_ID, output=str(target_zip), quiet=False, resume=True)
        return True
    except Exception as e:
        print(f"[!] Direct gdown download encountered: {e}")
        print("--> Fallback to subprocess gdown command with resume...")
        cmd = [sys.executable, "-m", "gdown", "--id", GDRIVE_FILE_ID, "-O", str(target_zip), "--continue"]
        res = subprocess.run(cmd)
        return res.returncode == 0

def extract_zip(zip_path, extract_to):
    """Extract downloaded dataset archive into target directory."""
    print(f"[+] Extracting {zip_path} into {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("--> Extraction complete.")


def build_celebdf_manifest(celeb_root, output_manifest, max_frames_per_video=10):
    """Thin wrapper around the canonical manifest generator (dataset.generate_celebdf_manifest).

    Kept as a module-level function for backward compatibility with any external callers,
    but no longer duplicates the manifest-building logic itself.
    """
    df = generate_celebdf_manifest(celeb_root, output_manifest, max_frames_per_video=max_frames_per_video)
    if df is None or len(df) == 0:
        print("[-] No dataset images or video files found in Celeb-DF folders.")
        return None
    print(f"[✓] Generated Celeb-DF manifest with {len(df):,} samples: {output_manifest}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Celeb-DF v2 Automated Dataset Downloader & Manifest Generator")
    parser.add_argument("--mini", action="store_true", help="Deprecated. Genuine Celeb-DF dataset is required for cross-dataset evaluation.")
    args = parser.parse_args()

    print("=========================================================================")
    print("        CELEB-DF V2 AUTOMATED DATASET DOWNLOADER & SETUP         ")
    print("=========================================================================")

    if args.mini:
        print("[!] Note: Synthetic 'mini Celeb-DF' generation from FF++ faces has been removed for scientific rigor.")
        print("--> Cross-dataset evaluation requires genuine Celeb-DF v2 images or videos.")

    celeb_dir = config.CELEBDF_ROOT
    celeb_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.CELEBDF_MANIFEST_PATH

    # Check if manifest already exists and is valid
    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
        is_valid, reason = validate_celebdf_manifest(df)
        if is_valid:
            print(f"[✓] Valid Celeb-DF manifest already exists with {len(df):,} samples: {manifest_path}")
            print("You can directly run: python evaluate.py --celebdf")
            return
        else:
            print(f"[!] Existing Celeb-DF manifest at {manifest_path} is invalid: {reason}")
            print("--> Removing invalid manifest and re-scanning...")
            manifest_path.unlink()

    # Try building manifest if files are already extracted
    existing_df = build_celebdf_manifest(celeb_dir, manifest_path)
    if existing_df is not None:
        print("You can now run: python evaluate.py --celebdf")
        return

    # Download dataset zip
    zip_path = celeb_dir.parent / "Celeb-DF-v2.zip"
    if not zip_path.exists():
        if check_gdown():
            success = download_via_gdown(zip_path)
            if not success or not zip_path.exists():
                print("\n[!] Full 10GB Google Drive download skipped or unavailable.")
                print("\nOfficial Download Links for full 10GB dataset:")
                print(f"- Google Drive (v2): {config.CELEBDF_V2_GDRIVE_URL}")
                print(f"- Baidu Net Disk (v2): {config.CELEBDF_V2_BAIDU_URL} (passcode: yxa1)")
                sys.exit(1)
        else:
            print("\n[-] gdown not available. Please manually download Celeb-DF-v2.zip.")
            sys.exit(1)

    # Extract zip if present
    if zip_path.exists():
        extract_zip(zip_path, celeb_dir)
        build_celebdf_manifest(celeb_dir, manifest_path)
        print("\n[✓] Celeb-DF v2 dataset setup complete!")
        print("Run cross-dataset generalization evaluation: python evaluate.py --celebdf")

if __name__ == "__main__":
    main()
