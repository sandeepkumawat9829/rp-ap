"""
Auto-save utility for Google Colab.
Copies key result files to Google Drive after each experiment.

Usage:
    import utils.autosave as autosave
    autosave.save_now()  # call after each experiment
    
Or from shell:
    python3 utils/autosave.py [optional_suffix]
"""

import os
import shutil
import datetime
import sys

# Files to always save
RESULT_FILES = [
    "result.txt",
    "efficiency_results.txt",
]

# Directories to save
RESULT_DIRS = [
    "analysis_plots",
    "logs",
]


def save_now(project_root=None, suffix=""):
    """Copy all result files to Google Drive."""
    if project_root is None:
        project_root = os.getcwd()
    
    # Check if Drive is mounted
    if not os.path.exists("/content/drive/MyDrive"):
        print("[autosave] Google Drive not mounted. Skipping.")
        return False
    
    # If a suffix is provided, save to a different directory to prevent overwriting
    if suffix:
        DRIVE_SAVE_DIR = f"/content/drive/MyDrive/AdaptivePowerformer_Results_{suffix}"
    else:
        DRIVE_SAVE_DIR = "/content/drive/MyDrive/AdaptivePowerformer_Results"

    os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    saved = 0
    
    # Save individual files
    for fname in RESULT_FILES:
        src = os.path.join(project_root, fname)
        if os.path.exists(src):
            dst = os.path.join(DRIVE_SAVE_DIR, fname)
            shutil.copy2(src, dst)
            # Also save a timestamped backup
            backup = os.path.join(DRIVE_SAVE_DIR, f"{fname}.backup_{timestamp}")
            shutil.copy2(src, backup)
            saved += 1
    
    # Save directories
    for dirname in RESULT_DIRS:
        src = os.path.join(project_root, dirname)
        if os.path.exists(src):
            dst = os.path.join(DRIVE_SAVE_DIR, dirname)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            saved += 1
    
    # Save checkpoints (just checkpoint list, not full weights — too large)
    chk_dir = os.path.join(project_root, "checkpoints")
    if os.path.exists(chk_dir):
        chk_list = os.path.join(DRIVE_SAVE_DIR, "checkpoint_list.txt")
        with open(chk_list, 'w') as f:
            for d in sorted(os.listdir(chk_dir)):
                chk_path = os.path.join(chk_dir, d, "checkpoint.pth")
                size = os.path.getsize(chk_path) if os.path.exists(chk_path) else 0
                f.write(f"{d}  ({size/1e6:.1f}MB)\n")
        saved += 1
    
    print(f"[autosave] Saved {saved} items to {DRIVE_SAVE_DIR} at {timestamp}")
    return True


if __name__ == "__main__":
    suffix_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    save_now(suffix=suffix_arg)
