"""
Download Bangladeshi medical prescription + reference datasets from Kaggle.

Reads KAGGLE_USERNAME and KAGGLE_KEY from .env — never prints or logs the values.
Uses the Kaggle API to download datasets into validation_set/ subdirs organized
by their role: ocr_layer (images), agent_layer (text), reference (lookup).

Usage:
    python scripts/download_kaggle.py

Requires KAGGLE_USERNAME and KAGGLE_KEY to be set in .env.
Get token from https://www.kaggle.com/settings (API section → Create New Token).
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "").strip()
KAGGLE_KEY = os.environ.get("KAGGLE_KEY", "").strip()

if not KAGGLE_USERNAME or not KAGGLE_KEY:
    print("ERROR: KAGGLE_USERNAME and/or KAGGLE_KEY not set in .env", file=sys.stderr)
    print("Get a token from https://www.kaggle.com/settings", file=sys.stderr)
    print("(API section → Create New Token → download kaggle.json → copy values to .env)", file=sys.stderr)
    sys.exit(1)

# Kaggle library reads credentials from environment variables at import time
os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
os.environ["KAGGLE_KEY"] = KAGGLE_KEY


@dataclass
class KaggleDataset:
    """A single Kaggle dataset to download."""
    dataset_id: str               # kaggle dataset identifier "owner/name"
    dest_subdir: str              # relative path under validation_set/
    description: str              # short description
    license: str                  # declared license
    notes: str = ""               # extra notes about the dataset


DATASETS = [
    KaggleDataset(
        dataset_id="mehaksingal/illegible-medical-prescription-images-dataset",
        dest_subdir="ocr_layer/illegible_prescriptions",
        description="Illegible Medical Prescription Images",
        license="Unknown (check Kaggle page)",
        notes="Full-page handwritten English prescriptions. NO ground truth - for qualitative eval only.",
    ),
    KaggleDataset(
        dataset_id="shashwatwork/bengali-medical-dataset",
        dest_subdir="agent_layer/bengali_medical_dataset",
        description="Bengali Medical Dataset (patient statements + specialist labels)",
        license="CC BY 4.0",
        notes="Text only. Used for conversational agent / specialist suggestion layer, not OCR.",
    ),
    KaggleDataset(
        dataset_id="ahmedshahriarsakib/assorted-medicine-dataset-of-bangladesh",
        dest_subdir="reference/assorted_medicine_bd",
        description="Assorted Medicine Dataset of Bangladesh (21K medicines)",
        license="CC0 Public Domain",
        notes="Structured CSV reference database. Used for drug name normalization + fuzzy matching.",
    ),
]

VALIDATION_SET_DIR = REPO_ROOT / "validation_set"


def write_source_file(dest: Path, ds: KaggleDataset) -> None:
    """Write source.txt with citation info."""
    source_txt = dest / "source.txt"
    source_txt.write_text(
        f"Dataset: {ds.description}\n"
        f"Kaggle ID: {ds.dataset_id}\n"
        f"URL: https://www.kaggle.com/datasets/{ds.dataset_id}\n"
        f"License: {ds.license}\n"
        f"Notes: {ds.notes}\n"
        f"Downloaded via Kaggle API (unzipped in place).\n"
    )


def download_one(api, ds: KaggleDataset) -> tuple[bool, str]:
    """Download + unzip a single Kaggle dataset. Returns (success, message)."""
    dest = VALIDATION_SET_DIR / ds.dest_subdir

    # Skip if already populated
    if (dest / "source.txt").exists():
        existing = list(dest.rglob("*"))
        non_hidden = [p for p in existing if not p.name.startswith(".") and p.is_file()]
        if len(non_hidden) > 1:  # more than just source.txt
            return True, f"SKIP (already downloaded, {len(non_hidden)} files)"

    dest.mkdir(parents=True, exist_ok=True)

    try:
        api.dataset_download_files(
            ds.dataset_id,
            path=str(dest),
            unzip=True,
            quiet=False,
        )
    except Exception as e:
        return False, f"FAIL ({type(e).__name__}: {e})"

    write_source_file(dest, ds)

    # Count downloaded files
    files = [p for p in dest.rglob("*") if p.is_file() and not p.name.startswith(".")]
    return True, f"OK ({len(files)} files)"


def main() -> int:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("ERROR: kaggle package not installed. Run: pip install kaggle", file=sys.stderr)
        return 1

    VALIDATION_SET_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(DATASETS)} Kaggle datasets to {VALIDATION_SET_DIR}")
    print("Using KAGGLE_USERNAME: [loaded from env]")
    print("Using KAGGLE_KEY: [loaded from env]")
    print()

    api = KaggleApi()
    api.authenticate()

    results = []
    for ds in DATASETS:
        print(f"[{ds.dest_subdir}] {ds.description}")
        print(f"  → downloading {ds.dataset_id}")
        ok, msg = download_one(api, ds)
        print(f"  → {msg}")
        print()
        results.append((ds.dest_subdir, ok, msg))

    # Summary
    print("=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    successes = sum(1 for _, ok, _ in results if ok)
    for slug, ok, msg in results:
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {slug}: {msg}")
    print(f"\n{successes}/{len(results)} datasets downloaded successfully")

    return 0 if successes == len(results) else 2


if __name__ == "__main__":
    sys.exit(main())
