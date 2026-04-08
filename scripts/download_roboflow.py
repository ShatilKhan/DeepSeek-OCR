"""
Download Bangladeshi medical prescription datasets from Roboflow Universe.

Reads ROBOFLOW_API_KEY from .env — never prints or logs the key value.
Uses the Roboflow Python SDK to download CC BY 4.0 datasets in COCO format
into validation_set/ocr_layer/.

Usage:
    python scripts/download_roboflow.py

Requires ROBOFLOW_API_KEY to be set in .env (use publishable/public key).
"""

import os
import sys
import shutil
from pathlib import Path
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env from repo root (script is in scripts/ subdir)
REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "").strip()

if not ROBOFLOW_API_KEY:
    print("ERROR: ROBOFLOW_API_KEY not set in .env", file=sys.stderr)
    print("Get a publishable API key from https://app.roboflow.com/settings/api", file=sys.stderr)
    sys.exit(1)


@dataclass
class RoboflowDataset:
    """A single Roboflow Universe dataset to download."""
    slug: str                     # local directory name under validation_set/ocr_layer/
    workspace: str                # Roboflow workspace slug
    project: str                  # Roboflow project slug
    version: int                  # dataset version number
    url: str                      # canonical universe URL for citation
    description: str              # short description for logs
    license: str = "CC BY 4.0"


DATASETS = [
    RoboflowDataset(
        slug="daffodil_merged_893",
        workspace="daffodil-international-university-s5vpr",
        project="merged-voyoh",
        version=1,
        url="https://universe.roboflow.com/daffodil-international-university-s5vpr/merged-voyoh",
        description="Daffodil Intl. University merged prescription dataset (~893 images)",
    ),
    RoboflowDataset(
        slug="uem_doctor_429",
        workspace="university-of-engineering-and-management",
        project="doctor-prescription",
        version=6,
        url="https://universe.roboflow.com/university-of-engineering-and-management/doctor-prescription",
        description="University of Engineering and Management doctor prescription (~429 images)",
    ),
    RoboflowDataset(
        slug="daffodil_doctors_379",
        workspace="daffodil-international-university-s5vpr",
        project="doctors-prescription",
        version=1,
        url="https://universe.roboflow.com/daffodil-international-university-s5vpr/doctors-prescription",
        description="Daffodil Intl. University original doctors prescription (~379 images)",
    ),
    # NOTE: Doctor Prescription 3 (computer-vision-iptzi/doctor-prescription-3) removed.
    # Forensic comparison showed it was a duplicate of Daffodil Doctors Prescription:
    # 379/379 original filenames overlapped — same source photos, re-uploaded to Roboflow
    # by a different user with different preprocessing (different bytes, same prescriptions).
    # Using both would double-count the same data. Keeping only Daffodil Doctors.
    RoboflowDataset(
        slug="rakib_main_272",
        workspace="sirens-workspace",
        project="main-w98xr-1ethf",
        version=1,
        url="https://app.roboflow.com/sirens-workspace/main-w98xr-1ethf",
        description="RAKIB MAIN medicine extraction dataset (272 images, 236 multi-class with dosages) - forked into sirens-workspace because original medicine-extraction-by-rakib/main-w98xr had no released versions",
    ),
]

OCR_LAYER_DIR = REPO_ROOT / "validation_set" / "ocr_layer"


def write_source_file(dest: Path, ds: RoboflowDataset) -> None:
    """Write a source.txt file with citation info for this dataset."""
    source_txt = dest / "source.txt"
    source_txt.write_text(
        f"Dataset: {ds.description}\n"
        f"URL: {ds.url}\n"
        f"Workspace: {ds.workspace}\n"
        f"Project: {ds.project}\n"
        f"Version: {ds.version}\n"
        f"License: {ds.license}\n"
        f"Downloaded via Roboflow Python SDK in COCO format.\n"
    )


def download_one(rf, ds: RoboflowDataset) -> tuple[bool, str]:
    """Download a single dataset. Returns (success, message)."""
    dest = OCR_LAYER_DIR / ds.slug

    # Check if already populated (has a source.txt we wrote previously)
    if (dest / "source.txt").exists():
        existing_imgs = list(dest.rglob("*.jpg")) + list(dest.rglob("*.jpeg")) + list(dest.rglob("*.png"))
        if existing_imgs:
            return True, f"SKIP (already downloaded, {len(existing_imgs)} images)"

    dest.mkdir(parents=True, exist_ok=True)

    try:
        project = rf.workspace(ds.workspace).project(ds.project)
        version = project.version(ds.version)
        # Roboflow SDK downloads into a subdir; we tell it where
        version.download("coco", location=str(dest), overwrite=True)
    except Exception as e:
        return False, f"FAIL ({type(e).__name__}: {e})"

    write_source_file(dest, ds)

    # Count downloaded images
    img_count = len(list(dest.rglob("*.jpg"))) + len(list(dest.rglob("*.jpeg"))) + len(list(dest.rglob("*.png")))
    return True, f"OK ({img_count} images)"


def main() -> int:
    # Import here so missing package error is caught gracefully above first
    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: roboflow package not installed. Run: pip install roboflow", file=sys.stderr)
        return 1

    OCR_LAYER_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(DATASETS)} Roboflow datasets to {OCR_LAYER_DIR}")
    print("Using ROBOFLOW_API_KEY: [loaded from env]")
    print()

    # Authenticate with a scrubbed error handler — Roboflow's 401 response
    # echoes the key value in its message; we must not forward that text.
    try:
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    except RuntimeError as e:
        msg = str(e).lower()
        if "does not exist" in msg or "revoked" in msg or "401" in msg or "oauthexception" in msg:
            print("ERROR: Roboflow rejected the API key (invalid or revoked).", file=sys.stderr)
            print("Go to https://app.roboflow.com/settings/api to create a new one.", file=sys.stderr)
            print("Use the PRIVATE API key (not publishable) for the Python SDK.", file=sys.stderr)
        else:
            # Generic failure — still do not forward the raw text (may contain key).
            print(f"ERROR: Roboflow authentication failed ({type(e).__name__}).", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ERROR: Unexpected failure during Roboflow auth ({type(e).__name__}).", file=sys.stderr)
        return 1

    results = []
    for ds in DATASETS:
        print(f"[{ds.slug}] {ds.description}")
        print(f"  → downloading v{ds.version} from {ds.workspace}/{ds.project}")
        ok, msg = download_one(rf, ds)
        print(f"  → {msg}")
        print()
        results.append((ds.slug, ok, msg))

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
