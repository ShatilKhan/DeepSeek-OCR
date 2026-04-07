# Validation Set — DeepSeek-OCR for Bangladeshi Medical Prescriptions

This directory holds datasets used to evaluate DeepSeek-OCR on the Swastho hospital use case: extracting drug names and information from real Bangladeshi doctor prescriptions.

> **Status:** directory structure only — datasets not yet downloaded. Run `scripts/download_roboflow.py` and `scripts/download_kaggle.py` to populate (Step 2).

## Directory Layout

```
validation_set/
├── README.md                          # this file (committed)
│
├── ocr_layer/                         # Prescription image datasets for OCR evaluation
│   ├── daffodil_merged_893/           # Roboflow — Daffodil Univ. combined dataset (largest)
│   ├── uem_doctor_429/                # Roboflow — Univ. of Engineering and Management
│   ├── daffodil_doctors_379/          # Roboflow — Daffodil Univ. primary set
│   ├── doctor_prescription_3_411/     # Roboflow — Third independent institution
│   ├── rakib_main_272/                # Roboflow — Medicine extraction focus
│   └── illegible_prescriptions/       # Kaggle — worst-case English handwriting (no GT)
│
├── agent_layer/                       # Text datasets for conversation/specialist suggestion
│   └── bengali_medical_dataset/       # Kaggle — patient messages → specialist mapping
│
└── reference/                         # Structured lookup databases
    └── assorted_medicine_bd/          # Kaggle — 21K Bangladeshi medicines (brand→generic)
```

## Dataset Inventory

### OCR Layer — Full-page prescription images

| Dataset | Source | Images | Annotation Type | License |
|---------|--------|--------|----------------|---------|
| Daffodil Merged | [Roboflow](https://universe.roboflow.com/daffodil-international-university-s5vpr/merged-voyoh) | 893 | Medicine name bounding boxes | CC BY 4.0 |
| UEM Doctor Prescription | [Roboflow](https://universe.roboflow.com/university-of-engineering-and-management/doctor-prescription) | 429 | 831 medicine classes | CC BY 4.0 |
| Daffodil Doctors Prescription | [Roboflow](https://universe.roboflow.com/daffodil-international-university-s5vpr/doctors-prescription) | 379 | 1,400 medicine classes | CC BY 4.0 |
| Doctor Prescription 3 | [Roboflow](https://universe.roboflow.com/computer-vision-iptzi/doctor-prescription-3) | 411 | Medicine bounding boxes | TBD |
| RAKIB MAIN | [Roboflow](https://universe.roboflow.com/medicine-extraction-by-rakib/main-w98xr) | 272 | Medicine bounding boxes | TBD |
| Illegible Medical Prescriptions | [Kaggle](https://www.kaggle.com/datasets/mehaksingal/illegible-medical-prescription-images-dataset) | ~200 | None — qualitative eval only | TBD |

**OCR layer total: ~2,584 prescription images**

### Agent Layer — Text for conversation/suggestion

| Dataset | Source | Rows | Content | License |
|---------|--------|------|---------|---------|
| Bengali Medical Dataset | [Kaggle](https://www.kaggle.com/datasets/shashwatwork/bengali-medical-dataset) | ~600 | Patient statements + specialist suggestions + symptom/body part NER | CC BY 4.0 |

### Reference — Structured lookup

| Dataset | Source | Rows | Content | License |
|---------|--------|------|---------|---------|
| Assorted Medicine Dataset of Bangladesh | [Kaggle](https://www.kaggle.com/datasets/ahmedshahriarsakib/assorted-medicine-dataset-of-bangladesh) | 21,000+ medicines | Brand name, generic, manufacturer, drug class, dosage form, indications | CC0 Public Domain |

## Evaluation Strategy

### Tier 1: Automatic medicine name detection (primary metric)

For each Roboflow dataset, measure:
- **Medicine detection accuracy** — does DeepSeek-OCR's output contain all the medicine names that the bounding boxes say should be there?
- **Per-drug recall** — which specific drugs get extracted reliably vs which get missed?
- **Precision** — how many extracted tokens are real medicines (validated against the 21K-medicine reference database)?

### Tier 2: Qualitative worst-case analysis

For the Illegible Prescriptions dataset (no GT), manually review a sample to categorize failure modes:
- Hallucinations (model invents text)
- Skipped content (model ignores regions)
- Character substitution (English handwriting confusion)
- Layout breaks (mixed Bengali/English form confusion)

### Tier 3: Drug normalization

Post-process OCR output using the reference database:
- Fuzzy match candidate tokens against `medicine.csv` (21K brands)
- Normalize brand → generic via `generic.csv`
- Validate dosage forms and drug classes

## Citation Requirements

All datasets marked CC BY 4.0 require attribution. Citations for research use are documented in each dataset's subdirectory `source.txt` file.

## Data NOT Committed to Git

Per `.gitignore`, the actual image files, zip archives, and raw annotation CSVs are NOT committed. Only this README and per-dataset `manifest.csv` files are version-controlled. Re-downloading is handled by the scripts in `scripts/`.

## How to Populate

```bash
# Ensure ROBOFLOW_API_KEY, KAGGLE_USERNAME, KAGGLE_KEY are set in .env
python scripts/download_roboflow.py
python scripts/download_kaggle.py
```

Both scripts read credentials from environment variables via `python-dotenv`. No hardcoded keys anywhere.
