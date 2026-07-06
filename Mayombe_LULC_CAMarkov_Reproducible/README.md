# Spatio-temporal Dynamics of the Congolese Mayombe Forest (2000–2025)
### Mvouti District — LULC Classification (GEE) & CA-Markov Modeling — Reproducibility Repository

> This is a code reproducibility repository: it contains the complete
> source code of both pipelines. This work was carried out within the
> Climate Smart Agrifood Systems (CSAS) project. Data access is restricted
> — see [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md).

---

## 1. Description

This repository merges the **two complementary pipelines** used in the article
into a single reproducibility package, run in sequence:

| Step | Folder | Engine | Purpose |
|---|---|---|---|
| **1** | [`pipeline_1_lulc_gee/`](pipeline_1_lulc_gee/) | Google Earth Engine (JavaScript) | Multi-temporal LULC classification (Random Forest, 6 classes, 2000–2025) over the Mvouti district from Landsat 5/7/8/9 Collection 2 Level 2 (Surface Reflectance) imagery. |
| **2** | [`pipeline_2_camarkov/`](pipeline_2_camarkov/) | Python | Ingestion of the (manually audited/corrected, `_CORRIGE_v3` suffix) LULC rasters produced in step 1, binarization at the national ≥ 30 % canopy threshold (Law n°33-2020), empirical validation (Overall Accuracy, Kappa, stratified area estimator), landscape statistics (multi-class gain/loss, McNemar test, fragmentation indices), and CA-Markov modeling/projection. |

Each pipeline folder is self-contained (own `README.md`, own `src/`, own
`results/`) and can be read or re-run independently; together they reproduce
the full analytical chain of the article, from raw satellite imagery to the
final validated forest-dynamics figures and tables.

## 2. Repository Structure

```
Mayombe_LULC_CAMarkov_Reproducible/
├── README.md                        → this file
├── LICENSE                          → code license (MIT), applies to the whole repository
├── CITATION.cff                     → repository citation metadata
├── DATA_AVAILABILITY.md             → access conditions for all input data (both steps)
│
├── pipeline_1_lulc_gee/             → STEP 1 — LULC classification (Google Earth Engine)
│   ├── README.md                    → GEE-specific instructions
│   ├── src/
│   │   └── LULC_Mvouti_Reproducible.js
│   └── results/                     → placeholder (GEE exports go to Google Drive)
│
└── pipeline_2_camarkov/             → STEP 2 — CA-Markov modeling & validation (Python)
    ├── README.md                    → Python-specific instructions
    ├── requirements.txt              → pip dependencies
    ├── environment.yml               → conda environment (recommended)
    ├── src/
    │   └── camarkov_mayombe_pipeline.py
    ├── local_inputs/
    │   ├── README.md                → expected structure and naming conventions
    │   ├── geospatial/              → LULC rasters (output of step 1) + area shapefiles (not versioned)
    │   └── ground_truth/            → ground-truth points per year (not versioned)
    └── results/
        ├── figures/                 → generated figures (PNG/TIFF/PDF/SVG, 600 DPI)
        └── tables/                  → validation and statistical tables (CSV/XLSX)
```

## 3. Quick Start

```bash
# STEP 1 — LULC classification (browser, no local install)
#   Open https://code.earthengine.google.com/, paste
#   pipeline_1_lulc_gee/src/LULC_Mvouti_Reproducible.js, run, launch export tasks.
#   -> produces one classified raster per date (2000, 2005, 2010, 2015, 2020, 2025)

# STEP 2 — CA-Markov modeling & validation (local Python)
cd pipeline_2_camarkov
conda env create -f environment.yml && conda activate mayombe-camarkov
# place the step-1 rasters (after manual audit/correction) + shapefiles in
# local_inputs/geospatial/, and ground-truth points in local_inputs/ground_truth/
# — see pipeline_2_camarkov/local_inputs/README.md
python src/camarkov_mayombe_pipeline.py
```

Detailed instructions for each step are in the corresponding folder's own
`README.md`.

## 4. Input Data

Raw satellite imagery is public (Google Earth Engine catalog). Classification
training points, the corrected LULC rasters, reference shapefiles, and
ground-truth validation points are subject to access conditions detailed in
[`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md).

## 5. Key Reproducibility Parameters

| Parameter | Value | Justification |
|---|---|---|
| Forest threshold | Canopy ≥ 30 % | Law n°33-2020, Congolese Forest Code |
| Classification | Random Forest, 6 classes, seed 42 | `pipeline_1_lulc_gee` |
| Target resolution | 30 m | Native Landsat resolution |
| Output CRS | EPSG:32733 (UTM 33S) | Mayombe/Mvouti zone |
| Markov calibration period | 2015–2020 | Stable historical interval used in the article |
| Figure resolution | 600 DPI | Publication requirement |

## 6. Citation

Please cite both the article and this repository (see `CITATION.cff`). A
Zenodo DOI will be added here after archival deposit.

## 7. License

The code in this repository is distributed under the MIT license (see
`LICENSE`). Input data, when shared separately, may be subject to different
conditions specified in `DATA_AVAILABILITY.md`.

## 8. Contact

[Cédrick ONDON] — [LGETA affiliation / Université Marien Ngouabi / CRDPI] — [contact email]
