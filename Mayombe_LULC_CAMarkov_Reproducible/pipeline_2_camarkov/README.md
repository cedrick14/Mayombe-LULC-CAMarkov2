# Step 2 — CA-Markov Modeling and Ground-Truth Validation Pipeline

Part of the [`Mayombe_LULC_CAMarkov_Reproducible`](../README.md) repository.
See the root `README.md` for the full two-step pipeline overview and
`DATA_AVAILABILITY.md` for data access conditions. This step consumes the
LULC rasters produced by [`pipeline_1_lulc_gee/`](../pipeline_1_lulc_gee/)
(after manual audit/correction, `_CORRIGE_v3` suffix).

## Description

Python pipeline used for:
- ingesting and standardizing multi-temporal LULC rasters (2000–2025) over the Mvouti area (Mayombe, Republic of Congo);
- binarizing forest/non-forest according to the regulatory canopy threshold of ≥ 30% (Law n°33-2020, Forest Code of the Republic of Congo);
- empirical validation using ground-truth points (Overall Accuracy, Kappa index);
- multi-class gain/loss/net-change/swap decomposition (Pontius et al. 2004);
- statistical significance testing of area change (McNemar's test);
- landscape fragmentation indices (NP, PD, MPS, LPI, ED — McGarigal & Marks 1995);
- stratified area estimation with 95% confidence intervals on areas and annual rates (Olofsson et al. 2014; Puyravaud 2003);
- spatio-temporal CA-Markov modeling (2015–2020 calibration, 2025 simulation);
- producing high-resolution scientific figures (600 DPI, PNG/TIFF/PDF/SVG formats).

Target resolution: 30 m. Method: Binary CA-Markov v34-L1L2L3.

## Structure

```
pipeline_2_camarkov/
├── README.md                    → this file
├── requirements.txt              → pip dependencies
├── environment.yml               → conda environment (recommended)
├── src/
│   └── camarkov_mayombe_pipeline.py   → main script
├── local_inputs/
│   ├── README.md                → expected structure and naming conventions
│   ├── geospatial/              → LULC rasters (from step 1) + area shapefiles (not versioned, see ../DATA_AVAILABILITY.md)
│   └── ground_truth/            → ground-truth points per year (not versioned)
└── results/
    ├── figures/                 → generated figures (PNG/TIFF/PDF/SVG, 600 DPI)
    └── tables/                  → validation and statistical tables (CSV/XLSX)
```

## Installation

### Recommended option: conda
```bash
conda env create -f environment.yml
conda activate mayombe-camarkov
```

### Alternative option: pip
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Input Data

The inputs (classified LULC rasters from step 1, study area shapefiles,
ground-truth validation points) **are not versioned in this Git repository**
due to their size and partial access restrictions. See
[`../DATA_AVAILABILITY.md`](../DATA_AVAILABILITY.md) for how to obtain them,
and [`local_inputs/README.md`](local_inputs/README.md) for the exact naming
conventions of files expected in `local_inputs/geospatial/` and
`local_inputs/ground_truth/`.

Once the files are placed in the expected locations, the script runs without
any path modification (paths are relative to this folder).

## Execution

```bash
python src/camarkov_mayombe_pipeline.py
```

Results (600 DPI figures, CSV/XLSX validation and statistical tables) are
generated in `results/figures/` and `results/tables/`.

## Key Reproducibility Parameters

| Parameter | Value | Justification |
|---|---|---|
| Forest threshold | Canopy ≥ 30% | Law n°33-2020, Congolese Forest Code |
| Target resolution | 30 m | Consistency with the Landsat-derived LULC rasters (step 1) |
| Output CRS | EPSG:32733 (UTM 33S) | Mayombe/Mvouti zone |
| Markov calibration period | 2015–2020 | Stable historical interval used in the article |
| Figure resolution | 600 DPI | Publication requirement |

## Citation

Please cite both the article and the repository (see `../CITATION.cff`).

## License

MIT (see `../LICENSE`). Input data, when shared separately, may be subject
to different conditions specified in `../DATA_AVAILABILITY.md`.
