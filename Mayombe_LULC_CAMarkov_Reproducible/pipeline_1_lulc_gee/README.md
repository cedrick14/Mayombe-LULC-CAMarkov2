# Step 1 — Multi-temporal LULC Classification (Google Earth Engine)

Part of the [`Mayombe_LULC_CAMarkov_Reproducible`](../README.md) repository.
See the root `README.md` for the full two-step pipeline overview,
`DATA_AVAILABILITY.md` for data access conditions, and
[`pipeline_2_camarkov/`](../pipeline_2_camarkov/) for the downstream
CA-Markov modeling step that consumes the rasters produced here.

## Description

Google Earth Engine (JavaScript API) script producing an LULC classification
(Random Forest, 6 classes) over 6 dates (2000, 2005, 2010, 2015, 2020, 2025)
for the Mvouti district (Mayombe forest, Republic of Congo), based on Landsat
5/7/8/9 Collection 2 Level 2 (Surface Reflectance) series.

Key features:
- **primary** temporal windows per period, with **automatic fallback to a
  wider window** in case of insufficient cloud coverage (logged in exports
  and the metrics CSV);
- atmospheric correction already applied upstream (SR Collection 2 products,
  LEDAPS/LaSRC algorithms) — built-in safeguard against accidental use of
  uncorrected TOA products;
- Random Forest classification with an explicit `seed` for strict
  reproducibility;
- automatic export of validation metrics (Overall Accuracy, Kappa) to CSV
  for all years.

## Structure

```
pipeline_1_lulc_gee/
├── README.md                          → this file
├── src/
│   └── LULC_Mvouti_Reproducible.js    → complete GEE script
└── results/                           → placeholder folder (actual exports are produced
                                           on Google Drive via Export.image.toDrive/Export.table.toDrive,
                                           as GEE does not allow direct local writing)
```

## Execution

1. Open the [Google Earth Engine Code Editor](https://code.earthengine.google.com/).
2. Copy the contents of `src/LULC_Mvouti_Reproducible.js` into a new script.
3. Adjust the parameters in **section 1**:
   - `aoi` (section 1.1): replace with the exact extent of the study area.
   - `trainingPoints` (section 1.2/1.3): replace the dummy demonstration
     points with the real asset (see `../DATA_AVAILABILITY.md`).
   - `OUTPUT_CRS` (section 1.5): verify the UTM zone matches the actual area.
4. Run (`Run`). Export tasks appear in the **Tasks** tab; launch them
   manually to trigger computations and downloads to Google Drive.
5. Once exported, manually audit/correct the classified rasters (the
   `_CORRIGE_v3` suffix expected by [`pipeline_2_camarkov`](../pipeline_2_camarkov/local_inputs/README.md)
   marks this manual correction step) before placing them in
   `../pipeline_2_camarkov/local_inputs/geospatial/`.

## Training Points

By default, the script contains a set of **dummy demonstration points**
(section 1.3), with no thematic value, allowing an end-to-end run without
sensitive data. See [`../DATA_AVAILABILITY.md`](../DATA_AVAILABILITY.md) for
the terms of access to the real training points used in the article.

## Key Reproducibility Parameters

| Parameter | Value | Justification |
|---|---|---|
| Random seed (`RANDOM_SEED`) | 42 | Train/test split and Random Forest |
| Number of trees (`RF_NUM_TREES`) | 100 | — |
| Training/validation ratio | 0.8 / 0.2 | — |
| Fallback window threshold (`MIN_IMAGE_THRESHOLD`) | 3 images | Minimum cloud coverage deemed usable |
| Output CRS | EPSG:32733 (UTM 33S) | Mvouti/Mayombe zone (consistent with the CA-Markov pipeline) |
| Resolution | 30 m | Native Landsat |
