# Data Availability Statement

> This is a code reproducibility repository: it contains the complete
> source code of both pipelines. This work was carried out within the
> Climate Smart Agrifood Systems (CSAS) project. The datasets are not
> publicly available and are not distributed through this repository due
> to project data ownership and confidentiality restrictions.

This repository has two complementary components — see the root
[`README.md`](README.md). Access conditions for each are detailed below.

## Code

The complete source code of both pipelines is publicly available in this
repository under the MIT license:
- `pipeline_1_lulc_gee/src/LULC_Mvouti_Reproducible.js` — LULC classification (Google Earth Engine).
- `pipeline_2_camarkov/src/camarkov_mayombe_pipeline.py` — validation, statistics, and CA-Markov modeling (Python).

## Satellite Imagery

The source imagery (Landsat 5/7/8/9, Collection 2 Level 2 Surface Reflectance)
is public, freely accessible via the Google Earth Engine catalog
(`LANDSAT/LT05/C02/T1_L2`, `LANDSAT/LE07/C02/T1_L2`,
`LANDSAT/LC08/C02/T1_L2`, `LANDSAT/LC09/C02/T1_L2`). No access restrictions.

## Classification Training Points (`pipeline_1_lulc_gee`)

The real training points used for the Random Forest classification
(`class` property, codes 1 to 6) were built through visual interpretation
and/or field surveys as part of the CSAS project.

**The datasets are not publicly available and are not distributed through
this repository due to project data ownership and confidentiality
restrictions.** A set of **dummy demonstration** points is included directly
in the script (`pipeline_1_lulc_gee/src/LULC_Mvouti_Reproducible.js`, section
1.3) to allow an end-to-end run without confidential data; these points have
no thematic value and must never be used to produce scientific results.

## LULC Raster Data 2000–2025 (`pipeline_2_camarkov/local_inputs/geospatial/`)

The classified LULC rasters (`MVOUTI_LULC_{year}_CORRIGE_v3.tif`) are the
direct output of the Random Forest classification pipeline in
`pipeline_1_lulc_gee/`, after manual audit and correction (the `_CORRIGE_v3`
suffix marks this manual quality-control step). They are not versioned in
this Git repository, both due to their size and because their distribution
is restricted under the CSAS project.

**The datasets are not publicly available and are not distributed through
this repository due to project data ownership and confidentiality
restrictions.** Any future public release, if one occurs, would only be
through a formal institutional channel (e.g. a Zenodo deposit tied to the
published article).

## Reference Shapefiles (study area, protected areas, concessions)

- `MVOUTI.shp` — study area boundaries (Mvouti district).
- `WDPA_Dimonika_13694.shp` — derived from the WDPA (World Database on
  Protected Areas), freely accessible at [protectedplanet.net](https://www.protectedplanet.net/).
- `Concession.shp` — forestry/mining concession boundaries.

**`MVOUTI.shp` and `Concession.shp`: the datasets are not publicly available
and are not distributed through this repository due to project data
ownership and confidentiality restrictions.** `WDPA_Dimonika_13694.shp` is
the only file in this group with no restriction — it can be re-downloaded
directly from Protected Planet.

## Ground-Truth Validation Points (`pipeline_2_camarkov/local_inputs/ground_truth/`)

The empirical validation points per year (`points_validation_{year}.shp`),
used to compute Overall Accuracy, the Kappa index, and the stratified area
estimator with confidence intervals, come from surveys and field data
collection conducted as part of the CSAS project. They are independent of
the classification training points used in `pipeline_1_lulc_gee`.

**The datasets are not publicly available and are not distributed through
this repository due to project data ownership and confidentiality
restrictions.** Their collection involved partner institutions (LGETA,
CRDPI, Marien Ngouabi University).

## Summary (to be inserted into the manuscript)

> *The Google Earth Engine classification script and the Python CA-Markov
> modeling/validation code are publicly available at [GitHub link / Zenodo
> DOI]. The source satellite imagery (Landsat Collection 2 Level 2) is public
> and accessible via the Google Earth Engine catalog. The datasets
> (classification training points, corrected LULC rasters, reference
> shapefiles, and ground-truth validation points) are not publicly available
> and are not distributed through this repository due to project data
> ownership and confidentiality restrictions.*
