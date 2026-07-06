# Expected Structure of the `local_inputs/` Folder

This folder is not versioned with the real inputs (see `../DATA_AVAILABILITY.md`).
It only documents the naming conventions and locations expected by the
`src/camarkov_mayombe_pipeline.py` script.

## `local_inputs/geospatial/`

| Expected file | Description |
|---|---|
| `MVOUTI.shp` (+ .dbf, .shx, .prj) | Study area boundaries |
| `WDPA_Dimonika_13694.shp` | Dimonika Biosphere Reserve (spatial constraint) |
| `Concession.shp` | Forestry/mining concessions |
| `MVOUTI_LULC_2000_CORRIGE_v3.tif` | Classified LULC raster, year 2000 |
| `MVOUTI_LULC_2005_CORRIGE_v3.tif` | Classified LULC raster, year 2005 |
| `MVOUTI_LULC_2010_CORRIGE_v3.tif` | Classified LULC raster, year 2010 |
| `MVOUTI_LULC_2015_CORRIGE_v3.tif` | Classified LULC raster, year 2015 |
| `MVOUTI_LULC_2020_CORRIGE_v3.tif` | Classified LULC raster, year 2020 |
| `MVOUTI_LULC_2025_CORRIGE_v3.tif` | Classified LULC raster, year 2025 (if available; otherwise simulated by the script) |

Fallback naming convention accepted by `find_raster_generic()`: `LULC_{year}_CORRIGE_v3.tif`,
`MVOUTI_LULC_{year}_CORRIGE_v2.tif`.

LULC class codes used (`val_class` and rasters):

| Code | Class |
|---|---|
| 1 | Dense forest |
| 2 | Degraded forest |
| 3 | Agricultural area |
| 5 | Water |
| 6 | Bare soil / Savanna |

## `local_inputs/ground_truth/`

| Expected file | Description |
|---|---|
| `points_validation_2000.shp` | Ground-truth points, year 2000 |
| `points_validation_2005.shp` | Ground-truth points, year 2005 |
| `points_validation_2010.shp` | Ground-truth points, year 2010 |
| `points_validation_2015.shp` | Ground-truth points, year 2015 |
| `points_validation_2020.shp` | Ground-truth points, year 2020 |
| `points_validation_2025.shp` | Ground-truth points, year 2025 |

Each shapefile must contain a `val_class` field (integer, codes 1 to 6 above)
giving the actual observed class at the point.

## Obtaining These Files

See [`../DATA_AVAILABILITY.md`](../DATA_AVAILABILITY.md).
