# ==============================================================================
# SPATIO-TEMPORAL DYNAMICS OF THE CONGOLESE MAYOMBE FOREST (2000–2025)
# CA-MARKOV MODELLING & GROUND-TRUTH VALIDATION PIPELINE
# ==============================================================================
# Author: Cédrick ONDON & Gemini Collaborator
# Method: Binary CA-Markov v34-L1L2L3 | Target Resolution: 30 m
# National Forest Threshold: Tree cover >= 30% (Law n°33-2020, Republic of Congo)
# ==============================================================================
#
# REPOSITORY / REPRODUCIBILITY NOTICE
# -------------------------------------------------------------------
# This script is part of the reproducibility repository associated with
# the article:
# [Article title], [Authors], [Journal, year] — DOI: [to be completed]
#
# See README.md for environment installation, DATA_AVAILABILITY.md for
# input access conditions, and local_inputs/README.md for the expected file
# structure.
# -------------------------------------------------------------------

import os
import sys
import datetime
import warnings
import gc
import shutil
import subprocess
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from scipy.ndimage import label, uniform_filter, binary_dilation, distance_transform_edt, generate_binary_structure
from scipy.stats import chi2
from tqdm import tqdm
from tabulate import tabulate

warnings.filterwarnings('ignore')

# ==============================================================================
# 🔵 PHASE 1 — GENERAL CONFIGURATION AND DIRECTORIES
# ==============================================================================

# ─── ROOT PATHS (MODIFY ACCORDING TO YOUR CONFIGURATION) ──────────────────────
# By default, these paths point to the GitHub repository structure itself
# (local_inputs/geospatial, local_inputs/ground_truth), so the script runs
# "as-is" once the files are placed in the correct locations.
# See local_inputs/README.md.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GEOSPATIAL_DIR = os.path.join(BASE_DIR, "local_inputs", "geospatial")

# ─── REFERENCE VECTOR FILES ────────────────────────────────────────────────────
SHAPEFILE_STUDY_AREA  = os.path.join(GEOSPATIAL_DIR, "MVOUTI.shp")
SHAPEFILE_DIMONIKA    = os.path.join(GEOSPATIAL_DIR, "WDPA_Dimonika_13694.shp")
SHAPEFILE_CONCESSIONS = os.path.join(GEOSPATIAL_DIR, "Concession.shp")

# ─── EMPIRICAL VALIDATION PROTOCOL: GROUND-TRUTH POINTS ───────────────────────
VALIDATION_DIR = os.path.join(BASE_DIR, "local_inputs", "ground_truth")
VALIDATION_POINTS_PATHS = {
    2000: os.path.join(VALIDATION_DIR, "points_validation_2000.shp"),
    2005: os.path.join(VALIDATION_DIR, "points_validation_2005.shp"),
    2010: os.path.join(VALIDATION_DIR, "points_validation_2010.shp"),
    2015: os.path.join(VALIDATION_DIR, "points_validation_2015.shp"),
    2020: os.path.join(VALIDATION_DIR, "points_validation_2020.shp"),
    2025: os.path.join(VALIDATION_DIR, "points_validation_2025.shp")
}
FIELD_VALIDATION_CLASS = 'val_class'  # Name of the field holding the actual class (1 to 6)

# ─── GEOSPATIAL PARAMETERS AND TIME SERIES ────────────────────────────────────
YEARS = [2000, 2005, 2010, 2015, 2020, 2025]
UTM_EPSG = 32733  # Target UTM 33S zone

OUTPUT_DIR = os.path.join(BASE_DIR, 'results')
FIG_DIR    = os.path.join(OUTPUT_DIR, 'figures')
TAB_DIR    = os.path.join(OUTPUT_DIR, 'tables')
TMP_DIR    = os.path.join(OUTPUT_DIR, 'tmp_reproj')
for d in [OUTPUT_DIR, FIG_DIR, TAB_DIR, TMP_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── MATPLOTLIB CONFIGURATION (TRUE 600 DPI) ──────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 600, 'savefig.dpi': 600, 'font.family': 'serif',
    'axes.labelsize': 12, 'axes.titlesize': 13, 'legend.fontsize': 10,
    'axes.grid': True, 'grid.alpha': 0.2
})

# ==============================================================================
# 🟢 PHASE 2 — REPRODUCIBLE MODULES AND FUNCTIONS
# ==============================================================================

def export_figure_hd(fig, basepath, dpi=600):
    """Exports a figure in all academic formats required (v33)."""
    fig.savefig(f'{basepath}_600dpi.png', dpi=dpi, bbox_inches='tight', pil_kwargs={'compress_level': 6})
    fig.savefig(f'{basepath}_600dpi.tiff', dpi=dpi, bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
    fig.savefig(f'{basepath}.pdf', bbox_inches='tight', backend='pdf')
    fig.savefig(f'{basepath}.svg', bbox_inches='tight')
    print(f"   -> Figures exported (.png, .tiff, .pdf, .svg): {os.path.basename(basepath)}")

def find_raster_generic(year, data_dir):
    """Generic raster lookup following the project's standardized naming convention."""
    candidates = [
        os.path.join(data_dir, f'MVOUTI_LULC_{year}_CORRIGE_v3.tif'),
        os.path.join(data_dir, f'LULC_{year}_CORRIGE_v3.tif'),
        os.path.join(data_dir, f'MVOUTI_LULC_{year}_CORRIGE_v2.tif'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def standardize_raster(path_src, zone_gdf, target_epsg, year):
    """Ensures perfect geometric alignment, spatial clipping, and UTM reprojection."""
    with rasterio.open(path_src) as src:
        crs_src = src.crs
        if crs_src != CRS.from_epsg(target_epsg):
            dst_crs = f'EPSG:{target_epsg}'
            transform, width, height = calculate_default_transform(
                crs_src, dst_crs, src.width, src.height, *src.bounds
            )
            meta = src.meta.copy()
            meta.update({
                'crs': dst_crs, 'transform': transform,
                'width': width, 'height': height, 'dtype': 'float32', 'nodata': 0
            })
            tmp_path = os.path.join(TMP_DIR, f'reproj_{year}.tif')
            with rasterio.open(tmp_path, 'w', **meta) as dst:
                reproject(
                    source=rasterio.band(src, 1), destination=rasterio.band(dst, 1),
                    src_transform=src.transform, src_crs=crs_src,
                    dst_transform=transform, dst_crs=dst_crs, resampling=Resampling.nearest
                )
            src_dataset = rasterio.open(tmp_path)
        else:
            src_dataset = src

        zone_proj = zone_gdf.to_crs(src_dataset.crs)
        geoms = [f.__geo_interface__ for f in zone_proj.geometry]
        out_img, out_tf = mask(src_dataset, geoms, crop=True, nodata=0)

        data = out_img[0].astype(np.float32)
        data[data == 0] = np.nan
        pixel_ha = abs(out_tf.a * out_tf.e) / 10000

        return data, out_tf, pixel_ha


# ─── §9b — Multi-class Gain/Loss/Net-change/Swap (Pontius et al. 2004) ────────
def multiclass_transition_ha(lulc_t1, lulc_t2, class_codes, pixel_ha):
    """n x n transition matrix in hectares between two multi-class LULC rasters."""
    r = min(lulc_t1.shape[0], lulc_t2.shape[0])
    c = min(lulc_t1.shape[1], lulc_t2.shape[1])
    a1 = lulc_t1[:r, :c]; a2 = lulc_t2[:r, :c]
    valid = (~np.isnan(a1)) & (~np.isnan(a2))
    n = len(class_codes)
    mat_px = np.zeros((n, n))
    for i, ci in enumerate(class_codes):
        for j, cj in enumerate(class_codes):
            mat_px[i, j] = np.sum((a1 == ci) & (a2 == cj) & valid)
    return mat_px * pixel_ha


def gain_loss_decomposition(mat_ha, class_codes, nomenclature):
    """
    Pontius et al. (2004) decomposition per class k:
      Loss = Area_t1 - Persistence | Gain = Area_t2 - Persistence
      Net change = Area_t2 - Area_t1 | Swap = 2 x min(Gain, Loss)
    """
    records = []
    for k, code in enumerate(class_codes):
        area_t1 = mat_ha[k, :].sum()
        area_t2 = mat_ha[:, k].sum()
        persistence = mat_ha[k, k]
        loss = area_t1 - persistence
        gain = area_t2 - persistence
        net_change = area_t2 - area_t1
        swap = 2 * min(gain, loss)
        records.append({
            'Class': nomenclature[code],
            'Area_t1_ha': round(area_t1, 1), 'Area_t2_ha': round(area_t2, 1),
            'Gain_ha': round(gain, 1), 'Loss_ha': round(loss, 1),
            'Net_change_ha': round(net_change, 1), 'Swap_ha': round(swap, 1),
            'Absolute_change_ha': round(gain + loss, 1),
        })
    return pd.DataFrame(records)


# ─── §12b — McNemar's test on paired Forest/Non-forest area change ────────────
def mcnemar_test_binary_change(bin_t1, bin_t2, pixel_ha):
    """
    McNemar's test (Yates continuity correction) on paired Forest(1)/Non-forest(0)
    classification of the SAME pixels at two dates.
    H0: n(F->NF) = n(NF->F) (no net change, pure spatial swap).
    Ref.: McNemar (1947); Pontius & Millones (2011).
    """
    r = min(bin_t1.shape[0], bin_t2.shape[0])
    c = min(bin_t1.shape[1], bin_t2.shape[1])
    b1 = bin_t1[:r, :c]; b2 = bin_t2[:r, :c]
    valid = (~np.isnan(b1)) & (~np.isnan(b2))

    n12 = int(np.sum((b1 == 1) & (b2 == 0) & valid))   # Forest -> Non-forest
    n21 = int(np.sum((b1 == 0) & (b2 == 1) & valid))   # Non-forest -> Forest
    n_disc = n12 + n21
    if n_disc == 0:
        return {'n12_loss_px': n12, 'n21_gain_px': n21, 'chi2': 0.0,
                'p_value': 1.0, 'net_change_ha': 0.0}

    chi2_stat = (abs(n12 - n21) - 1) ** 2 / n_disc
    p_value = 1 - chi2.cdf(chi2_stat, df=1)
    net_change_ha = (n21 - n12) * pixel_ha
    return {'n12_loss_px': n12, 'n21_gain_px': n21, 'chi2': chi2_stat,
            'p_value': p_value, 'net_change_ha': net_change_ha}


# ─── §14b — Landscape fragmentation indices (McGarigal & Marks 1995) ──────────
def fragmentation_metrics(bin_arr, pixel_ha, pixel_size_m, connectivity=8):
    """
    Patch-based fragmentation metrics for the Forest(1) class:
      NP (Number of Patches), PD (Patch Density /100ha), MPS (Mean Patch Size, ha),
      LPI (Largest Patch Index, % landscape), ED (Edge Density, m/ha).
    """
    mask_valid = ~np.isnan(bin_arr)
    forest = (bin_arr == 1) & mask_valid

    struct = generate_binary_structure(2, 2 if connectivity == 8 else 1)
    labeled, n_patches = label(forest, structure=struct)

    total_landscape_ha = float(mask_valid.sum()) * pixel_ha
    total_forest_ha = float(forest.sum()) * pixel_ha

    if n_patches == 0 or total_forest_ha == 0:
        return dict(NP=0, PD_100ha=0.0, MPS_ha=0.0, LPI_pct=0.0, ED_m_ha=0.0,
                    Total_forest_ha=total_forest_ha)

    patch_sizes_px = np.bincount(labeled.ravel())[1:]
    patch_sizes_ha = patch_sizes_px * pixel_ha
    mean_patch_ha = patch_sizes_ha.mean()
    largest_patch_ha = patch_sizes_ha.max()
    LPI = largest_patch_ha / total_landscape_ha * 100
    PD = n_patches / total_landscape_ha * 100

    edges = int(np.sum(forest[:, :-1] != forest[:, 1:]))
    edges += int(np.sum(forest[:-1, :] != forest[1:, :]))
    perimeter_m = edges * pixel_size_m
    ED = perimeter_m / total_landscape_ha

    return dict(NP=int(n_patches), PD_100ha=round(PD, 3), MPS_ha=round(mean_patch_ha, 2),
                LPI_pct=round(LPI, 2), ED_m_ha=round(ED, 2),
                Total_forest_ha=round(total_forest_ha, 1))


# ─── §18b — Stratified area estimator + CI (Olofsson et al. 2014) ─────────────
Z_95 = 1.96

def stratified_area_ci(cm_map_ref, Ni_map, pixel_ha, z=Z_95):
    """
    Stratified area estimator with standard error (Olofsson et al. 2014, eq. 7-10).
    Strata = MAPPED classes (rows of cm_map_ref). cm_map_ref[i, j] = number of
    validation points whose MAPPED class is i and REFERENCE (field) class is j.
    Ni_map[i] = total number of MAPPED pixels in class i (population size, from
    the full raster, not the sample).
    """
    n_classes = cm_map_ref.shape[0]
    ni = cm_map_ref.sum(axis=1).astype(float)
    Wi = Ni_map / Ni_map.sum()

    p_j = np.zeros(n_classes)
    var_p_j = np.zeros(n_classes)
    for j in range(n_classes):
        for i in range(n_classes):
            if ni[i] <= 0:
                continue
            pij = cm_map_ref[i, j] / ni[i]
            p_j[j] += Wi[i] * pij
            if ni[i] > 1:
                var_p_j[j] += (Wi[i] ** 2) * pij * (1 - pij) / (ni[i] - 1)

    se_p_j = np.sqrt(var_p_j)
    A_total_px = Ni_map.sum()
    area_j = p_j * A_total_px * pixel_ha
    se_area_j = se_p_j * A_total_px * pixel_ha

    return pd.DataFrame({
        'p_hat': p_j, 'SE_p': se_p_j, 'Area_ha': area_j, 'SE_ha': se_area_j,
        'CI95_lo_ha': area_j - z * se_area_j, 'CI95_hi_ha': area_j + z * se_area_j,
    })


def rate_ci_delta_method(A1, se_A1, A2, se_A2, n_years, z=Z_95):
    """95% CI for the annual log-ratio rate of change (Puyravaud 2003), via the
    delta method, assuming independent samples at the two dates."""
    rate = np.log(A2 / A1) / n_years * 100
    var_rate = (100 / n_years) ** 2 * ((se_A2 / A2) ** 2 + (se_A1 / A1) ** 2)
    se_rate = np.sqrt(var_rate)
    return rate, se_rate, rate - z * se_rate, rate + z * se_rate


# ==============================================================================
# 🟡 PHASE 3 — OFFICIAL NOMENCLATURE AND FOREST DEFINITION
# ==============================================================================

CLASSES = {
    1: {'name': 'Dense forest',       'color': (0,   90,  0)},
    2: {'name': 'Degraded forest',    'color': (100, 155, 53)},
    3: {'name': 'Agricultural area',  'color': (255, 215, 0)},
    5: {'name': 'Water',              'color': (0,   140, 190)},
    6: {'name': 'Bare soil/Savanna',  'color': (255, 135, 16)},
}

CLASS_CODES = sorted(CLASSES.keys())
PALETTE = {c: '#{:02x}{:02x}{:02x}'.format(*CLASSES[c]['color']) for c in CLASS_CODES}
NOMENCLATURE = {c: CLASSES[c]['name'] for c in CLASS_CODES}

# Application of Law n°33-2020 (Congolese Forest Code): 30% threshold.
# Degraded forest (class 2, 10-29% canopy cover) is reclassified as Non-Forest.
FOREST_CODES = [1]
NON_FOREST_CODES = [2, 3, 5, 6]

# ==============================================================================
# 🟠 PHASE 4 — RASTER DATA LOADING AND INGESTION
# ==============================================================================

print("=== GRID INGESTION AND CORRELATION ===")
if not os.path.exists(SHAPEFILE_STUDY_AREA):
    raise FileNotFoundError(
        f"Study area file not found: {SHAPEFILE_STUDY_AREA}\n"
        f"See local_inputs/README.md for the expected structure and "
        f"DATA_AVAILABILITY.md for access terms."
    )

zone = gpd.read_file(SHAPEFILE_STUDY_AREA)
ARRAYS = {}; TRANSFORMS = {}; PIXEL_HA = {}

for y in YEARS:
    raster_path = find_raster_generic(y, GEOSPATIAL_DIR)
    if raster_path is None:
        print(f"❌ No map data found for year {y}.")
        continue

    arr, tf, ha = standardize_raster(raster_path, zone, UTM_EPSG, y)
    ARRAYS[y] = arr
    TRANSFORMS[y] = tf
    PIXEL_HA[y] = ha
    print(f"   -> Year {y} successfully synchronized. Spatial scale: {ha:.4f} ha/pixel.")

# Immediate binary transformation (1 = Dense forest, 0 = Non-Forest)
BINARY_MAPS = {}
for y, arr in ARRAYS.items():
    binary = np.zeros_like(arr)
    binary[np.isin(arr, FOREST_CODES)] = 1
    binary[np.isnan(arr)] = np.nan
    BINARY_MAPS[y] = binary

# ─── Preserve the OBSERVED series before CA-Markov modeling ───────────────────
# Phase 6 below replaces BINARY_MAPS[2025] with a SIMULATED map (whether or not
# a real 2025 raster was available), for demonstration/projection purposes.
# The statistical analyses added in Phases 7-10 (§9b, §12b, §14b, §18b) describe
# OBSERVED dynamics and must never be computed against a simulated map, so a
# frozen copy of the real, mapped years is kept here for their exclusive use.
BINARY_MAPS_OBSERVED = {y: b.copy() for y, b in BINARY_MAPS.items()}

# ==============================================================================
# 🔴 PHASE 5 — EMPIRICAL VALIDATION USING GROUND-TRUTH DATA
# ==============================================================================

print("\n=== EMPIRICAL VALIDATION USING REAL POINTS ===")
validation_metrics = []
# {year: 2x2 confusion matrix, rows=MAPPED class, cols=REFERENCE(field) class,
#  order [Non-forest(0), Forest(1)]} — required by §18b (stratified area CI).
EMPIRICAL_CM_MAPREF = {}

for y in YEARS:
    pts_path = VALIDATION_POINTS_PATHS.get(y)
    if y not in ARRAYS or not pts_path or not os.path.exists(pts_path):
        print(f"⚠️ No empirical validation points available for year {y}. Step skipped.")
        continue

    pts_gdf = gpd.read_file(pts_path).to_crs(f"EPSG:{UTM_EPSG}")
    if FIELD_VALIDATION_CLASS not in pts_gdf.columns:
        print(f"❌ Error: field '{FIELD_VALIDATION_CLASS}' does not exist in the {y} vector layer.")
        continue

    y_true, y_pred = [], []
    arr_lulc = ARRAYS[y]
    transform = TRANSFORMS[y]

    for _, row in pts_gdf.iterrows():
        geom = row.geometry
        if geom.type == 'Point':
            col = int((geom.x - transform.c) / transform.a)
            row_idx = int((geom.y - transform.f) / transform.e)

            if 0 <= row_idx < arr_lulc.shape[0] and 0 <= col < arr_lulc.shape[1]:
                pixel_val = arr_lulc[row_idx, col]
                if not np.isnan(pixel_val):
                    true_binary = 1 if row[FIELD_VALIDATION_CLASS] in FOREST_CODES else 0
                    pred_binary = 1 if pixel_val in FOREST_CODES else 0
                    y_true.append(true_binary)
                    y_pred.append(pred_binary)

    if len(y_true) > 0:
        oa = accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
        validation_metrics.append({'Year': y, 'Points': len(y_true), 'OA': oa, 'Kappa': kappa})
        print(f"   [Real Validation {y}] OA: {oa*100:.2f}% | Kappa: {kappa:.4f} (computed on {len(y_true)} points)")

        # rows=MAPPED (pred), cols=REFERENCE/field (true), order [0,1] — for §18b
        EMPIRICAL_CM_MAPREF[y] = confusion_matrix(y_pred, y_true, labels=[0, 1])
    else:
        print(f"⚠️ No valid ground-truth point intersects the raster for year {y}.")

if validation_metrics:
    pd.DataFrame(validation_metrics).to_csv(os.path.join(TAB_DIR, 'Table_Empirical_Validation.csv'), index=False)

# ==============================================================================
# 🟣 PHASE 6 — SPATIO-TEMPORAL CA-MARKOV MODELING
# ==============================================================================

print("\n=== CA-MARKOV ENGINE V34: GENERIC MODELING ===")

def compute_markov_matrix(map_t1, map_t2):
    """Computes the empirical Markov transition probability matrix."""
    valid = ~np.isnan(map_t1) & ~np.isnan(map_t2)
    t1_flat = map_t1[valid].astype(int)
    t2_flat = map_t2[valid].astype(int)

    cm = confusion_matrix(t1_flat, t2_flat, labels=[0, 1])
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix_prob = cm / row_sums
    return matrix_prob, cm

# Modeling based on the stable historical period 2015-2020 to project 2025
if 2015 in BINARY_MAPS and 2020 in BINARY_MAPS:
    print("   -> Calibrating the transition matrix over the 2015-2020 interval...")
    p_matrix, counts = compute_markov_matrix(BINARY_MAPS[2015], BINARY_MAPS[2020])
    print(f"   Markov transition matrix (0=Non-Forest, 1=Forest):\n{p_matrix}")

    p_deforestation = p_matrix[1, 0]

    print("   -> Running the spatialized Cellular Automaton toward the 2025 horizon...")
    base_map = BINARY_MAPS[2020].copy()
    valid_mask = ~np.isnan(base_map)

    non_forest_mask = (base_map == 0) & valid_mask
    dist_to_deforestation = distance_transform_edt(~non_forest_mask)

    max_d = np.max(dist_to_deforestation[valid_mask]) if np.sum(valid_mask) > 0 else 1
    spatial_suitability = 1.0 - (dist_to_deforestation / max_d)

    simulated_2025 = base_map.copy()
    forest_pixels_indices = np.where((base_map == 1) & valid_mask)

    nb_to_deforest = int(len(forest_pixels_indices[0]) * p_deforestation)

    if nb_to_deforest > 0:
        vuln_values = spatial_suitability[forest_pixels_indices]
        threshold_val = np.partition(vuln_values, -nb_to_deforest)[-nb_to_deforest]

        to_convert = (base_map == 1) & (spatial_suitability >= threshold_val) & valid_mask
        simulated_2025[to_convert] = 0

    BINARY_MAPS[2025] = simulated_2025
    print(f"   2025 CA-Markov simulation executed. Losses affected: {nb_to_deforest} pixels.")
else:
    print("❌ Insufficient 2015/2020 data to calibrate the CA-Markov matrix.")

# ==============================================================================
# ⚪ PHASE 7 — §9b MULTI-CLASS GAIN/LOSS/NET-CHANGE/SWAP (Pontius et al. 2004)
# ==============================================================================
# Decomposes total multi-class LULC change into Gain, Loss, Net change and Swap
# (relocation without net area change) for each observed period and for the
# overall bookend comparison. Computed on ARRAYS (5-class LULC), OBSERVED years
# only. Ref.: Pontius Jr., Shusas & McEachern (2004), Agric. Ecosyst. Environ.
# 101(2-3), 251-268.
# ==============================================================================

print("\n=== §9b — MULTI-CLASS GAIN / LOSS / NET CHANGE / SWAP (Pontius et al. 2004) ===")

years_real = sorted(ARRAYS.keys())
gl_all_records = []
GAINLOSS_BY_PERIOD = {}

for t1, t2 in zip(years_real[:-1], years_real[1:]):
    mat_ha_5c = multiclass_transition_ha(ARRAYS[t1], ARRAYS[t2], CLASS_CODES, PIXEL_HA[t1])
    df_gl = gain_loss_decomposition(mat_ha_5c, CLASS_CODES, NOMENCLATURE)
    GAINLOSS_BY_PERIOD[(t1, t2)] = df_gl
    print(f"\n  -- Period {t1} -> {t2} --")
    print(tabulate(df_gl, headers='keys', tablefmt='grid', showindex=False))
    for _, row in df_gl.iterrows():
        rec = row.to_dict(); rec['Period'] = f'{t1}_{t2}'
        gl_all_records.append(rec)

df_gl_all = pd.DataFrame(gl_all_records)

if len(years_real) >= 2:
    t0, tN = years_real[0], years_real[-1]
    mat_ha_full = multiclass_transition_ha(ARRAYS[t0], ARRAYS[tN], CLASS_CODES, PIXEL_HA[t0])
    df_gl_full = gain_loss_decomposition(mat_ha_full, CLASS_CODES, NOMENCLATURE)
    print(f"\n  -- Overall {t0} -> {tN} --")
    print(tabulate(df_gl_full, headers='keys', tablefmt='grid', showindex=False))

    # Figure: gains (+) / losses (-) stacked per class per period
    periods_lbl = [f'{t1}->{t2}' for t1, t2 in GAINLOSS_BY_PERIOD]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.suptitle('Figure 9b. Gains and Losses by LULC Class per Period (Pontius et al. 2004)',
                 fontweight='bold', fontsize=12)
    x = np.arange(len(periods_lbl)); width = 0.15
    for k, code in enumerate(CLASS_CODES):
        gains = [GAINLOSS_BY_PERIOD[(t1, t2)].iloc[k]['Gain_ha'] for t1, t2 in GAINLOSS_BY_PERIOD]
        losses = [-GAINLOSS_BY_PERIOD[(t1, t2)].iloc[k]['Loss_ha'] for t1, t2 in GAINLOSS_BY_PERIOD]
        offset = (k - len(CLASS_CODES) / 2) * width
        ax.bar(x + offset, gains, width, color=PALETTE[code], alpha=0.9,
               edgecolor='white', label=f'{NOMENCLATURE[code]} (gain)')
        ax.bar(x + offset, losses, width, color=PALETTE[code], alpha=0.45,
               edgecolor='white', hatch='//')
    ax.axhline(0, color='black', lw=1)
    ax.set_xticks(x); ax.set_xticklabels(periods_lbl)
    ax.set_ylabel('Area (ha)  [+ gain / - loss]')
    ax.set_title('Solid = Gain | Hatched = Loss', fontsize=9, style='italic')
    ax.legend(fontsize=7, ncol=3, framealpha=0.85, loc='upper center', bbox_to_anchor=(0.5, -0.15))
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    export_figure_hd(fig, os.path.join(FIG_DIR, 'Fig9b_GainLoss_by_Class'))
    plt.close(fig)

    with pd.ExcelWriter(os.path.join(TAB_DIR, 'Table_9b_GainLoss_Multiclass.xlsx')) as w:
        df_gl_all.to_excel(w, sheet_name='By_Period', index=False)
        df_gl_full.to_excel(w, sheet_name=f'Overall_{t0}_{tN}', index=False)
    print("   -> Table exported: Table_9b_GainLoss_Multiclass.xlsx")
else:
    print("⚠️ Fewer than 2 real LULC dates available — §9b skipped.")

# ==============================================================================
# ⚫ PHASE 8 — §12b McNEMAR'S TEST: STATISTICAL SIGNIFICANCE OF AREA CHANGE
# ==============================================================================
# Tests whether the Forest/Non-forest area change between two dates is
# statistically significant, using McNemar's test (paired binary data), which
# correctly accounts for the fact that both classifications describe the same
# spatial population of pixels — unlike a standard independence chi-square.
# Computed on BINARY_MAPS_OBSERVED (never on the simulated 2025 map).
# Ref.: McNemar (1947), Psychometrika 12:153-157; Pontius & Millones (2011).
# ==============================================================================

print("\n=== §12b — McNEMAR'S TEST: STATISTICAL SIGNIFICANCE OF AREA CHANGE ===")
print("  H0: F->NF transitions = NF->F transitions (no net change, pure swap)")

years_obs = sorted(BINARY_MAPS_OBSERVED.keys())
mcnemar_records = []

for t1, t2 in zip(years_obs[:-1], years_obs[1:]):
    res = mcnemar_test_binary_change(BINARY_MAPS_OBSERVED[t1], BINARY_MAPS_OBSERVED[t2], PIXEL_HA[t1])
    sig = 'significant (p<0.05)' if res['p_value'] < 0.05 else 'not significant'
    mcnemar_records.append({
        'Period': f'{t1}->{t2}', 'Loss_px_F_to_NF': res['n12_loss_px'],
        'Gain_px_NF_to_F': res['n21_gain_px'], 'Net_change_ha': round(res['net_change_ha'], 1),
        'Chi2_McNemar': round(res['chi2'], 3),
        'p_value': f"{res['p_value']:.2e}" if res['p_value'] > 0 else '<1e-300',
        'Significance': sig,
    })
    print(f"  {t1}->{t2}: chi2={res['chi2']:.2f}  p={res['p_value']:.3e}  "
          f"Net={res['net_change_ha']:+,.1f} ha  [{sig}]")

if len(years_obs) >= 2:
    t0, tN = years_obs[0], years_obs[-1]
    res_full = mcnemar_test_binary_change(BINARY_MAPS_OBSERVED[t0], BINARY_MAPS_OBSERVED[tN], PIXEL_HA[t0])
    sig_full = 'significant (p<0.05)' if res_full['p_value'] < 0.05 else 'not significant'
    mcnemar_records.append({
        'Period': f'{t0}->{tN} (overall)', 'Loss_px_F_to_NF': res_full['n12_loss_px'],
        'Gain_px_NF_to_F': res_full['n21_gain_px'], 'Net_change_ha': round(res_full['net_change_ha'], 1),
        'Chi2_McNemar': round(res_full['chi2'], 3),
        'p_value': f"{res_full['p_value']:.2e}" if res_full['p_value'] > 0 else '<1e-300',
        'Significance': sig_full,
    })

df_mcnemar = pd.DataFrame(mcnemar_records)
print()
print(tabulate(df_mcnemar, headers='keys', tablefmt='grid', showindex=False))
df_mcnemar.to_csv(os.path.join(TAB_DIR, 'Table_12b_McNemar_AreaChange.csv'), index=False)
print("   -> Table exported: Table_12b_McNemar_AreaChange.csv")
print("  NOTE: with census-level pixel counts, McNemar's test has very high power;")
print("  read the p-value jointly with Net_change_ha and the 95% CI on area (§18b).")

# ==============================================================================
# 🟥 PHASE 9 — §14b LANDSCAPE FRAGMENTATION INDICES (Forest class)
# ==============================================================================
# Patch-based metrics computed on BINARY_MAPS_OBSERVED for each date: Number of
# Patches (NP), Patch Density (PD), Mean Patch Size (MPS), Largest Patch Index
# (LPI) and Edge Density (ED). Ref.: McGarigal & Marks (1995) FRAGSTATS;
# Turner (1989); O'Neill et al. (1988).
# ==============================================================================

print("\n=== §14b — LANDSCAPE FRAGMENTATION INDICES (Forest class, patch-based) ===")

frag_records = []
for y in years_obs:
    px_size_m = abs(TRANSFORMS[y].a)
    met = fragmentation_metrics(BINARY_MAPS_OBSERVED[y], PIXEL_HA[y], px_size_m)
    frag_records.append({'Year': y, **met})
    print(f"  {y}: NP={met['NP']:>6,}  PD={met['PD_100ha']:.3f}/100ha  "
          f"MPS={met['MPS_ha']:.2f} ha  LPI={met['LPI_pct']:.2f}%  ED={met['ED_m_ha']:.2f} m/ha")

df_frag = pd.DataFrame(frag_records)
print()
print(tabulate(df_frag, headers='keys', tablefmt='grid', showindex=False))

if len(df_frag) >= 2:
    np_trend = df_frag['NP'].iloc[-1] - df_frag['NP'].iloc[0]
    mps_trend = df_frag['MPS_ha'].iloc[-1] - df_frag['MPS_ha'].iloc[0]
    if np_trend > 0 and mps_trend < 0:
        print("  -> Landscape is FRAGMENTING: more, smaller forest patches over time.")
    elif np_trend < 0 and mps_trend > 0:
        print("  -> Landscape is CONSOLIDATING: fewer, larger forest patches over time.")
    else:
        print("  -> Mixed fragmentation signal — inspect metrics individually.")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle('Figure 14b. Landscape Fragmentation Indices — Forest Class',
                 fontweight='bold', fontsize=12)
    metric_specs = [('NP', '#8E44AD', 'Number of Patches (NP)'),
                     ('MPS_ha', '#27AE60', 'Mean Patch Size (MPS, ha)'),
                     ('LPI_pct', '#1a4d00', 'Largest Patch Index (LPI, %)'),
                     ('ED_m_ha', '#E74C3C', 'Edge Density (ED, m/ha)')]
    for ax, (col, color, title) in zip(axes.flatten(), metric_specs):
        ax.plot(df_frag['Year'], df_frag[col], 'o-', color=color, lw=2.2, ms=7)
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_xlabel('Year'); ax.grid(alpha=0.3)
    plt.tight_layout()
    export_figure_hd(fig, os.path.join(FIG_DIR, 'Fig14b_Fragmentation_Indices'))
    plt.close(fig)

df_frag.to_csv(os.path.join(TAB_DIR, 'Table_14b_Fragmentation_Indices.csv'), index=False)
print("   -> Table exported: Table_14b_Fragmentation_Indices.csv")

# ==============================================================================
# 🟦 PHASE 10 — §18b STRATIFIED AREA ESTIMATOR & CONFIDENCE INTERVALS
# ==============================================================================
# Unbiased area estimate + 95% CI per date from the REAL per-class ground-truth
# counts (EMPIRICAL_CM_MAPREF, Phase 5), with strata = mapped classes and
# weights = proportion of total mapped pixels (Olofsson et al. 2014). Annual
# rate CIs between consecutive validated dates are obtained by error
# propagation (delta method) on the Puyravaud (2003) log-ratio formula.
# Ref.: Olofsson et al. (2014), RSE 148:42-57; Puyravaud (2003); Stehman (2014).
# ==============================================================================

print("\n=== §18b — STRATIFIED AREA ESTIMATOR & 95% CONFIDENCE INTERVALS ===")

CLASS_LABELS_BIN = ['Non-forest (0)', 'Forest (1)']
area_ci_records = []
AREA_CI_BY_YEAR = {}

if len(EMPIRICAL_CM_MAPREF) == 0:
    print("⚠️ No empirical confusion matrix available (Phase 5 found no ground-truth "
          "points) — §18b skipped. Provide local_inputs/ground_truth/"
          "points_validation_*.shp to enable this analysis.")
else:
    for y in sorted(EMPIRICAL_CM_MAPREF.keys()):
        if y not in BINARY_MAPS_OBSERVED:
            continue
        cm_y = EMPIRICAL_CM_MAPREF[y]
        arr_y = BINARY_MAPS_OBSERVED[y]
        ph_y = PIXEL_HA[y]
        Ni_map = np.array([
            float(np.nansum(arr_y == 0)),   # total mapped Non-forest pixels
            float(np.nansum(arr_y == 1)),   # total mapped Forest pixels
        ])
        df_ci = stratified_area_ci(cm_y, Ni_map, ph_y)
        df_ci.index = CLASS_LABELS_BIN
        AREA_CI_BY_YEAR[y] = df_ci

        for cls in CLASS_LABELS_BIN:
            row = df_ci.loc[cls]
            area_ci_records.append({
                'Year': y, 'Class': cls,
                'N_sample_map_class': int(cm_y[CLASS_LABELS_BIN.index(cls), :].sum()),
                'Area_mapped_ha': round(Ni_map[CLASS_LABELS_BIN.index(cls)] * ph_y, 1),
                'Area_adjusted_ha': round(row['Area_ha'], 1),
                'SE_ha': round(row['SE_ha'], 1),
                'CI95_lo_ha': round(row['CI95_lo_ha'], 1),
                'CI95_hi_ha': round(row['CI95_hi_ha'], 1),
            })
        print(f"\n  -- {y} --")
        print(tabulate(df_ci.round(2), headers='keys', tablefmt='simple'))

    df_area_ci = pd.DataFrame(area_ci_records)
    print("\n" + tabulate(df_area_ci, headers='keys', tablefmt='grid', showindex=False))

    print("\n  -- Annual rate confidence intervals (Forest class) --")
    rate_ci_records = []
    years_ci = sorted(AREA_CI_BY_YEAR.keys())
    for y1, y2 in zip(years_ci[:-1], years_ci[1:]):
        n_yrs = y2 - y1
        A1 = AREA_CI_BY_YEAR[y1].loc['Forest (1)', 'Area_ha']
        se_A1 = AREA_CI_BY_YEAR[y1].loc['Forest (1)', 'SE_ha']
        A2 = AREA_CI_BY_YEAR[y2].loc['Forest (1)', 'Area_ha']
        se_A2 = AREA_CI_BY_YEAR[y2].loc['Forest (1)', 'SE_ha']
        rate, se_rate, ci_lo, ci_hi = rate_ci_delta_method(A1, se_A1, A2, se_A2, n_yrs)
        sig = 'significant (excludes 0)' if (ci_lo > 0 or ci_hi < 0) else 'CI includes 0'
        rate_ci_records.append({
            'Period': f'{y1}->{y2}', 'Forest_ha_adj_start': round(A1, 1),
            'Forest_ha_adj_end': round(A2, 1), 'Annual_rate_%_adj': round(rate, 4),
            'SE_%': round(se_rate, 4), 'CI95_lo_%': round(ci_lo, 4),
            'CI95_hi_%': round(ci_hi, 4), 'Significance': sig,
        })
        print(f"  {y1}->{y2}: rate = {rate:+.4f} %/yr  [95% CI: {ci_lo:+.4f} , {ci_hi:+.4f}]  [{sig}]")

    df_rate_ci = pd.DataFrame(rate_ci_records) if rate_ci_records else pd.DataFrame()
    if len(df_rate_ci) > 0:
        print("\n" + tabulate(df_rate_ci, headers='keys', tablefmt='grid', showindex=False))
    else:
        print("  ⚠️ Fewer than 2 validated dates — no annual rate CI computed.")

    with pd.ExcelWriter(os.path.join(TAB_DIR, 'Table_18b_Area_Rate_ConfidenceIntervals.xlsx')) as w:
        df_area_ci.to_excel(w, sheet_name='Area_CI_by_Year_Class', index=False)
        if len(df_rate_ci) > 0:
            df_rate_ci.to_excel(w, sheet_name='Annual_Rate_CI', index=False)
    print("   -> Table exported: Table_18b_Area_Rate_ConfidenceIntervals.xlsx")

# ==============================================================================
# 🟤 PHASE 11 — PRODUCTION AND EXPORT OF REPRODUCIBLE FIGURES (v34)
# ==============================================================================

print("\n=== SCIENTIFIC GRAPHIC PRODUCTION ===")

fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
cmap_binary = ListedColormap(['#d95f02', '#1b9e77'])  # Orange: Non-Forest, Green: Forest

if 2000 in BINARY_MAPS:
    im0 = axes[0].imshow(BINARY_MAPS[2000], cmap=cmap_binary, interpolation='nearest')
    axes[0].set_title("Initial Forest Cover (Year 2000)", fontsize=11, fontweight='bold')
    axes[0].axis('off')

if 2025 in BINARY_MAPS:
    im1 = axes[1].imshow(BINARY_MAPS[2025], cmap=cmap_binary, interpolation='nearest')
    axes[1].set_title("Simulated Forest Cover (CA-Markov 2025)", fontsize=11, fontweight='bold')
    axes[1].axis('off')

patches = [mpatches.Patch(color='#1b9e77', label='Forest (Canopy >= 30%)'),
           mpatches.Patch(color='#d95f02', label='Non-Forest / Anthropized')]
fig.legend(handles=patches, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.02), frameon=True)

plt.tight_layout(rect=[0, 0.08, 1, 1])
export_figure_hd(fig, os.path.join(FIG_DIR, 'Figure_LULC_Binary_Dynamics_2000_2025'))
plt.close(fig)

shutil.rmtree(TMP_DIR, ignore_errors=True)
print("\n✅ General pipeline executed successfully.")
print("   Cartographic deliverables, gain/loss, McNemar, fragmentation and")
print("   confidence-interval tables are available in results/figures and results/tables.")
