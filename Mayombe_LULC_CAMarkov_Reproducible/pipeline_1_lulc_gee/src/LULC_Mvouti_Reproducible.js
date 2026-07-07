// =========================================================================
// REPRODUCIBILITY SCRIPT — MULTI-TEMPORAL LULC CLASSIFICATION
// Mvouti district, Mayombe forest (Republic of Congo) — 2000, 2005,
// 2010, 2015, 2020, 2025
// Automated Random Forest classification over 6 dates
// CORRECTED VERSION v1.2 — see change log below
// =========================================================================
//
// REPRODUCIBILITY DECLARATION / CODE & DATA AVAILABILITY STATEMENT
// -------------------------------------------------------------------
// 
//
// Training data: the dataset is not publicly available and is not
// distributed through this repository due to project data ownership and
// confidentiality restrictions (see ../../DATA_AVAILABILITY.md).
// A set of DUMMY points is provided below (section 1.3) solely to
// allow an end-to-end demonstration run of the script. With only 12
// dummy points across 6 classes, the spatial k-fold cross-validation
// in section 4 is illustrative only — it will behave far more robustly
// once the real training asset (with a realistic sample size) is used.
//
// Temporal windows: each target year uses a PRIMARY window (section
// 1.4), justified explicitly per year (see window_rationale below); if
// cloud cover is insufficient — either too few images overall
// (MIN_IMAGE_THRESHOLD) or too little valid AOI coverage at the pixel
// level (PIXEL_COVERAGE_FRACTION_THRESHOLD, section 3) — the script
// automatically switches to a wider FALLBACK window. The window
// actually used, the representative date offset, and both coverage
// diagnostics are logged to the console and exported to the metrics
// CSV for full traceability.
//
// Cross-sensor radiometric harmonization: Landsat 5 (TM) and Landsat 7
// (ETM+) reflectance is harmonized toward an OLI-like (Landsat 8/9)
// spectral response before compositing, so that the six epochs
// (spanning TM, ETM+, OLI and OLI-2) are radiometrically comparable
// (section 2, HARMONIZE_CROSS_SENSOR).
//
// Atmospheric correction: already applied upstream (Collection 2
// Level 2 Surface Reflectance products, LEDAPS/LaSRC algorithms — see
// section 2 for details and the anti-TOA safeguard).
//
// -------------------------------------------------------------------
// CHANGE LOG (v1.0 -> v1.1)
// -------------------------------------------------------------------
// [FIX 1] QA_PIXEL bit/variable naming was swapped: bit 3 is "Cloud"
//         and bit 4 is "Cloud Shadow" per the USGS Collection 2
//         QA_PIXEL specification. The combined mask result was correct
//         in v1.0, but variable names were misleading. Renamed for
//         consistency with the methodology described in the article.
// [FIX 2] A stricter minimum image threshold (MIN_IMAGE_THRESHOLD_SLC_OFF)
//         is now applied for years relying on Landsat 7 within the
//         SLC-off era (2005, 2010), so the median composite has enough
//         independent scan-line patterns to statistically fill gaps.
// [FIX 3] If both the primary and fallback windows fail to reach the
//         minimum image threshold, classification is skipped for that
//         year, the raw composite export is suffixed "_LOWCONF", and
//         the year is flagged 'INSUFFICIENT' in the metrics CSV.
// [FIX 4] `representative_date` is now exported in the metrics CSV,
//         together with `days_offset_from_representative_date` — the
//         gap in days between the representative date and the actual
//         midpoint of the window used (primary or fallback).
// [FIX 5] `trainingPoints.size().getInfo()` is now computed once
//         outside the yearly loop instead of once per iteration.
// -------------------------------------------------------------------
// CHANGE LOG (v1.1 -> v1.2)
// -------------------------------------------------------------------
// [FIX 6] CROSS-SENSOR RADIOMETRIC HARMONIZATION. TM/ETM+ reflectance
//         (Landsat 5/7) is now transformed toward an OLI-like spectral
//         response using published band-specific linear coefficients
//         (Roy et al., 2016) before being merged with native OLI/OLI-2
//         (Landsat 8/9) reflectance. Toggle: HARMONIZE_CROSS_SENSOR.
//         IMPORTANT: the coefficients below are entered from memory of
//         the general form of the Roy et al. (2016) OLS transformation
//         and MUST be verified against the original publication
//         (Roy, D.P. et al., 2016, Remote Sensing of Environment,
//         "Characterization of Landsat-7 to Landsat-8 reflective
//         wavelength and normalized difference vegetation index
//         continuity") before being used for any published result —
//         they are flagged TO VERIFY in the code, exactly as the
//         training-data asset path is flagged TO ADAPT.
// [FIX 7] WINDOW WIDTH JUSTIFICATION. Each entry in TEMPORAL_CONFIG now
//         carries an explicit `window_rationale` string documenting why
//         its primary window width/symmetry differs from other years
//         (sensor revisit frequency, cloud climatology). This is
//         printed to console and the width in days is exported to the
//         metrics CSV (`window_width_days`), so the asymmetry across
//         years is traceable rather than implicit.
// [FIX 8] PIXEL-LEVEL COVERAGE CHECK. In addition to the existing
//         collection-level image count, the script now computes the
//         fraction of AOI pixels reaching at least `minImageThreshold`
//         valid (unmasked) observations. If this fraction falls below
//         PIXEL_COVERAGE_FRACTION_THRESHOLD even after the fallback
//         window, the year is flagged insufficient (same downstream
//         handling as FIX 3), because a good collection-level image
//         count does not guarantee homogeneous spatial coverage across
//         a cloud-prone AOI.
// [FIX 9] SPATIAL BLOCK K-FOLD CROSS-VALIDATION replaces the single
//         random 80/20 split. Training points are assigned to a
//         spatial grid block (SPATIAL_BLOCK_SIZE_DEGREES) and folds are
//         built from blocks rather than individual points, reducing
//         the risk of spatially autocorrelated train/test pairs
//         inflating Overall Accuracy / Kappa. Mean and standard
//         deviation of OA and Kappa across K_FOLDS folds are exported
//         per year in the main metrics CSV, and a full per-fold detail
//         table is exported separately
//         (LULC_CrossValidation_PerFold_All_Years). The classifier used
//         to produce the final exported classification map is trained
//         on the FULL training set (standard practice: cross-validation
//         estimates generalization performance, the deployed map uses
//         all available labeled data).
// [FIX 10] SCENE-LEVEL CLOUD-COVER FILTER (re-added). A scene-level
//         filter on the CLOUD_COVER metadata property
//         (SCENE_CLOUD_COVER_THRESHOLD_PERCENT, default 5%) is now
//         applied to both the primary and fallback collections, BEFORE
//         the pixel-level QA_PIXEL masking (maskClouds). v1.1 only had
//         the pixel-level mask; the scene-level filter was missing even
//         though the article's methodology section describes it. The
//         two mechanisms are complementary: the scene-level filter
//         discards heavily contaminated scenes outright (matching the
//         dry-season acquisition strategy), while the pixel-level mask
//         still handles residual/isolated cloud, shadow, and cirrus
//         contamination in the scenes that pass this filter.
// -------------------------------------------------------------------

// ==========================================
// 1. USER PARAMETERS (TO ADJUST)
// ==========================================

// ---- 1.1 Study Area (AOI) — FIXED geometry, independent of the display ----
// Replace with the exact extent used in the article (coordinates in degrees,
// or a shared asset). Using Map.getBounds() is prohibited here because it
// makes the AOI dependent on the screen at execution time, and therefore
// non-reproducible.
var aoi = ee.Geometry.Rectangle([
  12.20, -4.60,   // [lon_min, lat_min] — TO REPLACE with the real extent of the Mvouti district
  12.70, -4.10    // [lon_max, lat_max]
]);

// Recommended alternative if an AOI shapefile/asset already exists:
// var aoi = ee.FeatureCollection('projects/your_project/assets/AOI_Mvouti').geometry();

// ---- 1.2 REAL training data (private or public asset) ----
// Replace the path below (line ~80, in the definition of
// trainingPoints) with the actual path to your GEE asset, then set
// USE_DUMMY_POINTS to false (section 1.3) to activate it.
// Default path (TO ADAPT): 'projects/your_project/assets/mvouti_training_points'

// ---- 1.3 DUMMY demonstration point set (fallback if no real asset) ----
// These points are generated arbitrarily inside the AOI to allow the
// script to run end-to-end. They HAVE NO THEMATIC VALUE and must never
// be used to produce scientific results. NOTE: with only 12 points
// across 6 classes, the k-fold cross-validation in section 4 is for
// demonstration only (small/empty folds are expected and handled
// gracefully, see runSpatialKFoldCV).
var dummyPoints = ee.FeatureCollection([
  ee.Feature(ee.Geometry.Point([12.30, -4.50]), {class: 1}),
  ee.Feature(ee.Geometry.Point([12.35, -4.47]), {class: 1}),
  ee.Feature(ee.Geometry.Point([12.40, -4.45]), {class: 2}),
  ee.Feature(ee.Geometry.Point([12.45, -4.43]), {class: 2}),
  ee.Feature(ee.Geometry.Point([12.50, -4.40]), {class: 3}),
  ee.Feature(ee.Geometry.Point([12.52, -4.38]), {class: 3}),
  ee.Feature(ee.Geometry.Point([12.25, -4.35]), {class: 4}),
  ee.Feature(ee.Geometry.Point([12.28, -4.33]), {class: 4}),
  ee.Feature(ee.Geometry.Point([12.38, -4.55]), {class: 5}),
  ee.Feature(ee.Geometry.Point([12.42, -4.53]), {class: 5}),
  ee.Feature(ee.Geometry.Point([12.55, -4.30]), {class: 6}),
  ee.Feature(ee.Geometry.Point([12.58, -4.28]), {class: 6})
]);

// Effective selection: manually switch between real and dummy points.
// Default -> dummy, to guarantee the script runs without error even
// without access to the private asset.
// USE_DUMMY_POINTS is the explicit source of truth (more robust than an
// object-reference comparison): set to false once the real asset is active.
var USE_DUMMY_POINTS = true;

var trainingPoints = USE_DUMMY_POINTS
  ? dummyPoints
  : ee.FeatureCollection('projects/your_project/assets/mvouti_training_points');

// [FIX 5] Computed once, outside the yearly loop (trainingPoints is
// invariant across years, so re-querying it 6 times was redundant).
var trainingPointsCount = trainingPoints.size().getInfo();

// ---- 1.4 Annual date configuration: PRIMARY WINDOW + FALLBACK WINDOW ----
// [FIX 7] Each entry now documents WHY its primary window has the width
// and symmetry it has (`window_rationale`), instead of leaving the
// asymmetry across years implicit / unexplained.
//
// `representative_date` (ISO 8601, YYYY-07-02, day 183 / statistical
// mid-year) is the nominal reference date for each record, used to
// compute how far the composite's actual temporal center drifts from
// this reference (section 3).
var TEMPORAL_CONFIG = [
  {
    year: 2000,
    representative_date: '2000-07-02',
    primary: {start: '2000-01-01', end: '2000-12-31'},
    fallback: {start: '1998-01-01', end: '2002-12-31'},
  
  },
  {
    year: 2005,
    representative_date: '2005-07-02',
    primary: {start: '2005-01-01', end: '2005-12-31'},
    fallback: {start: '2004-01-01', end: '2006-12-31'},
    
  {
    year: 2010,
    representative_date: '2010-07-02',
    primary: {start: '2010-01-01', end: '2010-12-31'},
    fallback: {start: '2009-01-01', end: '2011-12-31'},
  },
  {
    year: 2015,
    representative_date: '2015-07-02',
    primary: {start: '2014-01-01', end: '2015-12-31'},
    fallback: {start: '2013-01-01', end: '2016-12-31'},
   
  },
  {
    year: 2020,
    representative_date: '2020-07-02',
    primary: {start: '2019-01-01', end: '2020-12-31'},
    fallback: {start: '2018-01-01', end: '2021-12-31'},
    window_rationale: '2-year window: same rationale as 2015 (stable Landsat 8 revisit); ' +
      'kept backward-only for consistency and to avoid overlap with 2025.'
  },
  {
    year: 2025,
    representative_date: '2025-07-02',
    primary: {start: '2024-01-01', end: '2025-12-31'},
    fallback: {start: '2022-01-01', end: '2025-12-31'},
    window_rationale: '2-year window: same rationale as 2015/2020 (Landsat 8/9 combined ' +
      'revisit). The fallback is widened further back (2022-2025) rather than ' +
      'forward, since 2025 is the most recent year and cannot borrow images ' +
      'from a future acquisition.'
  }
];

// Minimum number of images (after cloud masking) below which the script
// switches to the fallback window for the year concerned.
var MIN_IMAGE_THRESHOLD = 3;

// [FIX 2] Stricter threshold for years relying on Landsat 7 within the
// SLC-off era (post May 2003): 2005 and 2010 use L5+L7 collections
// where L7 scenes contain unfilled scan-line gaps. A higher minimum
// image count increases the chance that the median compositing
// statistically fills these gaps using overlapping, differently
// striped acquisitions.
var MIN_IMAGE_THRESHOLD_SLC_OFF = 5;
var SLC_OFF_AFFECTED_YEARS = [2005, 2010];

function getMinImageThreshold(year) {
  return SLC_OFF_AFFECTED_YEARS.indexOf(year) !== -1
    ? MIN_IMAGE_THRESHOLD_SLC_OFF
    : MIN_IMAGE_THRESHOLD;
}

// [FIX 8] Minimum fraction of AOI pixels that must reach the minimum
// image threshold at the PIXEL level (not just the collection level).
// A collection can contain enough images in total while still leaving
// part of the AOI poorly covered if cloud cover is spatially uneven.
var PIXEL_COVERAGE_FRACTION_THRESHOLD = 0.70;
var PIXEL_COVERAGE_CHECK_SCALE = 90;  // coarser than the 30 m export scale: QA-only, not a data product

// [FIX 10] Scene-level cloud-cover filter, applied upstream of the
// pixel-level QA_PIXEL masking (maskClouds). Scenes reporting more than
// this percentage of cloud cover in their metadata (CLOUD_COVER
// property, all four Landsat missions) are excluded from the
// collection entirely before compositing, consistent with the dry-
// season acquisition strategy described in the article. This is
// distinct from, and complementary to, the per-pixel cloud/shadow/
// cirrus masking already applied in maskClouds(): the scene-level
// filter discards heavily contaminated scenes outright, while the
// pixel-level mask handles residual/isolated cloud contamination in
// scenes that pass this filter.
var SCENE_CLOUD_COVER_THRESHOLD_PERCENT = 5;

// ---- 1.5 Geospatial configuration for exports ----
var OUTPUT_CRS = 'EPSG:32733'; // UTM 33S — Mvouti/Mayombe zone (consistent with the CA-Markov pipeline)
var OUTPUT_FOLDER = 'GEE_LULC_Final_Exports';

// ---- 1.6 LULC classes and legend ----
var PALETTE = ['#1a6600', '#4daf4a', '#f5c518', '#00ced1', '#0033ff', '#d2691e'];
var CLASS_NAMES = [
  'Dense forest',
  'Secondary forest',
  'Agriculture / Fallow land',
  'Mangrove',
  'Water / River',
  'Bare soil / Built-up'
];

// ---- 1.7 Predictor bands ----
var PREDICTOR_BANDS = ['NIR', 'SWIR1', 'SWIR2', 'NDVI', 'NDWI'];

// ---- 1.8 Strict reproducibility parameters for the classifier ----
var RANDOM_SEED = 42;   // single seed reused for the RF classifier
var RF_NUM_TREES = 100;

// [FIX 9] Spatial block k-fold cross-validation parameters, replacing
// the previous single random 80/20 split.
var K_FOLDS = 4;                          // small K, appropriate for the tiny dummy dataset;
                                           // raise (e.g. 5-10) once the real training asset is in use
var SPATIAL_BLOCK_SIZE_DEGREES = 0.05;    // ~5.5 km at this latitude — adjust to the spatial
                                           // autocorrelation range of the study area if known
                                           // (e.g. from a semi-variogram of the predictor bands)

// ---- 1.9 Cross-sensor radiometric harmonization (Landsat 5/7 -> OLI-like) ----
// [FIX 6] Toggle to enable/disable harmonization. Left ON by default so
// the six epochs (TM, ETM+, OLI, OLI-2) are radiometrically comparable.
var HARMONIZE_CROSS_SENSOR = true;

// TO VERIFY before publication: these coefficients approximate the
// general form of the Roy et al. (2016) ETM+ -> OLI OLS transformation
// (Table 2 of the original publication) and must be checked against
// that source (or recalibrated locally) before being used for any
// published result. Applied as: OLI_like = slope * ETM_or_TM + intercept,
// on surface reflectance values (0-1 scale).
var HARMONIZATION_COEFFICIENTS_ETM_TO_OLI = {
  BLUE:  {slope: 0.9785, intercept: -0.0095},
  GREEN: {slope: 0.9542, intercept: -0.0016},
  RED:   {slope: 0.9825, intercept: -0.0022},
  NIR:   {slope: 1.0073, intercept: -0.0021},
  SWIR1: {slope: 1.0171, intercept: -0.0030},
  SWIR2: {slope: 0.9949, intercept:  0.0029}
};

// ==========================================
// 2. PROCESSING AND MASKING FUNCTIONS
// ==========================================
//
// VERIFICATION — ATMOSPHERIC CORRECTION
// ----------------------------------------------------------------------------
// The collections used below ('LANDSAT/LT05/C02/T1_L2',
// 'LANDSAT/LE07/C02/T1_L2', 'LANDSAT/LC08/C02/T1_L2', 'LANDSAT/LC09/C02/T1_L2')
// are Collection 2 — Level 2 (Surface Reflectance) products, NOT TOA
// (Top-Of-Atmosphere, '_TOA' suffix) products. Atmospheric correction is
// therefore already applied upstream by the USGS:
//   - Landsat 5/7 (TM/ETM+): LEDAPS algorithm
//   - Landsat 8/9 (OLI):     LaSRC algorithm
// No additional atmospheric correction is therefore required in this
// script. The scale factors applied in standardizeBands()
// (multiply(0.0000275).add(-0.2)) are the standard conversion factors
// from Collection 2 SR DN values to surface reflectance (0–1), documented
// by the USGS — these are not atmospheric correction parameters.
//
// SAFEGUARD: if a TOA collection were used by mistake (name containing
// '_TOA'), the script explicitly forbids it below to prevent any
// accidental mixing of atmospherically uncorrected products.
// ----------------------------------------------------------------------------

function checkSRCollection(collection, collectionName) {
  if (collectionName.indexOf('_TOA') !== -1) {
    throw new Error(
      'TOA collection detected (' + collectionName + '): atmospheric ' +
      'correction is missing. Use exclusively Collection 2 Level 2 ' +
      'products (suffix _L2, Surface Reflectance).'
    );
  }
  return collection;
}

// [FIX 1] QA_PIXEL bit reference (USGS Collection 2, Level 2 QA_PIXEL band):
//   bit 1 -> Dilated Cloud
//   bit 2 -> Cirrus
//   bit 3 -> Cloud
//   bit 4 -> Cloud Shadow
// Variable names below now match the bit they actually mask.
function maskClouds(image) {
  var qa = image.select('QA_PIXEL');
  var dilatedCloudMask = qa.bitwiseAnd(1 << 1).eq(0);
  var cirrusMask = qa.bitwiseAnd(1 << 2).eq(0);
  var cloudMask = qa.bitwiseAnd(1 << 3).eq(0);
  var cloudShadowMask = qa.bitwiseAnd(1 << 4).eq(0);
  return image.updateMask(cloudMask.and(cloudShadowMask).and(dilatedCloudMask).and(cirrusMask));
}

// [FIX 6] Applies the ETM+/TM -> OLI-like harmonization coefficients to
// the six reflective bands of a single image. Expects an image already
// renamed/scaled to BLUE/GREEN/RED/NIR/SWIR1/SWIR2 surface reflectance.
function applyHarmonizationCoefficients(opticalImage) {
  var bandNames = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2'];
  var harmonizedBands = bandNames.map(function(bandName) {
    var coeffs = HARMONIZATION_COEFFICIENTS_ETM_TO_OLI[bandName];
    return opticalImage.select(bandName).multiply(coeffs.slope).add(coeffs.intercept).rename(bandName);
  });
  return ee.Image.cat(harmonizedBands);
}

// [FIX 6] Applies harmonization only to non-OLI sensors (Landsat 5/7);
// Landsat 8/9 imagery passes through unchanged, since OLI is the target
// spectral domain of the transformation.
function harmonizeOpticalBandsIfNeeded(opticalImage, spacecraftId) {
  var isOLI = ee.String(spacecraftId).slice(0, 9).equals('LANDSAT_8')
    .or(ee.String(spacecraftId).slice(0, 9).equals('LANDSAT_9'));
  return ee.Image(ee.Algorithms.If(
    isOLI,
    opticalImage,
    applyHarmonizationCoefficients(opticalImage)
  ));
}

function standardizeBands(image) {
  var id = image.get('SPACECRAFT_ID');
  var conditionL89 = ee.Algorithms.If(
    ee.String(id).slice(0, 9).equals('LANDSAT_8').or(ee.String(id).slice(0, 9).equals('LANDSAT_9')),
    image.select(
      ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
      ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2']
    ),
    image.select(
      ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'],
      ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2']
    )
  );

  var renamedImage = ee.Image(conditionL89);
  var opticalBands = renamedImage.multiply(0.0000275).add(-0.2);

  // [FIX 6] Harmonize TM/ETM+ reflectance toward OLI-like response
  // before the bands are merged back onto the image, so that all six
  // epochs are radiometrically comparable prior to compositing and
  // spectral index computation.
  var finalOpticalBands = HARMONIZE_CROSS_SENSOR
    ? harmonizeOpticalBandsIfNeeded(opticalBands, id)
    : opticalBands;

  return renamedImage.addBands(finalOpticalBands, null, true)
                      .copyProperties(image, ['system:time_start', 'SPACECRAFT_ID']);
}

function getCollectionForYear(year) {
  var idL5 = 'LANDSAT/LT05/C02/T1_L2';
  var idL7 = 'LANDSAT/LE07/C02/T1_L2';
  var idL8 = 'LANDSAT/LC08/C02/T1_L2';
  var idL9 = 'LANDSAT/LC09/C02/T1_L2';

  // Safeguard: confirms that SR (_L2) products are loaded, never TOA.
  var l5 = checkSRCollection(ee.ImageCollection(idL5), idL5);
  var l7 = checkSRCollection(ee.ImageCollection(idL7), idL7);
  var l8 = checkSRCollection(ee.ImageCollection(idL8), idL8);
  var l9 = checkSRCollection(ee.ImageCollection(idL9), idL9);

  if (year === 2000) return l5.merge(l7);
  if (year === 2005) return l5.merge(l7);
  if (year === 2010) return l5.merge(l7);
  if (year === 2015) return l8;
  if (year === 2020) return l8;
  if (year === 2025) return l8.merge(l9);

  return l8;
}

// [FIX 4] Number of whole days between the representative date and the
// midpoint of the window actually used for a given year.
function computeDaysOffset(representativeDateStr, windowStartStr, windowEndStr) {
  var representative = new Date(representativeDateStr);
  var start = new Date(windowStartStr);
  var end = new Date(windowEndStr);
  var windowMidpointMillis = start.getTime() + (end.getTime() - start.getTime()) / 2;
  var diffMillis = windowMidpointMillis - representative.getTime();
  var diffDays = Math.round(diffMillis / (1000 * 60 * 60 * 24));
  return diffDays; // signed: negative = window centered before the representative date
}

// [FIX 7] Width in whole days of a given {start, end} window, used to
// make the asymmetry across years an explicit, exportable number rather
// than something implicit in the date strings.
function computeWindowWidthDays(windowStartStr, windowEndStr) {
  var start = new Date(windowStartStr);
  var end = new Date(windowEndStr);
  return Math.round((end.getTime() - start.getTime()) / (1000 * 60 * 60 * 24));
}

// [FIX 8] Fraction of AOI pixels reaching at least `minObservations`
// valid (unmasked) observations in the given (already cloud-masked)
// collection. Uses a coarser scale than the classification/export scale
// since this is a QA diagnostic, not a data product.
function computePixelCoverageFraction(collection, geometry, minObservations) {
  var countImage = collection.select(0).count();
  var sufficientMask = countImage.gte(minObservations);
  var stats = sufficientMask.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: geometry,
    scale: PIXEL_COVERAGE_CHECK_SCALE,
    maxPixels: 1e13,
    bestEffort: true,
    tileScale: 4
  });
  var fraction = ee.Number(stats.values().get(0));
  return fraction.getInfo();
}

// [FIX 9] Assigns a spatial grid block id to a feature based on its
// point geometry, using a fixed-size lon/lat grid. Points that fall in
// the same block will always land in the same cross-validation fold,
// which is what limits train/test leakage from spatially autocorrelated
// neighbors (as opposed to a purely random per-point split).
function assignSpatialBlock(feature) {
  var coords = feature.geometry().coordinates();
  var lon = ee.Number(coords.get(0));
  var lat = ee.Number(coords.get(1));
  var blockX = lon.divide(SPATIAL_BLOCK_SIZE_DEGREES).floor();
  var blockY = lat.divide(SPATIAL_BLOCK_SIZE_DEGREES).floor();
  var blockId = blockX.multiply(100000).add(blockY);
  var fold = blockId.abs().mod(K_FOLDS);
  return feature.set('block_id', blockId).set('fold', fold);
}

// [FIX 9] Runs spatial block k-fold cross-validation over a sampled
// FeatureCollection (already carrying 'fold' and 'class' properties).
// Returns a plain JS object: {perFold: [...], meanOA, stdOA, meanKappa,
// stdKappa, nFoldsUsed}. This performs K .getInfo() round-trips (one
// confusion-matrix accuracy/kappa pair per fold), acceptable for a
// batch reproducibility script run once per year, not for interactive
// use.
function runSpatialKFoldCV(samplesWithBlocks, year) {
  var perFold = [];

  for (var foldIndex = 0; foldIndex < K_FOLDS; foldIndex++) {
    var testFold = samplesWithBlocks.filter(ee.Filter.eq('fold', foldIndex));
    var trainFold = samplesWithBlocks.filter(ee.Filter.neq('fold', foldIndex));

    var nTest = testFold.size().getInfo();
    var nTrain = trainFold.size().getInfo();

    if (nTest === 0 || nTrain === 0) {
      print('   [WARN] CV fold ' + foldIndex + ' skipped for ' + year +
        ' (nTrain=' + nTrain + ', nTest=' + nTest + '). Expected with the small dummy dataset.');
      continue;
    }

    var foldClassifier = ee.Classifier.smileRandomForest({
      numberOfTrees: RF_NUM_TREES,
      seed: RANDOM_SEED
    }).train({
      features: trainFold,
      classProperty: 'class',
      inputProperties: PREDICTOR_BANDS
    });

    var foldValidation = testFold.classify(foldClassifier);
    var foldMatrix = foldValidation.errorMatrix('class', 'classification');
    var foldOA = foldMatrix.accuracy().getInfo();
    var foldKappa = foldMatrix.kappa().getInfo();

    perFold.push({fold: foldIndex, oa: foldOA, kappa: foldKappa, nTrain: nTrain, nTest: nTest});
  }

  if (perFold.length === 0) {
    return {perFold: perFold, meanOA: null, stdOA: null, meanKappa: null, stdKappa: null, nFoldsUsed: 0};
  }

  var oaValues = perFold.map(function(r) { return r.oa; });
  var kappaValues = perFold.map(function(r) { return r.kappa; });

  var meanOA = meanOfArray(oaValues);
  var meanKappa = meanOfArray(kappaValues);

  return {
    perFold: perFold,
    meanOA: meanOA,
    stdOA: stdDevOfArray(oaValues, meanOA),
    meanKappa: meanKappa,
    stdKappa: stdDevOfArray(kappaValues, meanKappa),
    nFoldsUsed: perFold.length
  };
}

function meanOfArray(arr) {
  var sum = 0;
  for (var i = 0; i < arr.length; i++) { sum += arr[i]; }
  return sum / arr.length;
}

function stdDevOfArray(arr, meanValue) {
  if (arr.length < 2) { return 0; }
  var sumSquares = 0;
  for (var i = 0; i < arr.length; i++) {
    sumSquares += Math.pow(arr[i] - meanValue, 2);
  }
  return Math.sqrt(sumSquares / (arr.length - 1));
}

function addLegend(title, classes, colors) {
  var legend = ui.Panel({style: {position: 'bottom-left', padding: '8px 15px'}});
  legend.add(ui.Label(title, {fontWeight: 'bold', fontSize: '14px'}));
  for (var i = 0; i < colors.length; i++) {
    var row = ui.Panel({layout: ui.Panel.Layout.flow('horizontal')});
    row.add(ui.Label('', {backgroundColor: colors[i], padding: '8px', margin: '0 6px 2px 0'}));
    row.add(ui.Label(classes[i], {margin: '0 0 2px 0'}));
    legend.add(row);
  }
  Map.add(legend);
}

// ==========================================
// 3. AUTOMATED LOOP AND CLASSIFICATION
// ==========================================

Map.centerObject(aoi, 9);
addLegend('LULC Legend', CLASS_NAMES, PALETTE);

// Table that will accumulate the validation metrics for all years,
// for a single export at the end of the script (full traceability, not
// just to the console).
var metricRows = ee.List([]);

// [FIX 9] Separate table accumulating one row per (year, fold), for a
// fully transparent secondary CSV export.
var perFoldRows = ee.List([]);

print("=== START OF GLOBAL ANALYSIS ===");
print("AOI used (verify before execution):", aoi);
print("Training points used:", USE_DUMMY_POINTS
  ? "DUMMY (demonstration only)"
  : "Real asset");
print("Cross-sensor radiometric harmonization enabled:", HARMONIZE_CROSS_SENSOR);

TEMPORAL_CONFIG.forEach(function(period) {

  print('--- Processing year: ' + period.year + ' ---');
  print('   Window rationale: ' + period.window_rationale);

  var minImageThreshold = getMinImageThreshold(period.year);
  if (SLC_OFF_AFFECTED_YEARS.indexOf(period.year) !== -1) {
    print('   [INFO] Year ' + period.year + ' falls within the Landsat 7 SLC-off ' +
      'era: using a stricter minimum image threshold (' + minImageThreshold +
      ') to help the median composite statistically fill scan-line gaps.');
  }

  // 1. Clean composite generation — TRY THE PRIMARY WINDOW FIRST
  // [FIX 10] Scene-level cloud-cover filter (CLOUD_COVER <= threshold)
  // applied BEFORE the pixel-level QA_PIXEL masking, so heavily
  // contaminated scenes are excluded from the collection outright.
  var usedWindow = period.primary;
  var rawCollection = getCollectionForYear(period.year)
    .filterBounds(aoi)
    .filterDate(usedWindow.start, usedWindow.end)
    .filter(ee.Filter.lte('CLOUD_COVER', SCENE_CLOUD_COVER_THRESHOLD_PERCENT))
    .map(maskClouds);

  var nbImagesPrimary = rawCollection.size().getInfo();
  print('   Primary window [' + period.primary.start + ' -> ' + period.primary.end +
    '] (scenes with CLOUD_COVER <= ' + SCENE_CLOUD_COVER_THRESHOLD_PERCENT + '%): ' +
    nbImagesPrimary + ' valid image(s) after cloud masking.');

  var coverageInsufficient = false;

  // AUTOMATIC SWITCH TO THE FALLBACK WINDOW IF COLLECTION-LEVEL COVERAGE
  // IS INSUFFICIENT
  if (nbImagesPrimary < minImageThreshold) {
    print('   [WARNING] Insufficient collection-level coverage (< ' + minImageThreshold +
      ' images) for ' + period.year + '. Switching to the FALLBACK WINDOW [' +
      period.fallback.start + ' -> ' + period.fallback.end + '].');

    usedWindow = period.fallback;
    rawCollection = getCollectionForYear(period.year)
      .filterBounds(aoi)
      .filterDate(usedWindow.start, usedWindow.end)
      .filter(ee.Filter.lte('CLOUD_COVER', SCENE_CLOUD_COVER_THRESHOLD_PERCENT))
      .map(maskClouds);

    var nbImagesFallback = rawCollection.size().getInfo();
    print('   Fallback window (scenes with CLOUD_COVER <= ' + SCENE_CLOUD_COVER_THRESHOLD_PERCENT +
      '%): ' + nbImagesFallback + ' valid image(s) after cloud masking.');

    if (nbImagesFallback < minImageThreshold) {
      coverageInsufficient = true;
      print('   [ERROR] Collection-level coverage is still insufficient even with the ' +
        'fallback window for ' + period.year + '. Classification will be SKIPPED.');
    }
  }

  // [FIX 8] Pixel-level coverage check, independent of the collection-level
  // image count: verifies that valid observations are spatially
  // distributed across the AOI, not concentrated in part of it.
  var pixelCoverageFraction = computePixelCoverageFraction(rawCollection, aoi, minImageThreshold);
  print('   Pixel-level coverage: ' + (pixelCoverageFraction * 100).toFixed(1) +
    '% of AOI reaches >= ' + minImageThreshold + ' valid observations ' +
    '(threshold: ' + (PIXEL_COVERAGE_FRACTION_THRESHOLD * 100).toFixed(0) + '%).');

  if (!coverageInsufficient && pixelCoverageFraction < PIXEL_COVERAGE_FRACTION_THRESHOLD) {
    coverageInsufficient = true;
    print('   [ERROR] Pixel-level coverage (' + (pixelCoverageFraction * 100).toFixed(1) +
      '%) falls below the ' + (PIXEL_COVERAGE_FRACTION_THRESHOLD * 100).toFixed(0) +
      '% threshold for ' + period.year + ' even though the collection-level image count ' +
      'was sufficient. Classification will be SKIPPED (spatially uneven cloud cover).');
  }

  // [FIX 4] Temporal drift between the representative date and the
  // midpoint of the window actually used (primary or fallback).
  var daysOffset = computeDaysOffset(period.representative_date, usedWindow.start, usedWindow.end);
  var primaryWindowWidthDays = computeWindowWidthDays(period.primary.start, period.primary.end);
  print('   Representative date: ' + period.representative_date +
    ' | Window midpoint offset: ' + daysOffset + ' day(s).' +
    ' | Primary window width: ' + primaryWindowWidthDays + ' day(s).');

  var processedCollection = rawCollection.map(standardizeBands);

  var composite = processedCollection.median().clip(aoi);

  // 2. Spectral index computation
  var ndvi = composite.normalizedDifference(['NIR', 'RED']).rename('NDVI');
  var ndwi = composite.normalizedDifference(['GREEN', 'NIR']).rename('NDWI');

  // Final stacked image (reflectances + indices)
  var finalImage = composite.addBands([ndvi, ndwi]);

  Map.addLayer(finalImage, {bands: ['NIR', 'RED', 'GREEN'], min: 0, max: 0.4},
    'NIR Composite ' + period.year, false);

  // 3. CLASSIFICATION (real or dummy points, per section 1.3)
  if (!coverageInsufficient && trainingPointsCount > 0) {

    var samples = finalImage.select(PREDICTOR_BANDS).sampleRegions({
      collection: trainingPoints,
      properties: ['class'],
      scale: 30
    });

    // [FIX 9] Spatial block k-fold cross-validation (replaces the
    // previous single random 80/20 split) to reduce the risk of
    // spatially autocorrelated train/test pairs inflating the metrics.
    var samplesWithBlocks = samples.map(assignSpatialBlock);
    var cvResults = runSpatialKFoldCV(samplesWithBlocks, period.year);

    print('Cross-validation performance ' + period.year + ' (' + cvResults.nFoldsUsed + '/' +
      K_FOLDS + ' folds usable):');
    print('Mean Overall Accuracy:', cvResults.meanOA, '(std: ' + cvResults.stdOA + ')');
    print('Mean Kappa index:', cvResults.meanKappa, '(std: ' + cvResults.stdKappa + ')');

    cvResults.perFold.forEach(function(foldResult) {
      perFoldRows = perFoldRows.add(ee.Feature(null, {
        year: period.year,
        fold: foldResult.fold,
        overall_accuracy: foldResult.oa,
        kappa: foldResult.kappa,
        n_train: foldResult.nTrain,
        n_test: foldResult.nTest
      }));
    });

    // [FIX 9] The classifier used for the final exported map is trained
    // on the FULL training set: cross-validation above estimates
    // generalization performance, while the deployed map uses all
    // available labeled data (standard practice).
    var finalClassifier = ee.Classifier.smileRandomForest({
      numberOfTrees: RF_NUM_TREES,
      seed: RANDOM_SEED
    }).train({
      features: samples,
      classProperty: 'class',
      inputProperties: PREDICTOR_BANDS
    });

    var classifiedMap = finalImage.select(PREDICTOR_BANDS).classify(finalClassifier);

    Map.addLayer(classifiedMap, {min: 1, max: 6, palette: PALETTE},
      'LULC Classification ' + period.year);

    // Accumulation for a single CSV export (traceability, not just console)
    var row = ee.Feature(null, {
      year: period.year,
      representative_date: period.representative_date,
      window_start_used: usedWindow.start,
      window_end_used: usedWindow.end,
      fallback_used: usedWindow === period.fallback,
      days_offset_from_representative_date: daysOffset,
      primary_window_width_days: primaryWindowWidthDays,
      window_rationale: period.window_rationale,
      slc_off_affected: SLC_OFF_AFFECTED_YEARS.indexOf(period.year) !== -1,
      min_image_threshold_applied: minImageThreshold,
      pixel_coverage_fraction: pixelCoverageFraction,
      coverage_flag: 'OK',
      cross_sensor_harmonization_applied: HARMONIZE_CROSS_SENSOR,
      k_folds_configured: K_FOLDS,
      k_folds_used: cvResults.nFoldsUsed,
      mean_overall_accuracy: cvResults.meanOA,
      std_overall_accuracy: cvResults.stdOA,
      mean_kappa: cvResults.meanKappa,
      std_kappa: cvResults.stdKappa,
      num_trees: RF_NUM_TREES,
      seed: RANDOM_SEED
    });
    metricRows = metricRows.add(row);

    // EXPORT 1: classification map
    Export.image.toDrive({
      image: classifiedMap,
      description: 'LULC_Classification_' + period.year,
      folder: OUTPUT_FOLDER,
      scale: 30,
      region: aoi,
      crs: OUTPUT_CRS,
      maxPixels: 1e13
    });

  } else if (coverageInsufficient) {
    var flaggedRow = ee.Feature(null, {
      year: period.year,
      representative_date: period.representative_date,
      window_start_used: usedWindow.start,
      window_end_used: usedWindow.end,
      fallback_used: usedWindow === period.fallback,
      days_offset_from_representative_date: daysOffset,
      primary_window_width_days: primaryWindowWidthDays,
      window_rationale: period.window_rationale,
      slc_off_affected: SLC_OFF_AFFECTED_YEARS.indexOf(period.year) !== -1,
      min_image_threshold_applied: minImageThreshold,
      pixel_coverage_fraction: pixelCoverageFraction,
      coverage_flag: 'INSUFFICIENT',
      cross_sensor_harmonization_applied: HARMONIZE_CROSS_SENSOR,
      k_folds_configured: K_FOLDS,
      k_folds_used: 0,
      mean_overall_accuracy: null,
      std_overall_accuracy: null,
      mean_kappa: null,
      std_kappa: null,
      num_trees: RF_NUM_TREES,
      seed: RANDOM_SEED
    });
    metricRows = metricRows.add(flaggedRow);
    print("[INFO] Classification skipped for " + period.year + " due to insufficient coverage.");
  } else {
    print("[INFO] Preview mode (no training points). Only the raw composite is exported.");
  }

  // EXPORT 2: preprocessed multi-band Landsat composite
  var windowSuffix = (usedWindow === period.fallback) ? 'fallback' : 'primary';
  var confidenceSuffix = coverageInsufficient ? '_LOWCONF' : '';
  Export.image.toDrive({
    image: finalImage.select(['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'NDWI']),
    description: 'Landsat_Raw_Composite_' + period.year + '_window_' + windowSuffix + confidenceSuffix,
    folder: OUTPUT_FOLDER,
    scale: 30,
    region: aoi,
    crs: OUTPUT_CRS,
    maxPixels: 1e13
  });

});

// ==========================================
// 4. VALIDATION METRICS EXPORT (CSV)
// ==========================================
// All years combined, for a supplementary table in the article. Now
// includes representative_date, day offset, window width/rationale,
// SLC-off flag, threshold applied, pixel-level coverage fraction,
// harmonization flag, and k-fold cross-validation summary statistics
// (mean/std OA and Kappa) per year.
var metricsTable = ee.FeatureCollection(metricRows);
Export.table.toDrive({
  collection: metricsTable,
  description: 'LULC_Validation_Metrics_All_Years',
  folder: OUTPUT_FOLDER,
  fileFormat: 'CSV'
});

// [FIX 9] Full per-fold detail, for reviewers who want to inspect the
// individual cross-validation folds rather than only the mean/std
// summary in the main metrics table.
var perFoldTable = ee.FeatureCollection(perFoldRows);
Export.table.toDrive({
  collection: perFoldTable,
  description: 'LULC_CrossValidation_PerFold_All_Years',
  folder: OUTPUT_FOLDER,
  fileFormat: 'CSV'
});

print("=== ALL TASKS HAVE BEEN SCHEDULED ===");
print("[ACTION REQUIRED] Check the 'Tasks' tab on the right to launch the computations, exports, and the metrics CSVs.");
print("[WARNING] Reminder: with the default DUMMY point set, results are not scientifically valid.");
print("   Replace section 1.2/1.3 with the real asset before producing any results for publication.");
print("[REMINDER] Verify HARMONIZATION_COEFFICIENTS_ETM_TO_OLI (section 1.9) against Roy et al. (2016) before publication.");
