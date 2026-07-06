// =========================================================================
// REPRODUCIBILITY SCRIPT — MULTI-TEMPORAL LULC CLASSIFICATION
// Mvouti district, Mayombe forest (Republic of Congo) — 2000, 2005,
// 2010, 2015, 2020, 2025
// Automated Random Forest classification over 6 dates
// =========================================================================
//
// REPRODUCIBILITY DECLARATION / CODE & DATA AVAILABILITY STATEMENT
// -------------------------------------------------------------------
// Associated article: [Full article title], [Authors], [Journal, year]
// Script publicly deposited at: [GEE "Get Link" and/or Zenodo DOI]
// Deposit date / version: [DD/MM/YYYY] — [v1.0]
//
// Training data: the dataset is not publicly available and is not
// distributed through this repository due to project data ownership and
// confidentiality restrictions (see ../../DATA_AVAILABILITY.md).
// A set of DUMMY points is provided below (section 1.3) solely to
// allow an end-to-end demonstration run of the script.
// To reproduce the exact results of the article, replace this dummy
// set with the real asset indicated in the comment.
//
// Temporal windows: each target year uses a PRIMARY window
// (section 1.4); if cloud cover is insufficient (< MIN_IMAGE_THRESHOLD
// valid images), the script automatically switches to a wider FALLBACK
// window. The window actually used is logged to the console and
// exported to the metrics CSV for full traceability.
//
// Atmospheric correction: already applied upstream (Collection 2
// Level 2 Surface Reflectance products, LEDAPS/LaSRC algorithms — see
// section 2 for details and the anti-TOA safeguard).
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
// be used to produce scientific results.
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

// ---- 1.4 Annual date configuration: PRIMARY WINDOW + FALLBACK WINDOW ----
// Primary windows: balance between cloud cover and temporal consistency,
// tightened or widened according to known sensor availability per period
// (Landsat 7 SLC-off gap post-2003, Landsat 5 end-of-life, Landsat 8/9
// stability).
//
// FALLBACK window: if the primary window does not provide a minimum
// number of usable images after cloud filtering (e.g. a very cloudy zone
// in a given year), the script automatically widens the search period for
// that year to avoid an empty or overly noisy composite — at the cost of
// a larger temporal offset from the target year, explicitly flagged in
// the output.
var TEMPORAL_CONFIG = [
  {
    year: 2000,
    primary:  {start: '1999-01-01', end: '2001-12-31'},   // 3-year window (sound)
    fallback: {start: '1998-01-01', end: '2002-12-31'}    // ±1-year widening
  },
  {
    year: 2005,
    primary:  {start: '2004-01-01', end: '2006-12-31'},   // 3-year window (handling the L7 SLC-off gap)
    fallback: {start: '2003-01-01', end: '2007-12-31'}
  },
  {
    year: 2010,
    primary:  {start: '2009-01-01', end: '2011-12-31'},   // 3-year window (L5 end-of-life)
    fallback: {start: '2008-01-01', end: '2012-12-31'}
  },
  {
    year: 2015,
    primary:  {start: '2014-01-01', end: '2015-12-31'},   // 2-year window (L8 stability)
    fallback: {start: '2013-01-01', end: '2016-12-31'}
  },
  {
    year: 2020,
    primary:  {start: '2019-01-01', end: '2020-12-31'},   // 2-year window (L8)
    fallback: {start: '2018-01-01', end: '2021-12-31'}
  },
  {
    year: 2025,
    primary:  {start: '2024-01-01', end: '2025-12-31'},   // 2-year window (maximum L8 & L9 recency)
    fallback: {start: '2022-01-01', end: '2025-12-31'}    // No widening beyond 2025
  }
];

// Minimum number of images (after cloud masking) below which the script
// switches to the fallback window for the year concerned.
var MIN_IMAGE_THRESHOLD = 3;

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
var RANDOM_SEED = 42;   // single seed reused for both the split and the RF
var RF_NUM_TREES = 100;
var TRAINING_RATIO = 0.8;

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

function maskClouds(image) {
  var qa = image.select('QA_PIXEL');
  var cloudShadowMask = qa.bitwiseAnd(1 << 3).eq(0);
  var cloudMask = qa.bitwiseAnd(1 << 4).eq(0);
  var dilatedCloud = qa.bitwiseAnd(1 << 1).eq(0);
  var cirrusMask = qa.bitwiseAnd(1 << 2).eq(0);
  return image.updateMask(cloudShadowMask.and(cloudMask).and(dilatedCloud).and(cirrusMask));
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

  return renamedImage.addBands(opticalBands, null, true)
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

print("=== START OF GLOBAL ANALYSIS ===");
print("AOI used (verify before execution):", aoi);
print("Training points used:", USE_DUMMY_POINTS
  ? "DUMMY (demonstration only)"
  : "Real asset");

TEMPORAL_CONFIG.forEach(function(period) {

  print('--- Processing year: ' + period.year + ' ---');

  // 1. Clean composite generation — TRY THE PRIMARY WINDOW FIRST
  var usedWindow = period.primary;
  var rawCollection = getCollectionForYear(period.year)
    .filterBounds(aoi)
    .filterDate(usedWindow.start, usedWindow.end)
    .map(maskClouds);

  var nbImagesPrimary = rawCollection.size().getInfo();
  print('   Primary window [' + period.primary.start + ' -> ' + period.primary.end +
    ']: ' + nbImagesPrimary + ' valid image(s) after cloud masking.');

  // AUTOMATIC SWITCH TO THE FALLBACK WINDOW IF COVERAGE IS INSUFFICIENT
  if (nbImagesPrimary < MIN_IMAGE_THRESHOLD) {
    print('   [WARNING] Insufficient coverage (< ' + MIN_IMAGE_THRESHOLD + ' images) for ' +
      period.year + '. Switching to the FALLBACK WINDOW [' +
      period.fallback.start + ' -> ' + period.fallback.end + '].');

    usedWindow = period.fallback;
    rawCollection = getCollectionForYear(period.year)
      .filterBounds(aoi)
      .filterDate(usedWindow.start, usedWindow.end)
      .map(maskClouds);

    var nbImagesFallback = rawCollection.size().getInfo();
    print('   Fallback window: ' + nbImagesFallback + ' valid image(s) after cloud masking.');

    if (nbImagesFallback < MIN_IMAGE_THRESHOLD) {
      print('   [ERROR] Coverage is still insufficient even with the fallback window for ' +
        period.year + '. The resulting composite must be interpreted with caution ' +
        '(risk of gaps or residual noise).');
    }
  }

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
  if (trainingPoints.size().getInfo() > 0) {

    var samples = finalImage.select(PREDICTOR_BANDS).sampleRegions({
      collection: trainingPoints,
      properties: ['class'],
      scale: 30
    });

    // Train/test split with a fixed, explicit seed
    var dataset = samples.randomColumn('random', RANDOM_SEED);
    var trainSet = dataset.filter(ee.Filter.lessThan('random', TRAINING_RATIO));
    var testSet = dataset.filter(ee.Filter.greaterThanOrEquals('random', TRAINING_RATIO));

    // Random Forest with explicit seed (strict reproducibility)
    var classifier = ee.Classifier.smileRandomForest({
      numberOfTrees: RF_NUM_TREES,
      seed: RANDOM_SEED
    }).train({
      features: trainSet,
      classProperty: 'class',
      inputProperties: PREDICTOR_BANDS
    });

    var classifiedMap = finalImage.select(PREDICTOR_BANDS).classify(classifier);

    Map.addLayer(classifiedMap, {min: 1, max: 6, palette: PALETTE},
      'LULC Classification ' + period.year);

    // Validation
    var validation = testSet.classify(classifier);
    var matrix = validation.errorMatrix('class', 'classification');
    var oa = matrix.accuracy();
    var kappa = matrix.kappa();

    print('Validation performance ' + period.year + ':');
    print('Overall Accuracy:', oa);
    print('Kappa index:', kappa);

    // Accumulation for a single CSV export (traceability, not just console)
    // Includes the temporal window actually used (primary or fallback)
    // for full traceability toward the reviewer.
    var row = ee.Feature(null, {
      year: period.year,
      window_start_used: usedWindow.start,
      window_end_used: usedWindow.end,
      fallback_used: usedWindow === period.fallback,
      overall_accuracy: oa,
      kappa: kappa,
      num_trees: RF_NUM_TREES,
      seed: RANDOM_SEED,
      training_ratio: TRAINING_RATIO
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

  } else {
    print("[INFO] Preview mode (no training points). Only the raw composite is exported.");
  }

  // EXPORT 2: preprocessed multi-band Landsat composite
  // The description name includes the window actually used (P = primary,
  // F = fallback) so that traceability is directly visible in the Tasks tab.
  var windowSuffix = (usedWindow === period.fallback) ? 'fallback' : 'primary';
  Export.image.toDrive({
    image: finalImage.select(['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'NDWI']),
    description: 'Landsat_Raw_Composite_' + period.year + '_window_' + windowSuffix,
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
// All years combined, for a supplementary table in the article.
var metricsTable = ee.FeatureCollection(metricRows);
Export.table.toDrive({
  collection: metricsTable,
  description: 'LULC_Validation_Metrics_All_Years',
  folder: OUTPUT_FOLDER,
  fileFormat: 'CSV'
});

print("=== ALL TASKS HAVE BEEN SCHEDULED ===");
print("[ACTION REQUIRED] Check the 'Tasks' tab on the right to launch the computations, exports, and the metrics CSV.");
print("[WARNING] Reminder: with the default DUMMY point set, results are not scientifically valid.");
print("   Replace section 1.2/1.3 with the real asset before producing any results for publication.");
