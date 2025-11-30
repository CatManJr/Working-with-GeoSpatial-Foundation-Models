// =================================================================================
// Hurricane Helene 2024 | Fort Myers
// L1C + Cloud Score+ Cloud Shadow Mask (SAR no mask version) - Demo version (using official boundary)
// =================================================================================

// This script fetches and visualizes the GeoTIFF in 'raw' folder for this project.
// Run this script on GEE code editor.

// Minimum Rectangular of Fort Myers Boundary
var aoi = fort_myers.geometry().bounds()

// Time Range
var startDate = '2024-09-21';
var endDate   = '2024-09-29';

// Sentinel-2 L1C (no mask)
var s2Sr = ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
              .filterBounds(aoi)
              .filterDate(startDate, endDate);

// Cloud Score+
var csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
                .filterBounds(aoi)
                .filterDate(startDate, endDate);

// Sentinel-1 GRD (SAR no mask)
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
            .filterBounds(aoi)
            .filterDate(startDate, endDate)
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
            .select(['VV', 'VH']);

// AEF Annual Embeddings
var aef = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
             .filterDate('2024-01-01', '2025-01-01')
             .filterBounds(aoi)
             .mosaic().select('A.*');

// 7. Cloud Score+ Mask Extraction (for S2 training labels only)
var csMedian = csPlus.median().clip(aoi);
var cloudScore = csMedian.select('cs');

// Cloud mask: cloud area = 1, non-cloud area = 0
var cloudMask = cloudScore.lt(0.45);

// Composite Data
var s2Original = s2Sr.median().clip(aoi);
var s1Composite = s1.median().clip(aoi);  // SAR naturally unaffected by clouds
var aefComposite = aef.clip(aoi);

// Visualization Parameters
var visTrue = {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000, gamma: 1.4};
var visSAR  = {min: -20, max: 0};
var visAef  = {bands: ['A01', 'A16', 'A09'], min: -0.3, max: 0.3};

// Map Center & Layers
Map.centerObject(aoi, 12);

Map.addLayer(s2Original, visTrue, 'Helene S2 Original True Color');
Map.addLayer(s1Composite.select('VV'), visSAR, 'Helene SAR VV', false);
Map.addLayer(s1Composite.select('VH'), visSAR, 'Helene SAR VH', false);
Map.addLayer(aefComposite, visAef, 'Helene AEF Embeddings', false);
Map.addLayer(fort_myers, {color: 'blue'}, 'Fort Myers Boundary');

// Cloud Score+ Mask Visualization
Map.addLayer(cloudMask.selfMask(), {palette: ['yellow']}, 'Cloud Score+ Cloud Mask');


// Export Data
var prithviBands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'];

// Segmentation labels
var cloudMaskExport = cloudMask.uint8().rename('cloud').unmask(0).clip(aoi);      // cloud=1, non-cloud=0

// Original data (no mask)
var s2Export = s2Original.select(prithviBands)  // unmasked S2
                       .divide(10000)
                       .float();
var s1Export = s1Composite.select(['VV','VH']).clip(aoi);  // SAR no mask needed
var aefExport = aef.clip(aoi);

// Export function
function exportWhole(img, desc, folder){
  Export.image.toDrive({
    image        : img,
    description  : desc,
    folder       : folder,
    region       : aoi,
    scale        : 10,
    crs          : 'EPSG:32617',
    maxPixels    : 1e13,
    fileFormat   : 'GeoTIFF',
  });
}

// Export according to your segmentation requirements
exportWhole(cloudMaskExport,   'FortMyers_Helene2024_cloud_mask',  'FortMyers');  // cloud segmentation labels
exportWhole(s2Export,          'FortMyers_Helene2024_S2',          'FortMyers');  // unmasked optical
exportWhole(s1Export,          'FortMyers_Helene2024_S1',          'FortMyers');  // unmasked SAR
exportWhole(aefExport,         'FortMyers_Helene2024_AEF64',       'FortMyers');
