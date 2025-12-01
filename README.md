# Working with GeoSpatial Foundation Models
## Fort Myers Flood Risk Analysis & Dashboard for Hurricane Helene

A comprehensive geospatial analysis system for assessing flood exposure and population risk from Hurricane Helene 2024 in Fort Myers, Florida, using satellite imagery, machine learning-based water segmentation, and spatial accessibility modeling.

## Overview

This project implements an end-to-end workflow for flood risk assessment:

1. **Cloud Reconstruction**: Reconstruct cloud-obscured Sentinel-2 imagery using Sentinel-1 and satellite embeddings
2. **Water Segmentation**: Detect flood extent using IBM Prithvi foundation model
3. **Population Exposure**: Calculate exposed population using WorldPop data
4. **Risk Analysis**: Compute spatial accessibility-based risk scores (G2SFCA method)
5. **Web Dashboard**: Interactive visualization and UI for the analysis results.

**On CLoud Live Dashboard**: https://two024-hurricane-helene-flood-risk.onrender.com (you may need to wait for seconds to awake the Render server)  
**Source Code Repository**: https://github.com/CatManJr/Working-with-GeoSpatial-Foundation-Models

## Project Structure

```
Root/
├── paths.py                  # Centralized path configuration
├── pyproject.toml            # Python dependencies
├── Dockerfile                # Container configuration for deployment
├── clean_index.sh            # MacOS index cleanup utility
│
├── app/                      # Full-stack web application
│   ├── backend/              # FastAPI backend
│   │   ├── main.py           # API endpoints
│   │   ├── file_geodatabase.py  # Spatial data management
│   │   └── requirements.txt  # Python dependencies
│   ├── frontend/             # React dashboard
│   │   └── src/
│   │       ├── App.js        # Main UI component
│   │       └── App.css       # Styles
│   └── file_database/        # Organized geospatial data
│       ├── rasters/          # Risk layers, population, flood extent
│       ├── vectors/          # Boundaries, geometries
│       └── tables/           # Statistics (CSV format)
│
├── data/                     # Raw and processed data
│   ├── raw/                  # Satellite imagery from GEE
│   ├── processed/            # Cloud-free mosaics
│   ├── IBM/                  # Prithvi model inputs/outputs
│   ├── flood/                # Extracted flood extent
│   ├── NHD/                  # National Hydrology Dataset
│   ├── permanent_water/      # Permanent water features
│   ├── Fort_Myers_City_Boundary/  # Study area boundary
│   └── pop/                  # WorldPop 2024 (100m resolution)
│
├── GEE_script/               # Google Earth Engine data acquisition
│   └── fetch_data.js
│
├── reconstruct/              # Cloud removal workflow
│   ├── make_dataset.py       # Feature engineering for LightGBM
│   ├── train.py              # Train reconstruction model
│   ├── train.py              # Plot the regression metrics by bands
│   ├── reconstruct.py        # Apply model to cloudy pixels
│   └── viz.py                # Visualization
│
├── water_segmentation/       # Flood detection workflow
│   ├── prepare_Prithvi.py    # Tile preparation for foundation model
│   └── predict.py            # Prithvi inference by querying the official demo
│
├── flood_extract/            # Post-processing
│   ├── permanent_water.py    # Extract permanent water from NHD
│   └── extract_flood.py      # Isolate flood-only pixels
│
└── pop_exposure/             # Population exposure analysis
    ├── clip.py               # Clip population to study area
    ├── overlay.py            # Calculate exposed population
    └── accessibility.py      # G2SFCA risk modeling
```

## Installation

Install dependencies using uv (recommended) or pip:

```bash
# Using uv (fast dependency resolver)
pip install uv
uv sync

# Or using pip
pip install -r app/backend/requirements.txt
```

## Workflow

### 1. Data Acquisition
Run `GEE_script/fetch_data.js` in Google Earth Engine Code Editor to download:
- Sentinel-2 L1C imagery (2024-09-26, Hurricane Helene event)
- Sentinel-1 GRD backscatter
- ASTER GDEM elevation
- Cloud masks

### 2. Cloud Reconstruction
```bash
uv run reconstruct/check.py        # Validate input data
uv run reconstruct/make_dataset.py # Feature engineering
uv run reconstruct/train.py        # Train LightGBM models
uv run reconstruct/reconstruct.py  # Reconstruct cloudy pixels
uv run reconstruct/viz.py          # Generate visualizations
```

**Output**: `data/processed/S2_mosaic.tif` (cloud-free composite)

### 3. Permanent Water Extraction
```bash
uv run flood_extract/permanent_water.py
```

Extracts permanent water bodies from NHD (National Hydrology Dataset) to isolate flood-only areas.

### 4. Flood Detection
```bash
uv run water_segmentation/prepare_Prithvi.py  # Prepare 224x224 tiles
uv run water_segmentation/predict.py          # Run Prithvi model
uv run flood_extract/extract_flood.py         # Extract flood pixels
```

**Output**: 
- `data/flood/FortMyersHelene_2024T269_flood_clipped.tif` (raster)
- `data/flood/FortMyersHelene_2024T269_flood_clipped.shp` (vector)

### 5. Population Exposure Analysis
```bash
uv run pop_exposure/clip.py        # Clip WorldPop to study area
uv run pop_exposure/overlay.py     # Calculate exposed population
uv run pop_exposure/accessibility.py --bandwidth 500  # G2SFCA risk analysis
```

Generates risk layers at multiple bandwidths (250m, 500m, 1000m, 2500m).

### 6. Web Application

**Development**:
```bash
cd app
./set_up.sh     # Install dependencies
./run_dev.sh    # Start backend (port 8000) and frontend (port 3000)
```

**Production**:
```bash
./run_prod.sh   # Build frontend and serve with backend
```

**Deployment**: Application is containerized using Docker and deployed on Render.

## Key Technologies

**Geospatial**: rasterio, geopandas, shapely, GDAL  
**Machine Learning**: LightGBM, HuggingFace Transformers, IBM Prithvi  
**Web Stack**: FastAPI, React, Leaflet, Recharts  
**Deployment**: Docker, Render  

## Configuration

All file paths are centralized in `paths.py`:

```python
from paths import DATA_DIR, FLOOD_DIR, CITY_BOUNDARY

# Example usage
flood_raster = FLOOD_DIR / "FortMyersHelene_2024T269_flood_clipped.tif"
```

Key paths:
- `RAW_S2`, `RAW_S1`, `RAW_AEF` - Raw satellite data
- `S2_MOSAIC` - Reconstructed cloud-free mosaic
- `FLOOD_DIR` - Flood extent outputs
- `CITY_BOUNDARY` - Fort Myers boundary shapefile

### HuggingFace Token

For Prithvi model inference, obtain a token from https://huggingface.co/settings/tokens and either:
1. Save to `hf_token.txt` in project root, or
2. Set environment variable: `export HF_TOKEN='your_token'`

Alternatively, run inference locally (Windows/Linux) following: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11

## Output Files

**Cloud Reconstruction**:
- `data/processed/S2_mosaic.tif` - Cloud-free Sentinel-2 composite

**Flood Detection**:
- `data/IBM/predict/FortMyersHelene_2024T269_inundated.tif` - Water segmentation
- `data/flood/FortMyersHelene_2024T269_flood_clipped.{tif,shp}` - Final flood extent

**Population Exposure**:
- `data/pop_exposure/flood_risk_g2sfca_raster_{bandwidth}m.tif` - Risk layers
- `data/pop_exposure/flood_risk_g2sfca_raster_{bandwidth}m_summary.csv` - Statistics

**Web Application Data**:
- `app/file_database/rasters/` - All raster layers
- `app/file_database/vectors/` - Boundaries and geometries
- `app/file_database/tables/` - Statistical summaries

## Submitting Code as Appendix

For academic submissions with page limits, include only core components in the appendix:

**Recommended Appendix Structure** (~1,200 lines total):

```
Appendix A: Web Application Core (500 lines)
  A.1 Backend API (app/backend/main.py)
  A.2 File Geodatabase Manager (app/backend/file_geodatabase.py)
  A.3 Frontend Dashboard (app/frontend/src/App.js)

Appendix B: Spatial Analysis Algorithms (400 lines)
  B.1 G2SFCA Risk Model (pop_exposure/accessibility.py)
  B.2 Population Overlay (pop_exposure/overlay.py)

Appendix C: Data Processing (200 lines)
  C.1 Cloud Reconstruction (reconstruct/reconstruct.py)
  C.2 Flood Extraction (flood_extract/extract_flood.py)

Appendix D: Deployment Configuration (100 lines)
  D.1 Dockerfile
  D.2 Dependencies (requirements.txt, package.json)

Complete source code (30,000+ lines) available at:
https://github.com/CatManJr/Working-with-GeoSpatial-Foundation-Models
```

**Exclude from appendix**: CSS files, utility scripts, test files, build artifacts, data processing logs.

## Citation

```bibtex
@software{fortmyers_flood_2024,
  author = {Your Name},
  title = {Fort Myers Hurricane Helene Flood Risk Analysis},
  year = {2024},
  url = {https://github.com/CatManJr/Working-with-GeoSpatial-Foundation-Models}
}
```

## License

MIT License - See LICENSE file for details.
