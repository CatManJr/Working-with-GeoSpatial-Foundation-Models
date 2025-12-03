# Working with GeoSpatial Foundation Models
## Fort Myers Flood Influence Analysis & Dashboard for Hurricane Helene

This is a hands-on project for me to practice building a full-circle machine learning system combining `cloud computing`, `ML`, `foundation models`, `spatial analysis`, `full-stack development`, and `CI/CD`. Due to financial concerns, I put the web dashboard (source code in `/app`) on the Render rather than my AWS or Azure containers, and you can find the address below. The step-by-step processing pipeline (see below) is not on the cloud because I can't afford AWS EC2. Overall, this repo contains an ML pipeline from fetching data on GEE, training and predicting, querying Foundation Model inference, to spatial analysis; and a Docker container web app as the results dashboard.

## Overview

This project implements an end-to-end workflow for flood influence assessment:

1. **Cloud/Shadow Reconstruction**: Reconstruct cloud-obscured Sentinel-2 imagery using Sentinel-1 GRD and Satellite Embeddings V1 (AlphaEarth Foundations). The pipeline of Cloud Reconstruction trains a LightGBM regression tree model to predict the Sentinel-2 L1C (Bands 2,3,4,8,11,12) by Sentinel-1 GRD (as short-term reference) and Satellite Embeddings V1 (as long-term reference). The average R-squared is above 0.85, without adjusting hyperparameters.
   
2. **Water Segmentation**: Segment water pixels using IBM/NASA Prithvi-EO-2.0-300M-TL-Sen1Floods11 foundation model. Because the model can not run on macOS, I queried the official demo, which exposed an inference API, to do this task. Permanent water pixels are merged and selected by both NHDArea and NHDWaterbody in Fort Myers. And the flood pixels are defined as: `Segment water pixels - Permanent water pixels`.

3. **Population Exposure**: Calculate the exposed (inundated) population using WorldPop data
4. **Risk Analysis**: Compute spatial accessibility-based risk (influence) scores (G2SFCA method). Totally 4 bandwidths: 250m, 500m, 1000m, 2500m, to simulate the spread of the surface water.

5. **Web Dashboard**: Interactive visualization and UI for the analysis results. The server only has 0.1 CPU and 512 RAM. Be careful when scrolling. And you may need to wait for several minutes to wake up the app. The Web app can fetch the latest ML pipeline output when offline. The fetched data will be stored in [/app/file_database](https://github.com/CatManJr/Working-with-GeoSpatial-Foundation-Models/tree/main/app/file_database), and the FastAPI backend can fetch the data here using Python's sqlite3. The frontend is built on React (JavaScript).

**Cloud dashboard**: https://two024-hurricane-helene-flood-risk.onrender.com (you may need to wait for seconds to awaken the Render server)  
**Source Code Repository (This repo)**: https://github.com/CatManJr/Working-with-GeoSpatial-Foundation-Models

## Project Structure

```bash
Root/
├── paths.py                  # Centralized path configuration
├── pyproject.toml            # Python dependencies
├── Dockerfile                # Container configuration for deployment
├── clean_index.sh            # MacOS index cleanup utility
├── run_accessibility.sh      # Shell script for batch G2SFCA analysis
│
├── utils/                    # Utility scripts
│   └── clean_._.py           # Python script to clean MacOS index files
│
├── app/                      # Full-stack web application
│   ├── backend/              # FastAPI backend
│   │   ├── main.py           # API endpoints
│   │   ├── import_data.py    # ETL script: Imports analysis results to File Geodatabase
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

# Or using pip (suggesting creating a virtue env in the root directory)
pip install uv
uv export --no-hashes --no-dev > requirements.txt # Or manually create requirements.txt based on 
pip install -r requirements.txt
```

## Workflow

### 1. Data Acquisition
Run `GEE_script/fetch_data.js` in Google Earth Engine Code Editor to download:
- [Sentinel-2 L1C (2024-09-21 - 2024-09-29, Hurricane Helene event)](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_HARMONIZED)
- [Sentinel-1 SAR GRD (2024-09-21 - 2024-09-29, Hurricane Helene event)](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD)
- [GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL (Only 2024)](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL?hl=zh-cn#description)
- [Cloud masks (contains shadow)](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED)

### 2. Cloud Reconstruction
```bash
uv run reconstruct/check.py        # Validate input data
uv run reconstruct/make_dataset.py # Mask out grids needing reconstruction and construct csv data and numpy cache
uv run reconstruct/train.py        # Train a LightGBM model for each band
uv run reconstruct/reconstruct.py  # Reconstruct cloudy pixels
uv run reconstruct/viz.py          # Visualization of the construction input and output
```

**Output**: `data/processed/S2_mosaic.tif` (cloud-free composite)

### 3. Permanent Water Extraction
```bash
uv run flood_extract/permanent_water.py # arrange data from NHD
```

Extracts permanent water bodies from NHD (National Hydrology Dataset) to isolate flood-only areas.

### 4. Flood Detection
```bash
uv run water_segmentation/prepare_Prithvi.py  # Prepare 512x512 tiles considering the balance of precision and speed
uv run water_segmentation/predict.py          # Query the model from hugginface space cuz terratorch can not ran on macO
uv run flood_extract/extract_flood.py         # Extract flood pixels by exclude permanent_water from the prediction
```

**Output**: 
- `data/flood/FortMyersHelene_2024T269_flood_clipped.tif` (raster)
- `data/flood/FortMyersHelene_2024T269_flood_clipped.shp` (vector)

### 5. Population Exposure Analysis
```bash
uv run pop_exposure/clip.py        # Clip WorldPop to study area
uv run pop_exposure/overlay.py     # Calculate exposed population and flood extent
./run_accessibility.sh  # Automatically run G2SFCA influence analysis for 4 bandwidths by 'uv run accessibility.py -bandwidth "bandwidth"'
```

Generates influence layers at multiple bandwidths (250m, 500m, 1000m, 2500m).

### 6. Data Migration (Crucial)
Before running the web app, you must import the analysis results into the application's File Geodatabase. So we could run the web app with only 0.1 COU and 512MB RAM with an offline database.

```bash
# Clean MacOS hidden files (Optional, for Mac users)
./clean_index.sh

# Import data from data/ to app/file_database/
uv run app/backend/import_data.py
```

### 7. Web Dashboard

**Development**:
```bash
cd app
# Ensure data is imported first (see Step 6)
./set_up.sh # Recommend running this first if you are working on macOS with an ExFAT disk
./run_dev.sh    # Start backend (FastAPI) and frontend (React) in dev mode
```

**Production**:

```bash
./run_prod.sh   # Build frontend and serve with backend. Remember that you need to first clean the app/frontend/build folder to run run_dev.sh again.
```

**Deployment**: The application is containerized with Docker and deployed on Render.

## Key Technologies

**Languages**: Python, JavaScript, HTML, CSS, Shell
**Geospatial**: rasterio, geopandas, shapely
**Machine Learning**: LightGBM, HuggingFace Transformers, IBM/NASA Prithvi-EO
**Web Stack**: FastAPI, React, Leaflet, Gradio
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
1. Save to `hf_token.txt` in the project root, or
2. Add a `.env` file and write HF_TOKEN='your_token.'`

Alternatively, run inference locally (Windows/Linux) following: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11

## Key Output Files

**Cloud Reconstruction**:
- `data/processed/S2_mosaic.tif` - Cloud-free Sentinel-2 composite

**Flood Detection**:

- `data/IBM/predict/FortMyersHelene_2024T269_inundated.tif` - Water segmentation
- `data/flood/FortMyersHelene_2024T269_flood_clipped.{tif,shp}` - Extracted flood extent

**Population Exposure**:

- `data/pop_exposure/flood_risk_g2sfca_raster_{bandwidth}m.tif` - Influence layers
- `data/pop_exposure/flood_risk_g2sfca_raster_{bandwidth}m_summary.csv` - Statistics

**Web Application Database**:

Arranged like an ArcGIS File Geodatabase （directly fetched and synchronized with /data）

- `app/file_database/rasters/` - All raster layers
- `app/file_database/vectors/` - Boundaries and geometries
- `app/file_database/tables/` - Statistical summaries

## Citation

```bibtex
@software{fortmyers_flood_2024,
  author = {Your Name},
  title = {Fort Myers Hurricane Helene Flood Influence Analysis},
  year = {2025},
  url = {https://github.com/CatManJr/Working-with-GeoSpatial-Foundation-Models}
}
```

## License

MIT License - See LICENSE file for details.
