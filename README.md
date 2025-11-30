# Fort Myers Hurricane Helene Flood Analysis

This project analyzes flood extent from Hurricane Helene in Fort Myers, Florida using satellite imagery and machine learning.

## Project Structure

```
Root/
├── paths.py                  # Centralized path configuration
├── pyproject.toml            # Python dependencies
├── README.md                 # This file
├── clean_index.sh            # Call the cleaning script in terminal. Click to run this command
├── hf_token.txt              # HuggingFace token. Create this file using your own token
├── utils                     # System-wide utilities
│   └── clean_._.py/          # Recursively clean the index file on MacOS
│
├── data/                     # All data files (download from release page)
│   ├── raw/                  # Raw satellite data from GEE
│   ├── processed/            # Processed and reconstructed data
│   ├── IBM/                  # Prithvi model inputs/outputs
│   ├── flood/                # Final flood results
│   ├── NHD/                  # National Hydrology Data
│   ├── permanent_water/      # Permanent water data from the NHD
│   ├── Fort_Myers_City_Boundary # Fort Myers City Boundary
│   └── pop/                  # WorldPop 2024 100m population raster
│
├── GEE_script/               # Google Earth Engine scripts
│   └── fetch_data.js
│
├── reconstruct/              # S2 reconstruction workflow
│   ├── make_dataset.py       # Build ML dataset
│   ├── train.py              # Train LightGBM
│   ├── reconstruct.py        # Reconstruct cloudy pixels
│   └── viz.py                # Visualize results
│
├── water_segmentation/       # Flood detection workflow
│   ├── prepare_Prithvi.py    # Prepare tiles for Prithvi
│   └── predict.py            # Run Prithvi inference
│
└── flood_extract/            # Flood extraction
    ├── permanent_water.py    # Extract permanent water
    └── extract_flood.py      # Extract flood areas

```

## Install
```bash
pip install uv
```
uv is a tool for managing virtual environments. It's a wrapper around `venv` and `conda`.

```bash
uv sync # Because I provided the pyproject.toml, you can just run `uv sync`
```

```bash
uv add [library] # Use this to add a library to your environment
```

## Workflow

### 1. Data Download (GEE)
```bash
# Run in Google Earth Engine Code Editor
# GEE_script/fetch_data.js
```

### 2. Extract Permanent Water
```bash
uv run flood_extract/permanent_water.py
```

### 3. S2 Reconstruction (Remove Clouds)
```bash
# Step 1: Build ML dataset
uv run reconstruct/make_dataset.py

# Step 2: Train LightGBM models
uv run reconstruct/train.py

# Step 3: Reconstruct cloudy pixels
uv run reconstruct/reconstruct.py

# Step 4: Visualize results
uv run reconstruct/viz.py
```

### 4. Flood Detection
```bash
# Step 1: Prepare tiles for Prithvi
uv run water_segmentation/prepare_Prithvi.py

# Step 2: Run Prithvi inference (requires HuggingFace token)
uv run water_segmentation/predict.py

# Step 3: Extract flood areas
uv run flood_extract/extract_flood.py
```

## Path Configuration

All scripts now use **centralized path management** via `paths.py`. This means:

### Using Paths in Your Scripts

```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import RAW_S2, PROCESSED_DIR, FLOOD_DIR, get_hf_token

# Now use the paths
print(RAW_S2)  # data/raw/FortMyers_Helene2024_S2.tif
```

### Key Paths Defined

```python
# Raw data
RAW_S2          # data/raw/FortMyers_Helene2024_S2.tif
RAW_S1          # data/raw/FortMyers_Helene2024_S1.tif
RAW_AEF         # data/raw/FortMyers_Helene2024_AEF64.tif
RAW_CLOUD       # data/raw/FortMyers_Helene2024_cloud_mask.tif

# Processed data
S2_MOSAIC       # data/processed/S2_mosaic.tif (reconstructed)
LGBM_DIR        # data/processed/lgbm/ (ML datasets)

# Prithvi results
IBM_TILES_DIR   # data/IBM/tiles_2024T269/
PRITHVI_PREDICTION  # data/IBM/predict/...inundated.tif

# Flood results
FLOOD_DIR       # data/flood/
FLOOD_CLIPPED   # data/flood/...flood_clipped.tif
```

## Requirements

```bash
# I use uv to manage the virtual environment
# Install dependencies
uv sync
```
If you do not want to use uv, you can install dependencies manually as listed in `pyproject.toml`.

### HuggingFace Token
For Prithvi model inference on the cloud, you need a HuggingFace token:

1. Get token from: https://huggingface.co/settings/tokens
2. Save your hf token to `hf_token.txt` in the project root, OR
3. Set environment variable: `export HF_TOKEN='your_token'`

Or you could infer locally on Windows or Linux. Please refer to https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11/tree/main

## Output Files

After running all workflows, you'll have:

- `data/processed/S2_mosaic.tif` - Cloud-free Sentinel-2 mosaic
- `data/IBM/predict/FortMyersHelene_2024T269_inundated.tif` - Water detection
- `data/flood/FortMyersHelene_2024T269_flood_clipped.tif` - Final flood extent (raster)
- `data/flood/FortMyersHelene_2024T269_flood_clipped.shp` - Final flood extent (vector)
