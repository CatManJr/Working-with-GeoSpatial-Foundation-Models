"""
Path Configuration for Fort Myers Flood Analysis
Provides centralized path management using relative paths from project root
"""

from pathlib import Path
import os

# Project root directory (where this file is located)
PROJECT_ROOT = Path(__file__).parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"

# Raw data
RAW_DIR = DATA_DIR / "raw"
RAW_S2 = RAW_DIR / "FortMyers_Helene2024_S2.tif"
RAW_S1 = RAW_DIR / "FortMyers_Helene2024_S1.tif"
RAW_AEF = RAW_DIR / "FortMyers_Helene2024_AEF64.tif"
RAW_CLOUD = RAW_DIR / "FortMyers_Helene2024_cloud_mask.tif"

# Processed data
PROCESSED_DIR = DATA_DIR / "processed"
ALIGNED_S2 = PROCESSED_DIR / "aligned_S2.tif"
ALIGNED_S1 = PROCESSED_DIR / "aligned_S1.tif"
ALIGNED_AEF = PROCESSED_DIR / "aligned_AEF64.tif"
ALIGNED_CLOUD = PROCESSED_DIR / "aligned_cloud_mask.tif"

S2_RECONSTRUCTED = PROCESSED_DIR / "S2_reconstructed_infer.tif"
S2_MOSAIC = PROCESSED_DIR / "S2_mosaic.tif"

# LGBM dataset and outputs
LGBM_DIR = PROCESSED_DIR / "lgbm"
LGBM_META = LGBM_DIR / "lgbm_meta.json"

# For make_dataset.py - add all required paths
S2_ALIGNED = ALIGNED_S2
S1_ALIGNED = ALIGNED_S1
AEF_ALIGNED = ALIGNED_AEF
CLD_ALIGNED = ALIGNED_CLOUD

S2_RAW = RAW_S2
S1_RAW = RAW_S1
AEF_RAW = RAW_AEF
CLOUD_RAW = RAW_CLOUD

TRAIN_COVER_MASK = PROCESSED_DIR / "train_cover_mask.tif"
VAL_COVER_MASK = PROCESSED_DIR / "val_cover_mask.tif"
TEST_COVER_MASK = PROCESSED_DIR / "test_cover_mask.tif"

VALIDITY_CODE_TIF = PROCESSED_DIR / "validity_code.tif"
MIXED_NODATA_TIF = PROCESSED_DIR / "mixed_nodata_mask.tif"
CLASS_MAP_TIF = PROCESSED_DIR / "class_map.tif"

NODATA_BREAKDOWN_CSV = PROCESSED_DIR / "nodata_handling_breakdown.csv"
NODATA_SUMMARY_CSV = PROCESSED_DIR / "nodata_handling_summary.csv"
META_JSON = LGBM_META

# IBM Prithvi
IBM_DIR = DATA_DIR / "IBM"
IBM_TILES_DIR = IBM_DIR / "tiles_2024T269"
IBM_PREDICT_DIR = IBM_DIR / "predict"
IBM_FULL_IMAGE = IBM_DIR / "FortMyersHelene_2024T269.tif"
PRITHVI_PREDICTION = IBM_PREDICT_DIR / "FortMyersHelene_2024T269_inundated.tif"

# Auxiliary data
CITY_BOUNDARY = DATA_DIR / "Fort_Myers_City_Boundary" / "City_Boundary.shp"
PERMANENT_WATER = DATA_DIR / "permanent_water" / "permanent_water.shp"
POPULATION = DATA_DIR / "pop" / "usa_pop_2024_CN_100m_R2025A_v1.tif"
NHD_DIR = DATA_DIR / "NHD"

# Flood results
FLOOD_DIR = DATA_DIR / "flood"
FLOOD_FULL = FLOOD_DIR / "FortMyersHelene_2024T269_flood.tif"
FLOOD_CLIPPED = FLOOD_DIR / "FortMyersHelene_2024T269_flood_clipped.tif"
FLOOD_VECTOR = FLOOD_DIR / "FortMyersHelene_2024T269_flood_clipped.shp"

# HuggingFace token
HF_TOKEN_FILE = PROJECT_ROOT / "hf_token.txt"


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_hf_token() -> str:
    """Read HuggingFace token from file"""
    if HF_TOKEN_FILE.exists():
        return HF_TOKEN_FILE.read_text().strip()
    return os.getenv("HF_TOKEN", "")
