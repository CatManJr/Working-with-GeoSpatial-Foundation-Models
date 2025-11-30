import os
import sys
import pandas as pd
import lightgbm as lgb
import json
import numpy as np
import xarray as xr
import rioxarray as rxr
from rasterio.transform import Affine

# Import paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import LGBM_DIR, ALIGNED_S2, ALIGNED_CLOUD, S2_RECONSTRUCTED, S2_MOSAIC, LGBM_META

def mosaic_tif(pred_tif: str, s2_tif: str, out_tif: str, cloud_tif: str):
    """
    Create a 6-band mosaic: use original S2 where cloud==0, and predicted S2 where cloud==1.
    """
    s2 = rxr.open_rasterio(s2_tif).astype("float32")
    pred = rxr.open_rasterio(pred_tif).astype("float32")
    cloud = rxr.open_rasterio(cloud_tif, masked=True)
    if "band" in cloud.dims:
        cloud = cloud.squeeze("band", drop=True)
    cloud_bool = (cloud.astype("uint8") == 1)

    assert s2.sizes["y"] == pred.sizes["y"] and s2.sizes["x"] == pred.sizes["x"], "S2/pred shape mismatch"
    mosaic = xr.where(cloud_bool, pred, s2)
    mosaic = mosaic.transpose("band", "y", "x")
    mosaic = mosaic.rio.write_crs(s2.rio.crs, inplace=False)
    try:
        mosaic.rio.write_transform(s2.rio.transform(recalc=True), inplace=True)
    except Exception:
        pass
    mosaic.rio.to_raster(out_tif, compress="LZW")

def reconstruct_geotiff_from_table(pred_table_path: str, out_tif_path: str, meta_json: str):
    meta = pd.read_json(meta_json, typ="series")
    H, W = map(int, meta["shape_yx"])
    crs = meta["geo"].get("crs")
    tfm = meta["geo"].get("transform")

    df = pd.read_parquet(pred_table_path) if pred_table_path.endswith(".parquet") else pd.read_csv(pred_table_path)
    targ_cols = sorted([c for c in df.columns if c.startswith("t")], key=lambda c: int("".join(ch for ch in c if ch.isdigit()) or "0"))
    arr = np.full((len(targ_cols), H, W), np.nan, dtype=np.float32)
    ys = df["row"].to_numpy(np.int64); xs = df["col"].to_numpy(np.int64); vals = df[targ_cols].to_numpy(np.float32)
    for b in range(len(targ_cols)): arr[b, ys, xs] = vals[:, b]

    da = xr.DataArray(arr, dims=("band","y","x"))
    if crs is not None:
        da = da.rio.write_crs(crs, inplace=False)

    if isinstance(tfm, (list, tuple)):
        aff = Affine(tfm[0], tfm[1], tfm[2], tfm[3], tfm[4], tfm[5]) if len(tfm) >= 6 else Affine.identity()
        da.rio.write_transform(aff, inplace=True)

    da.rio.to_raster(out_tif_path, compress="LZW")

# Load data
train = pd.read_parquet(f"{LGBM_DIR}/train.parquet")
val   = pd.read_parquet(f"{LGBM_DIR}/val.parquet")
test  = pd.read_parquet(f"{LGBM_DIR}/test.parquet")
infer = pd.read_parquet(f"{LGBM_DIR}/infer.parquet")

models = [lgb.Booster(model_file=f"{LGBM_DIR}/model_t{i}.txt") for i in range(6)]
feat_cols = [c for c in train.columns if c.startswith("f")]

# Predict and save
preds = [m.predict(infer[feat_cols], num_iteration=m.best_iteration) for m in models]
pred_df = pd.DataFrame({"row": infer["row"], "col": infer["col"], **{f"t{i}": preds[i] for i in range(6)}})
pred_df.to_parquet(f"{LGBM_DIR}/preds_infer.parquet", index=False)

# Reconstruct geotiff from prediction table
reconstruct_geotiff_from_table(
    f"{LGBM_DIR}/preds_infer.parquet",
    str(S2_RECONSTRUCTED),
    str(LGBM_META)
)

# Mosaic the reconstructed tif and the original tif
mosaic_tif(
    str(S2_RECONSTRUCTED),
    str(ALIGNED_S2),
    str(S2_MOSAIC),
    str(ALIGNED_CLOUD)
)

print(f"Reconstruction complete!")
print(f"  Reconstructed S2: {S2_RECONSTRUCTED}")
print(f"  Final mosaic: {S2_MOSAIC}")