#!/usr/bin/env python3
import os, json
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
from rasterio.enums import Resampling

# Import paths
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import *

# ========= Config =========
OUT_DIR = LGBM_DIR
FILL = -9999.0

TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1
SEED = 42

# ========= IO helpers =========
def open_da(path):
    da = rxr.open_rasterio(path, masked=True, chunks=None)
    if "band" not in da.dims:
        da = da.expand_dims("band")
    return da

def write_gtiff(da: xr.DataArray, path: str, compress="LZW", nodata=None, dtype=None, ref: xr.DataArray|None=None):
    if dtype is not None:
        da = da.astype(dtype)
    if nodata is not None:
        try:
            da = da.fillna(nodata)
        except Exception:
            pass
        da = da.rio.write_nodata(nodata, inplace=False, encoded=True)
    if getattr(da.rio, "crs", None) is None:
        if ref is None or getattr(ref.rio, "crs", None) is None:
            raise ValueError("Missing CRS; provide ref")
        da = da.rio.write_crs(ref.rio.crs, inplace=False)
    try:
        da.rio.write_transform(ref.rio.transform(recalc=True), inplace=True)
    except Exception:
        pass
    da.rio.to_raster(path, compress=compress)

# ========= Core logic =========
def reproject_match(src: xr.DataArray, ref: xr.DataArray, resampling: Resampling) -> xr.DataArray:
    return src.rio.reproject_match(ref, resampling=resampling)

def load_or_align():
    if all(os.path.exists(p) for p in [S2_ALIGNED,S1_ALIGNED,AEF_ALIGNED,CLD_ALIGNED]):
        s2 = open_da(S2_ALIGNED); s1 = open_da(S1_ALIGNED)
        aef = open_da(AEF_ALIGNED); cloud = open_da(CLD_ALIGNED)
    else:
        s2 = open_da(S2_RAW).astype("float32")
        s1 = reproject_match(open_da(S1_RAW), s2, Resampling.bilinear).astype("float32")
        aef = reproject_match(open_da(AEF_RAW), s2, Resampling.bilinear).astype("float32")
        cloud = reproject_match(open_da(CLOUD_RAW), s2, Resampling.nearest).astype("uint8")
        write_gtiff(s2, S2_ALIGNED, ref=s2)
        write_gtiff(s1, S1_ALIGNED, ref=s2)
        write_gtiff(aef, AEF_ALIGNED, ref=s2)
        write_gtiff(cloud, CLD_ALIGNED, nodata=255, dtype="uint8", ref=s2)
    return s2, s1, aef, cloud

def build_valid_maps(s2: xr.DataArray, s1: xr.DataArray, aef: xr.DataArray, cloud: xr.DataArray):
    v_s2  = np.isfinite(s2).all("band").values
    v_s1  = np.isfinite(s1).all("band").values
    v_aef = np.isfinite(aef).all("band").values
    v_cld = np.isfinite(cloud.squeeze("band", drop=True)).values if "band" in cloud.dims else np.isfinite(cloud).values
    code = (v_s2.astype(np.uint8) | (v_s1.astype(np.uint8)<<1) | (v_aef.astype(np.uint8)<<2) | (v_cld.astype(np.uint8)<<3))
    any_nodata   = (code != 0b1111)
    all_nodata   = (code == 0)
    mixed_nodata = any_nodata & (~all_nodata)
    return v_s2, v_s1, v_aef, v_cld, code.astype(np.uint8), any_nodata, all_nodata, mixed_nodata

def build_masks(s2: xr.DataArray, cloud: xr.DataArray, s1: xr.DataArray, aef: xr.DataArray):
    cloud_bool = cloud.squeeze("band", drop=True).astype(bool)
    s2_valid = np.isfinite(s2).all("band")
    target_valid = (~cloud_bool) & s2_valid
    feat_valid = np.isfinite(s1).all("band") & np.isfinite(aef).all("band")
    return target_valid, feat_valid

def split_random_pixels(valid_mask: np.ndarray, train_ratio: float, val_ratio: float, test_ratio: float, seed: int = 42):
    H, W = valid_mask.shape
    ys, xs = np.where(valid_mask)
    n = ys.size
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_train = int(round(train_ratio * n))
    n_val   = int(round(val_ratio * n))
    n_test  = max(n - n_train - n_val, 0)
    train_mask = np.zeros((H, W), dtype=bool); train_mask[ys[order[:n_train]], xs[order[:n_train]]] = True
    val_mask   = np.zeros((H, W), dtype=bool); val_mask[ys[order[n_train:n_train+n_val]], xs[order[n_train:n_train+n_val]]] = True
    test_mask  = np.zeros((H, W), dtype=bool); test_mask[ys[order[n_train+n_val:]], xs[order[n_train+n_val:]]] = True
    return train_mask, val_mask, test_mask

def fill_nodata(arr: np.ndarray, fill=FILL):
    return np.where(np.isfinite(arr), arr, fill).astype(np.float32)

def save_cover_masks(train_cov, val_cov, test_cov, ref):
    write_gtiff(xr.DataArray(train_cov.astype("uint8"), dims=("y","x")).expand_dims("band"), TRAIN_COVER_MASK, nodata=255, dtype="uint8", ref=ref)
    write_gtiff(xr.DataArray(val_cov.astype("uint8"),   dims=("y","x")).expand_dims("band"), VAL_COVER_MASK,   nodata=255, dtype="uint8", ref=ref)
    write_gtiff(xr.DataArray(test_cov.astype("uint8"),  dims=("y","x")).expand_dims("band"), TEST_COVER_MASK,  nodata=255, dtype="uint8", ref=ref)

def build_and_save_diagnostics(s2, s1, aef, cloud, validity_code, mixed_nodata, class_map):
    write_gtiff(xr.DataArray(validity_code, dims=("y","x")).expand_dims("band"), VALIDITY_CODE_TIF, nodata=255, dtype="uint8", ref=s2)
    write_gtiff(xr.DataArray(mixed_nodata.astype("uint8"), dims=("y","x")).expand_dims("band"), MIXED_NODATA_TIF, nodata=255, dtype="uint8", ref=s2)
    write_gtiff(xr.DataArray(class_map, dims=("y","x")).expand_dims("band"), CLASS_MAP_TIF, nodata=255, dtype="uint8", ref=s2)

    # Breakdown by validity code and class
    rows = []
    for code_val in range(16):
        mask_code = (validity_code == code_val) & (validity_code != 0b1111)
        if not mask_code.any(): continue
        for cls, name in [(1,"train"),(2,"val"),(3,"test"),(4,"infer"),(0,"none")]:
            cnt = int((mask_code & (class_map==cls)).sum())
            if cnt>0:
                rows.append({"code": code_val, "class": name, "count": cnt,
                             "S2_valid": (code_val & 1)>0, "S1_valid": (code_val & 2)>0,
                             "AEF_valid": (code_val & 4)>0, "CLOUD_valid": (code_val & 8)>0})
    pd.DataFrame(rows).to_csv(NODATA_BREAKDOWN_CSV, index=False)

    handled_counts = {
        "any_nodata_total": int((validity_code != 0b1111).sum()),
        "all_nodata_total": int((validity_code == 0).sum()),
        "mixed_nodata_total": int(mixed_nodata.sum()),
        "train_any_nodata": int(((validity_code != 0b1111) & (class_map==1)).sum()),
        "val_any_nodata":   int(((validity_code != 0b1111) & (class_map==2)).sum()),
        "test_any_nodata":  int(((validity_code != 0b1111) & (class_map==3)).sum()),
        "infer_any_nodata": int(((validity_code != 0b1111) & (class_map==4)).sum()),
        "none_any_nodata":  int(((validity_code != 0b1111) & (class_map==0)).sum()),
    }
    pd.Series(handled_counts).to_csv(NODATA_SUMMARY_CSV)

def save_split(name, ys, xs, feat_np, targ_np):
    X = feat_np[ys, xs, :]
    y = targ_np[ys, xs, :]
    Xf, yf = fill_nodata(X), fill_nodata(y)

    np.savez_compressed(f"{OUT_DIR}/{name}.npz",
                        X=Xf, y=yf, row=ys.astype(np.int32), col=xs.astype(np.int32))
    cols_feat = [f"f{str(i).zfill(2)}" for i in range(Xf.shape[1])]
    cols_targ = [f"t{str(i)}" for i in range(yf.shape[1])]
    df = pd.DataFrame({"row": ys.astype(np.int32), "col": xs.astype(np.int32),
                       **{c: Xf[:, i] for i,c in enumerate(cols_feat)},
                       **{c: yf[:, j] for j,c in enumerate(cols_targ)}})
    try:
        df.to_parquet(f"{OUT_DIR}/{name}.parquet", index=False)
        table_path = f"{OUT_DIR}/{name}.parquet"
    except Exception:
        df.to_csv(f"{OUT_DIR}/{name}.csv.gz", index=False, compression="gzip")
        table_path = f"{OUT_DIR}/{name}.csv.gz"
    return df.shape[0], table_path

def save_inference(infer_mask, feat_np):
    ysi, xsi = np.where(infer_mask)
    X = feat_np[ysi, xsi, :]
    Xf = fill_nodata(X)
    np.savez_compressed(f"{OUT_DIR}/infer.npz", X=Xf, row=ysi.astype(np.int32), col=xsi.astype(np.int32))
    df = pd.DataFrame({"row": ysi.astype(np.int32), "col": xsi.astype(np.int32),
                       **{f"f{str(i).zfill(2)}": Xf[:, i] for i in range(Xf.shape[1])}})
    try:
        df.to_parquet(f"{OUT_DIR}/infer.parquet", index=False)
        table = f"{OUT_DIR}/infer.parquet"
    except Exception:
        df.to_csv(f"{OUT_DIR}/infer.csv.gz", index=False, compression="gzip")
        table = f"{OUT_DIR}/infer.csv.gz"
    return df.shape[0], table

def write_meta(H, W, n_train, n_val, n_test, tables, covers, infer_info, s2):
    try:
        transform = tuple(s2.rio.transform(recalc=True))
    except Exception:
        transform = None
    crs = str(s2.rio.crs) if hasattr(s2.rio, "crs") else None
    meta = {
        "shape_yx": [int(H), int(W)],
        "features": 66,
        "targets": 6,
        "split_method": "random_pixel",
        "pixel_counts": {"train": int(n_train), "val": int(n_val), "test": int(n_test)},
        "split": {"train_ratio": TRAIN_RATIO, "val_ratio": VAL_RATIO, "test_ratio": TEST_RATIO, "method": "random_pixel"},
        "sampling_ratio": {"train": 1.0, "val": 1.0, "test": 1.0},
        "nodata_fill_value": FILL,
        "geo": {"crs": crs, "transform": transform},
        "tables": tables,
        "covers": covers,
        "diagnostics": {
            "validity_code_tif": VALIDITY_CODE_TIF,
            "mixed_nodata_mask_tif": MIXED_NODATA_TIF,
            "nodata_handling_summary_csv": NODATA_SUMMARY_CSV,
            "nodata_handling_breakdown_csv": NODATA_BREAKDOWN_CSV,
            "class_map_tif": CLASS_MAP_TIF,
        },
        "inference": infer_info,
    }
    pd.Series(meta).to_json(META_JSON, indent=2)

# ========= Orchestration =========
def main():
    # 1) Load or align
    s2, s1, aef, cloud = load_or_align()

    # 2) Validity and diagnostics base
    v_s2, v_s1, v_aef, v_cld, validity_code, any_nodata, all_nodata, mixed_nodata = build_valid_maps(s2, s1, aef, cloud)

    # 3) Split masks & coverage
    target_valid, feature_valid = build_masks(s2, cloud, s1, aef)
    valid_both = (target_valid & feature_valid).values.astype(bool)
    H = target_valid.sizes["y"]; W = target_valid.sizes["x"]
    train_cov, val_cov, test_cov = split_random_pixels(valid_both, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED)
    save_cover_masks(train_cov, val_cov, test_cov, s2)

    # 4) Feature/target arrays
    features = xr.concat([s1, aef], dim="band").astype("float32")
    targets  = s2.astype("float32")
    feat_np = features.transpose("y","x","band").values
    targ_np = targets.transpose("y","x","band").values

    # 5) Save splits
    rng = np.random.default_rng(SEED)
    ys, xs = np.where(train_cov); n_train, train_tbl = save_split("train", ys, xs, feat_np, targ_np)
    ys, xs = np.where(val_cov);   n_val,   val_tbl   = save_split("val", ys, xs, feat_np, targ_np)
    ys, xs = np.where(test_cov);  n_test,  test_tbl  = save_split("test", ys, xs, feat_np, targ_np)

    # 6) Inference set (cloud==1 & features finite)
    cloud_bool = cloud.squeeze("band", drop=True).values.astype(bool)
    feat_finite = np.isfinite(features).all("band").values
    infer_mask = cloud_bool & feat_finite
    infer_pixels, infer_table = save_inference(infer_mask, feat_np)

    # 7) Class map & diagnostics
    class_map = np.zeros((H, W), dtype=np.uint8)
    class_map[train_cov] = 1; class_map[val_cov] = 2; class_map[test_cov] = 3; class_map[infer_mask] = 4
    build_and_save_diagnostics(s2, s1, aef, cloud, validity_code, mixed_nodata, class_map)

    # 8) Meta
    write_meta(
        H, W,
        n_train, n_val, n_test,
        tables={"train": train_tbl, "val": val_tbl, "test": test_tbl},
        covers={"train": TRAIN_COVER_MASK, "val": VAL_COVER_MASK, "test": TEST_COVER_MASK},
        infer_info={"pixels": int(infer_pixels), "table": infer_table, "npz": f"{OUT_DIR}/infer.npz"},
        s2=s2
    )
    
    print("Dataset preparation complete.")
    print(f"Train: {n_train} pixels")
    print(f"Val:   {n_val} pixels")
    print(f"Test:  {n_test} pixels")
    print(f"Inference: {infer_pixels} pixels")
    print("Go to the 'data/processed/lgbm' directory for saved datasets and metadata.")

if __name__ == "__main__":
    main()