#!/usr/bin/env python3
import os
import sys
import numpy as np
import xarray as xr
import rioxarray as rxr
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from datetime import datetime

# Import paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import S2_MOSAIC, ALIGNED_S2, IBM_DIR

IN_TIF  = S2_MOSAIC
REF_TIF = ALIGNED_S2
OUT_DIR = IBM_DIR

# Optional temporal metadata for the scene (Prithvi demo can parse these)
Year = "2024"
Month = "09"
Day = "25"
dt0 = datetime(int(Year), int(Month), int(Day))
JulianDay = dt0.timetuple().tm_yday

# Expected six S2 L1C bands used by the model: Blue, Green, Red, Narrow NIR (B8A), SWIR (B11), SWIR 2 (B12)
# We will set both friendly names and standard codes for clarity
BAND_DESCRIPTION = ["Blue", "Green", "Red", "Narrow NIR", "SWIR", "SWIR 2"]
BAND_CODES = ["B02",  "B03",  "B04", "B8A", "B11",  "B12"]

def load_da(path: str) -> xr.DataArray:
    da = rxr.open_rasterio(path, masked=True)
    if "band" not in da.dims:
        da = da.expand_dims("band")
    return da.astype("float32")

def write_with_descriptions(src_da: xr.DataArray, out_path: str, ref_path: str | None = None):
    da = src_da.transpose("band", "y", "x")
    assert da.sizes["band"] == 6, f"Expected 6 bands, got {da.sizes['band']}"

    # Build profile from reference, else from data
    if ref_path and os.path.exists(ref_path):
        with rasterio.open(ref_path) as ref:
            profile = ref.profile.copy()
            profile.update(count=6, dtype="uint16", compress="LZW", nodata=0)
            transform = ref.transform
            crs = ref.crs
            height, width = ref.height, ref.width
    else:
        with rasterio.Env():
            transform = da.rio.transform(recalc=True)
            crs = da.rio.crs
            height, width = da.sizes["y"], da.sizes["x"]
            profile = {
                "driver": "GTiff",
                "height": height,
                "width": width,
                "count": 6,
                "dtype": "uint16",
                "crs": crs,
                "transform": transform,
                "compress": "LZW",
                "nodata": 0,
            }

    # Write data and set band descriptions; UINT16 TOA reflectance scaled by 10000, nodata=0
    data = da.values  # (6, H, W) float32 in [0,1] expected
    data = np.where(np.isfinite(data), data, 0.0)
    data = np.clip(np.rint(data * 10000.0), 0, 10000).astype(np.uint16)
    with rasterio.open(out_path, "w", **profile) as dst:
        for i in range(6):
            dst.write(data[i, :, :], i + 1)
            # Set both friendly description and a tag with the code
            try:
                dst.set_band_description(i + 1, BAND_DESCRIPTION[i])
            except Exception:
                pass
            try:
                dst.update_tags(i + 1, S2_BAND=BAND_CODES[i])
            except Exception:
                pass
        # No extra tags per user requirement; match input data semantics

def main():
    # Load input mosaic and ensure CRS/transform preserved from S2
    da = load_da(IN_TIF)
    # If your band order differs, reorder here to [B02,B03,B04,B8A,B11,B12]
    # Example: da = da.isel(band=[idx_B02, idx_B03, idx_B04, idx_B8A, idx_B11, idx_B12])
    # Build final filename: FortMyers_Helene2024_{year}T{julian_day}.tif
    final_path = os.path.join(OUT_DIR, f"FortMyersHelene_{Year}T{JulianDay:03d}.tif")

    # Write directly to the final path
    write_with_descriptions(da, final_path, REF_TIF)

    # Create evenly overlapped tiles identified by row/col indices
    TILE = 512
    MIN_OVERLAP = 128

    def even_positions(dim: int, tile: int, min_overlap: int):
        if dim <= tile:
            return [0], tile
        step_min = tile - min_overlap
        n = int(np.ceil((dim - tile) / step_min)) + 1
        if n <= 1:
            return [0], tile
        positions = [int(round(i * (dim - tile) / (n - 1))) for i in range(n)]
        stride = positions[1] - positions[0] if len(positions) > 1 else tile
        return positions, stride

    tiles_dir = os.path.join(OUT_DIR, f"tiles_{Year}T{JulianDay:03d}")
    os.makedirs(tiles_dir, exist_ok=True)

    import csv
    with rasterio.open(final_path) as src:
        H, W = src.height, src.width
        base_profile = src.profile.copy()
        ys, sy = even_positions(H, TILE, MIN_OVERLAP)
        xs, sx = even_positions(W, TILE, MIN_OVERLAP)

        rows = []
        for ri, y0 in enumerate(ys):
            for ci, x0 in enumerate(xs):
                win_h = min(TILE, H - y0)
                win_w = min(TILE, W - x0)
                window = Window(x0, y0, win_w, win_h)
                prof = base_profile.copy()
                prof.update(
                    height=win_h,
                    width=win_w,
                    transform=window_transform(window, src.transform),
                    compress="LZW",
                    nodata=0,
                    dtype="uint16",
                )
                tile_name = f"FortMyersHelene_{Year}T{JulianDay:03d}_r{ri}_c{ci}.tif"
                tile_path = os.path.join(tiles_dir, tile_name)
                data = src.read(window=window)
                if data.dtype != np.uint16:
                    data = np.where(np.isfinite(data), data, 0.0)
                    data = np.clip(np.rint(data * 10000.0), 0, 10000).astype(np.uint16)
                with rasterio.open(tile_path, "w", **prof) as dst:
                    dst.write(data)
                rows.append({"ri": int(ri), "ci": int(ci), "y": int(y0), "x": int(x0), "h": int(win_h), "w": int(win_w)})

    # Save tile index CSV
    csv_path = os.path.join(tiles_dir, "tiles.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ri", "ci", "y", "x", "h", "w"])
        writer.writeheader()
        writer.writerows(rows)

    print({
        "out_tif": final_path,
        "bands": BAND_CODES,
        "names": BAND_DESCRIPTION,
        "tiles_dir": tiles_dir,
        "tile": TILE,
    })
    
    # rm all ._ files
    for file in os.listdir(OUT_DIR):
        if file.startswith("._"):
            os.remove(os.path.join(OUT_DIR, file))

if __name__ == "__main__":
    main()
