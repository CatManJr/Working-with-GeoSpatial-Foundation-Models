import os
import sys
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# Import paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import RAW_S2, RAW_S1, RAW_AEF, RAW_CLOUD, S2_RECONSTRUCTED, S2_MOSAIC, DATA_DIR

# Figure: Show raw inputs and reconstructed outputs (2x3 subplots)
# - All labels in English, no non-ASCII characters

def read_as_rgb_quicklook(path, band_indices=None, percentile=(2, 98)):
    """Read a raster and return an RGB image (H, W, 3) with percentile stretching.
    - If band_indices is None:
      - If 3+ bands: use [1,2,3] (1-based) as a simple RGB quicklook
      - If 2 bands: use [1,2,1]
      - If 1 band: repeat [1,1,1]
    - Applies per-channel percentile stretching and clips to [0,1].
    """
    with rasterio.open(path) as ds:
        num_bands = ds.count
        if band_indices is None:
            if num_bands >= 3:
                band_indices = [1, 2, 3]
            elif num_bands == 2:
                band_indices = [1, 2, 1]
            else:
                band_indices = [1, 1, 1]

        rgb = []
        for idx in band_indices:
            idx = max(1, min(idx, num_bands))
            band = ds.read(idx).astype(np.float32)
            nodata = ds.nodata
            if nodata is not None:
                band = np.where(band == nodata, np.nan, band)
            # Robust percentile stretch
            valid = band[np.isfinite(band)]
            if valid.size == 0:
                lo, hi = 0.0, 1.0
            else:
                lo, hi = np.percentile(valid, percentile)
                if hi <= lo:
                    hi = lo + 1.0
            band = (band - lo) / (hi - lo)
            band = np.clip(band, 0, 1)
            # Replace NaNs (e.g., nodata) with zeros for display
            band = np.nan_to_num(band, nan=0.0)
            rgb.append(band)

        rgb = np.stack(rgb, axis=-1)
        return rgb


def read_mask(path):
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        nodata = ds.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, 0, arr)
        # Ensure binary/0-1 range for display
        unique_vals = np.unique(arr)
        if unique_vals.max() > 1:
            arr = (arr > 0).astype(np.uint8)
        return arr


fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

# 1) Raw S2 (cloudy)
try:
    img_raw_s2 = read_as_rgb_quicklook(RAW_S2, band_indices=None)
    axes[0].imshow(img_raw_s2)
    axes[0].set_title("Sentinel-2 L1C (cloudy)", fontsize=16)
except Exception as e:
    axes[0].text(0.5, 0.5, f"Failed to load S2: {e}", ha="center", va="center")
axes[0].axis("off")

# 2) Raw S1 (VV, VH composite)
try:
    # VV -> R, VH -> G, R again as B for visibility
    img_raw_s1 = read_as_rgb_quicklook(RAW_S1, band_indices=[1, 2, 1])
    axes[1].imshow(img_raw_s1)
    axes[1].set_title("Sentinel-1 GRD (SAR VV/VH)", fontsize=16)
except Exception as e:
    axes[1].text(0.5, 0.5, f"Failed to load S1: {e}", ha="center", va="center")
axes[1].axis("off")

# 3) Raw Embeddings (AEF64) pseudo-RGB
try:
    img_emb = read_as_rgb_quicklook(RAW_AEF, band_indices=[1, 2, 3])
    axes[2].imshow(img_emb)
    axes[2].set_title("Satellite Embeddings (AEF64) first 3 bands", fontsize=16)
except Exception as e:
    axes[2].text(0.5, 0.5, f"Failed to load AEF64: {e}", ha="center", va="center")
axes[2].axis("off")

# 4) Cloud/Shadow Mask
try:
    mask = read_mask(RAW_CLOUD)
    axes[3].imshow(mask, cmap="magma", vmin=0, vmax=1)
    axes[3].set_title("Cloud/Shadow Mask", fontsize=16)
except Exception as e:
    axes[3].text(0.5, 0.5, f"Failed to load mask: {e}", ha="center", va="center")
axes[3].axis("off")

# 5) Reconstructed S2 (predictions over cloudy pixels)
try:
    img_recon = read_as_rgb_quicklook(S2_RECONSTRUCTED, band_indices=None)
    axes[4].imshow(img_recon)
    axes[4].set_title("Reconstructed S2 (predicted over clouds)", fontsize=16)
except Exception as e:
    axes[4].text(0.5, 0.5, f"Failed to load reconstructed S2: {e}", ha="center", va="center")
axes[4].axis("off")

# 6) Cloud-free S2 Mosaic (merge)
try:
    img_mosaic = read_as_rgb_quicklook(S2_MOSAIC, band_indices=None)
    axes[5].imshow(img_mosaic)
    axes[5].set_title("Reconstructed S2", fontsize=16)
except Exception as e:
    axes[5].text(0.5, 0.5, f"Failed to load mosaic: {e}", ha="center", va="center")
axes[5].axis("off")

plt.tight_layout()
out_path = DATA_DIR / "reconstructed.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Figure saved to: {out_path}")
plt.show()