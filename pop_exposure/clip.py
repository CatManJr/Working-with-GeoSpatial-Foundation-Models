"""
Clip population raster data to Fort Myers city boundary
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import rasterio
from rasterio.mask import mask
import geopandas as gpd
from paths import CITY_BOUNDARY, POPULATION, DATA_DIR, ensure_dir


def clip_population_to_city():
    """
    Clip the population raster to Fort Myers city boundary
    """
    # Read city boundary shapefile
    print(f"Reading city boundary from: {CITY_BOUNDARY}")
    boundary = gpd.read_file(CITY_BOUNDARY)
    print(f"City boundary CRS: {boundary.crs}")
    
    # Read population raster
    print(f"\nReading population raster from: {POPULATION}")
    with rasterio.open(POPULATION) as src:
        print(f"Population raster CRS: {src.crs}")
        print(f"Population raster shape: {src.shape}")
        print(f"Population raster bounds: {src.bounds}")
        
        # Reproject boundary to match raster CRS if needed
        if boundary.crs != src.crs:
            print(f"\nReprojecting boundary from {boundary.crs} to {src.crs}")
            boundary = boundary.to_crs(src.crs)
        
        # Clip raster to boundary
        print("\nClipping population raster to city boundary...")
        out_image, out_transform = mask(
            src, 
            boundary.geometry, 
            crop=True,
            nodata=src.nodata
        )
        
        # Update metadata
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "compress": "lzw"
        })
        
        # Create output directory
        output_dir = ensure_dir(DATA_DIR / "pop")
        output_path = output_dir / "fort_myers_worldpop.tif"
        
        # Write clipped raster
        print(f"\nWriting clipped population raster to: {output_path}")
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)
        
        print(f"\n✓ Successfully clipped population raster")
        print(f"  Output shape: {out_image.shape}")
        print(f"  Output file: {output_path}")
        
        # Calculate statistics
        valid_data = out_image[out_image != src.nodata]
        if len(valid_data) > 0:
            print(f"\nPopulation Statistics:")
            print(f"  Total population: {valid_data.sum():,.0f}")
            print(f"  Min: {valid_data.min():.2f}")
            print(f"  Max: {valid_data.max():.2f}")
            print(f"  Mean: {valid_data.mean():.2f}")
        
        return output_path


if __name__ == "__main__":
    clip_population_to_city()
