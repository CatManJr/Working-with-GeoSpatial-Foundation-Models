"""
Extract Inundated Areas
Extract flood-only areas by removing permanent water, then clip to Fort Myers boundary
"""

import os
import sys
from pathlib import Path
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.features import rasterize, shapes
import geopandas as gpd
from shapely.geometry import shape
import warnings
warnings.filterwarnings('ignore')

# Import paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import PRITHVI_PREDICTION, PERMANENT_WATER, CITY_BOUNDARY, FLOOD_DIR


class InundatedAreaExtractor:
    """Extract flood-only areas by removing permanent water."""
    
    def __init__(
        self,
        predicted_water: str = None,
        permanent_water: str = None,
        city_boundary: str = None,
        output_dir: str = None,
    ):
        """Initialize extractor"""
        self.predicted_water_path = Path(predicted_water) if predicted_water else PRITHVI_PREDICTION
        self.permanent_water_path = Path(permanent_water) if permanent_water else PERMANENT_WATER
        self.city_boundary_path = Path(city_boundary) if city_boundary else CITY_BOUNDARY
        self.output_dir = Path(output_dir) if output_dir else FLOOD_DIR
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_predicted_water(self):
        """Load predicted water mask from Prithvi model"""
        print("=" * 80)
        print("Loading predicted water mask...")
        print("=" * 80)
        
        with rasterio.open(self.predicted_water_path) as src:
            water_mask = src.read(1)
            profile = src.profile
            transform = src.transform
            crs = src.crs
            
        water_pixels = np.sum(water_mask == 1)
        total_pixels = water_mask.size
        
        print(f"  Input: {self.predicted_water_path}")
        print(f"  Dimensions: {water_mask.shape}")
        print(f"  Water pixels: {water_pixels:,} ({water_pixels/total_pixels*100:.2f}%)")
        print()
        
        return water_mask, profile, transform, crs
    
    def rasterize_permanent_water(self, water_profile, transform, crs):
        """Rasterize permanent water shapefile to match water mask"""
        print("Rasterizing permanent water...")
        
        # Load permanent water shapefile
        perm_water_gdf = gpd.read_file(self.permanent_water_path)
        
        # Reproject to match water mask CRS if needed
        if perm_water_gdf.crs != crs:
            print(f"  Reprojecting from {perm_water_gdf.crs} to {crs}")
            perm_water_gdf = perm_water_gdf.to_crs(crs)
        
        print(f"  Input: {self.permanent_water_path}")
        print(f"  Features: {len(perm_water_gdf)}")
        
        # Rasterize
        shapes_gen = ((geom, 1) for geom in perm_water_gdf.geometry)
        
        perm_water_mask = rasterize(
            shapes_gen,
            out_shape=(water_profile['height'], water_profile['width']),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )
        
        perm_water_pixels = np.sum(perm_water_mask == 1)
        total_pixels = perm_water_mask.size
        
        print(f"  Permanent water pixels: {perm_water_pixels:,} ({perm_water_pixels/total_pixels*100:.2f}%)")
        print()
        
        return perm_water_mask
    
    def extract_flood_only(self, water_mask, perm_water_mask):
        """Extract flood-only areas by removing permanent water"""
        print("=" * 80)
        print("Extracting flood-only areas...")
        print("=" * 80)
        
        # Start with predicted water
        flood_mask = water_mask.copy()
        
        # Remove permanent water (where perm_water_mask == 1)
        perm_water_removed = np.sum((flood_mask == 1) & (perm_water_mask == 1))
        flood_mask[perm_water_mask == 1] = 0
        
        flood_pixels = np.sum(flood_mask == 1)
        total_pixels = flood_mask.size
        
        print(f"  Original water pixels: {np.sum(water_mask == 1):,}")
        print(f"  Removed permanent water: {perm_water_removed:,}")
        print(f"  Final flood pixels: {flood_pixels:,} ({flood_pixels/total_pixels*100:.2f}%)")
        print()
        
        return flood_mask
    
    def clip_to_boundary(self, flood_mask, profile, transform, crs):
        """Clip flood mask to Fort Myers city boundary"""
        print("Clipping to Fort Myers boundary...")
        
        # Load city boundary
        boundary_gdf = gpd.read_file(self.city_boundary_path)
        
        # Reproject if needed
        if boundary_gdf.crs != crs:
            print(f"  Reprojecting boundary from {boundary_gdf.crs} to {crs}")
            boundary_gdf = boundary_gdf.to_crs(crs)
        
        print(f"  Boundary: {self.city_boundary_path}")
        print(f"  Features: {len(boundary_gdf)}")
        
        # Create a temporary raster file to use rasterio.mask
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Write flood mask to temp file
            with rasterio.open(tmp_path, 'w', **profile) as dst:
                dst.write(flood_mask, 1)
            
            # Clip using rasterio.mask
            with rasterio.open(tmp_path) as src:
                clipped_data, clipped_transform = mask(
                    src,
                    boundary_gdf.geometry,
                    crop=True,
                    all_touched=True
                )
                
                clipped_mask = clipped_data[0]
                
                # Update profile
                clipped_profile = profile.copy()
                clipped_profile.update({
                    'height': clipped_mask.shape[0],
                    'width': clipped_mask.shape[1],
                    'transform': clipped_transform
                })
        
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
        clipped_pixels = np.sum(clipped_mask == 1)
        total_pixels = clipped_mask.size
        
        print(f"  Clipped dimensions: {clipped_mask.shape}")
        print(f"  Flood pixels in boundary: {clipped_pixels:,} ({clipped_pixels/total_pixels*100:.2f}%)")
        print()
        
        return clipped_mask, clipped_profile
    
    def save_raster(self, flood_mask, profile, output_name: str):
        """Save flood mask as GeoTIFF"""
        output_path = self.output_dir / output_name
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(flood_mask, 1)
        
        print(f"  Raster saved: {output_path}")
        
        return output_path
    
    def vectorize_flood(self, flood_mask, profile, output_name: str):
        """Vectorize flood mask to shapefile"""
        print("Vectorizing flood areas...")
        
        # Extract shapes
        geoms = []
        values = []
        
        for geom, value in shapes(flood_mask, transform=profile['transform']):
            if value == 1:  # Only flood pixels
                geoms.append(shape(geom))
                values.append(1)
        
        if not geoms:
            print("  No flood polygons found!")
            return None
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            {'flood': values},
            geometry=geoms,
            crs=profile['crs']
        )
        
        # Calculate area in square meters (assuming projected CRS)
        gdf['area_m2'] = gdf.geometry.area
        gdf['area_km2'] = gdf['area_m2'] / 1e6
        
        # Save to shapefile
        output_path = self.output_dir / output_name.replace('.tif', '.shp')
        gdf.to_file(output_path)
        
        total_area = gdf['area_km2'].sum()
        
        print(f"  Polygons: {len(gdf)}")
        print(f"  Total flood area: {total_area:.2f} km²")
        print(f"  Vector saved: {output_path}")
        print()
        
        return output_path
    
    def run(self):
        """Execute complete extraction workflow"""
        print("\n" + "=" * 80)
        print("Fort Myers Flood-Only Area Extraction")
        print("=" * 80 + "\n")
        
        # Load predicted water
        water_mask, profile, transform, crs = self.load_predicted_water()
        
        # Rasterize permanent water
        perm_water_mask = self.rasterize_permanent_water(profile, transform, crs)
        
        # Extract flood-only areas
        flood_mask = self.extract_flood_only(water_mask, perm_water_mask)
        
        # Save full flood mask (before clipping)
        print("=" * 80)
        print("Saving results...")
        print("=" * 80)
        
        full_output = self.save_raster(
            flood_mask,
            profile,
            "FortMyersHelene_2024T269_flood.tif"
        )
        
        # Clip to Fort Myers boundary
        clipped_mask, clipped_profile = self.clip_to_boundary(
            flood_mask,
            profile,
            transform,
            crs
        )
        
        # Save clipped flood mask
        clipped_output = self.save_raster(
            clipped_mask,
            clipped_profile,
            "FortMyersHelene_2024T269_flood_clipped.tif"
        )
        
        # Vectorize clipped flood areas
        vector_output = self.vectorize_flood(
            clipped_mask,
            clipped_profile,
            "FortMyersHelene_2024T269_flood_clipped.tif"
        )
        
        print("=" * 80)
        print("Flood extraction completed!")
        print("=" * 80)
        print(f"  Full flood mask: {full_output}")
        print(f"  Clipped flood mask: {clipped_output}")
        if vector_output:
            print(f"  Flood polygons: {vector_output}")
        print("=" * 80)
        
        return {
            'full_raster': full_output,
            'clipped_raster': clipped_output,
            'vector': vector_output
        }


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract flood-only areas (excluding permanent water)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--predicted_water",
        default=None,
        help="Path to predicted water mask"
    )
    
    parser.add_argument(
        "--permanent_water",
        default=None,
        help="Path to permanent water shapefile"
    )
    
    parser.add_argument(
        "--city_boundary",
        default=None,
        help="Path to Fort Myers boundary shapefile"
    )
    
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Create and run extractor
    extractor = InundatedAreaExtractor(
        predicted_water=args.predicted_water,
        permanent_water=args.permanent_water,
        city_boundary=args.city_boundary,
        output_dir=args.output_dir,
    )
    
    extractor.run()


if __name__ == "__main__":
    main()
