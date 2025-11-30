import os
import sys
import geopandas as gpd
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import CITY_BOUNDARY, NHD_DIR, DATA_DIR

def extract_permanent_water():
    """
    Merges NHD water data and extracts permanent water bodies within the Fort Myers city boundary.
    Uses 'intersects' to allow for boundary crossing, rather than strict containment.
    """
    
    # Define file paths using centralized paths
    city_boundary_path = CITY_BOUNDARY
    
    nhd_files = [
        NHD_DIR / "NHD_H_03090205_HU8_Shape/Shape/NHDArea.shp",
        NHD_DIR / "NHD_H_03090205_HU8_Shape/Shape/NHDWaterbody.shp",
        NHD_DIR / "NHD_H_03090204_HU8_Shape/Shape/NHDWaterbody.shp"
    ]
    
    print("Loading city boundary...")
    city_boundary = gpd.read_file(city_boundary_path)
    print(f"City boundary CRS: {city_boundary.crs}")
    print(f"City boundary bounds: {city_boundary.total_bounds}")
    
    # Store all water features
    all_water_features = []
    
    # Read and merge all NHD water files
    for nhd_file in nhd_files:
        if Path(nhd_file).exists():
            print(f"\nProcessing: {nhd_file}")
            try:
                gdf = gpd.read_file(nhd_file)
                print(f"  Original feature count: {len(gdf)}")
                print(f"  CRS: {gdf.crs}")
                
                # Ensure CRS consistency
                if gdf.crs != city_boundary.crs:
                    print(f"  Transforming CRS to: {city_boundary.crs}")
                    gdf = gdf.to_crs(city_boundary.crs)
                
                # Keep only Polygon type geometries
                gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
                print(f"  Polygon feature count: {len(gdf)}")
                
                # Use intersects to filter features that intersect with the city boundary
                intersects_mask = gdf.geometry.intersects(city_boundary.unary_union)
                gdf_filtered = gdf[intersects_mask].copy()
                print(f"  Features intersecting with boundary: {len(gdf_filtered)}")
                
                if len(gdf_filtered) > 0:
                    # Add source information
                    gdf_filtered['source_file'] = Path(nhd_file).name
                    all_water_features.append(gdf_filtered)
                    
            except Exception as e:
                print(f"  Error: {e}")
        else:
            print(f"\nFile not found: {nhd_file}")
    
    # Merge all water features
    if all_water_features:
        print(f"\nMerging all water features...")
        merged_water = pd.concat(all_water_features, ignore_index=True)
        print(f"Total features after merging: {len(merged_water)}")
        
        # Check field information
        print(f"\nAvailable fields: {merged_water.columns.tolist()}")
        
        # Filter for permanent water (if FType or FCode fields exist)
        if 'FType' in merged_water.columns:
            print(f"\nFType distribution:")
            print(merged_water['FType'].value_counts())
        
        if 'FCode' in merged_water.columns:
            print(f"\nFCode distribution (top 10):")
            print(merged_water['FCode'].value_counts().head(10))
        
        # Create output directory
        output_dir = DATA_DIR / "permanent_water"
        output_dir.mkdir(exist_ok=True)
        
        # Save results
        output_path = output_dir / "permanent_water.shp"
        print(f"\nSaving to: {output_path}")
        merged_water.to_file(output_path)
        print(f"Successfully saved! Total {len(merged_water)} water features.")
        
        # Save in GeoJSON format (more universal)
        geojson_path = output_dir / "permanent_water.geojson"
        merged_water.to_file(geojson_path, driver='GeoJSON')
        print(f"Also saved in GeoJSON format: {geojson_path}")
        
        # Statistics
        total_area = merged_water.geometry.area.sum()
        print(f"\nTotal area: {total_area:.2f} square units")
        print(f"Average area: {merged_water.geometry.area.mean():.2f} square units")
        
        return merged_water
    else:
        print("\nNo matching water features found!")
        return None

if __name__ == "__main__":
    permanent_water = extract_permanent_water()
