import rasterio
import os
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_geotiff_files(directory_path):
    """
    Analyze GeoTIFF files in a directory to check geographic extent and resolution consistency.
    
    Parameters:
    directory_path (str): Path to directory containing GeoTIFF files
    
    Returns:
    pandas.DataFrame: Analysis results for all GeoTIFF files
    """
    
    # Get all .tif files in the directory
    tif_files = list(Path(directory_path).glob("*.tif"))
    
    if not tif_files:
        print("No GeoTIFF files found in the directory.")
        return None
    
    results = []
    
    for file_path in tif_files:
        try:
            with rasterio.open(file_path) as src:
                # Get basic information
                info = {
                    'filename': file_path.name,
                    'width': src.width,
                    'height': src.height,
                    'count': src.count,
                    'dtype': str(src.dtypes[0]),
                    'crs': str(src.crs),
                    'nodata': src.nodata,
                }
                
                # Get geographic bounds
                bounds = src.bounds
                info.update({
                    'left': bounds.left,
                    'bottom': bounds.bottom,
                    'right': bounds.right,
                    'top': bounds.top,
                })
                
                # Get resolution
                transform = src.transform
                info.update({
                    'pixel_size_x': abs(transform[0]),
                    'pixel_size_y': abs(transform[4]),
                    'rotation_x': transform[1],
                    'rotation_y': transform[3],
                })
                
                # Calculate extent
                info.update({
                    'extent_width': bounds.right - bounds.left,
                    'extent_height': bounds.top - bounds.bottom,
                })
                
                results.append(info)
                
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
            results.append({
                'filename': file_path.name,
                'error': str(e)
            })
    
    return pd.DataFrame(results)

def check_consistency(df):
    """
    Check if all GeoTIFF files have consistent geographic extent and resolution.
    
    Parameters:
    df (pandas.DataFrame): DataFrame from analyze_geotiff_files
    
    Returns:
    dict: Consistency check results
    """
    
    if df is None or df.empty:
        return {"error": "No data to analyze"}
    
    # Remove rows with errors
    valid_df = df[~df['filename'].str.contains('error', na=False)]
    
    if len(valid_df) == 0:
        return {"error": "No valid GeoTIFF files found"}
    
    if len(valid_df) == 1:
        return {"warning": "Only one GeoTIFF file found, cannot compare consistency"}
    
    consistency_results = {}
    
    # Check CRS consistency
    crs_unique = valid_df['crs'].nunique()
    consistency_results['crs_consistent'] = crs_unique == 1
    consistency_results['crs_count'] = crs_unique
    consistency_results['crs_values'] = valid_df['crs'].unique().tolist()
    
    # Check resolution consistency
    pixel_size_x_unique = valid_df['pixel_size_x'].nunique()
    pixel_size_y_unique = valid_df['pixel_size_y'].nunique()
    consistency_results['resolution_consistent'] = (pixel_size_x_unique == 1 and pixel_size_y_unique == 1)
    consistency_results['pixel_size_x_values'] = valid_df['pixel_size_x'].unique().tolist()
    consistency_results['pixel_size_y_values'] = valid_df['pixel_size_y'].unique().tolist()
    
    # Check geographic extent overlap
    min_left = valid_df['left'].min()
    max_right = valid_df['right'].max()
    min_bottom = valid_df['bottom'].min()
    max_top = valid_df['top'].max()
    
    consistency_results['extent_info'] = {
        'min_left': min_left,
        'max_right': max_right,
        'min_bottom': min_bottom,
        'max_top': max_top,
        'total_width': max_right - min_left,
        'total_height': max_top - min_bottom
    }
    
    # Check if all files have the same extent
    extent_consistent = (
        valid_df['left'].nunique() == 1 and
        valid_df['right'].nunique() == 1 and
        valid_df['bottom'].nunique() == 1 and
        valid_df['top'].nunique() == 1
    )
    consistency_results['extent_consistent'] = extent_consistent
    
    # Check dimensions consistency
    width_consistent = valid_df['width'].nunique() == 1
    height_consistent = valid_df['height'].nunique() == 1
    consistency_results['dimensions_consistent'] = width_consistent and height_consistent
    consistency_results['width_values'] = valid_df['width'].unique().tolist()
    consistency_results['height_values'] = valid_df['height'].unique().tolist()
    
    return consistency_results

# Analyze the raw directory
raw_directory = "/Volumes/WD_BLACK/FortMyers/data/raw"
print("Analyzing GeoTIFF files in raw directory...")
print("=" * 50)

# Get analysis results
df = analyze_geotiff_files(raw_directory)

if df is not None:
    print("\nDetailed Analysis Results:")
    print("-" * 30)
    print(df.to_string(index=False))
    
    print("\nConsistency Check:")
    print("-" * 30)
    consistency = check_consistency(df)
    
    for key, value in consistency.items():
        if key == 'extent_info':
            print(f"\n{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print("=" * 50)
    
    if consistency.get('crs_consistent', False):
        print("✓ All files have the same CRS")
    else:
        print("✗ Files have different CRS values")
    
    if consistency.get('resolution_consistent', False):
        print("✓ All files have the same pixel resolution")
    else:
        print("✗ Files have different pixel resolutions")
    
    if consistency.get('extent_consistent', False):
        print("✓ All files have the same geographic extent")
    else:
        print("✗ Files have different geographic extents")
    
    if consistency.get('dimensions_consistent', False):
        print("✓ All files have the same dimensions (width x height)")
    else:
        print("✗ Files have different dimensions")
else:
    print("No GeoTIFF files found or error occurred.")