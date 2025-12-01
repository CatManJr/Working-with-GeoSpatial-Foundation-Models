"""
FastAPI Backend for Flood Risk Analysis Dashboard
Serves geospatial data and statistics for interactive visualization
Uses File Geodatabase for centralized data management

Production: Serves built React app as static files
Development: API only with CORS enabled for separate React dev server
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
import rasterio
from rasterio.warp import transform_bounds
import geopandas as gpd
import numpy as np
import pandas as pd
import json
from pathlib import Path
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import shape
import sys
import os

# Add parent directory to path to import paths.py
sys.path.append(str(Path(__file__).parent.parent.parent))
from paths import DATA_DIR, CITY_BOUNDARY

# Import file geodatabase
from file_geodatabase import get_geodatabase

# Check if running in production mode (frontend build exists)
FRONTEND_BUILD = Path(__file__).parent.parent / "frontend" / "build"
PRODUCTION_MODE = FRONTEND_BUILD.exists()

app = FastAPI(
    title="Fort Myers Flood Risk Analysis API",
    description="Full-stack flood risk analysis dashboard with File Geodatabase",
    version="2.0.0"
)

# Enable CORS only in development mode
if not PRODUCTION_MODE:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],  # React dev server
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Initialize file geodatabase
gdb = get_geodatabase()

@app.get("/")
async def root():
    return {
        "message": "Fort Myers Flood Risk Analysis API",
        "version": "2.0.0",
        "features": ["File Geodatabase", "Centralized Data Management"],
        "endpoints": [
            "/api/boundary",
            "/api/flood-extent",
            "/api/statistics",
            "/api/risk-layers",
            "/api/geodatabase/summary",
            "/api/geodatabase/datasets",
            "/api/raster-bounds/{layer}",
            "/api/raster-png/{layer}"
        ]
    }

@app.get("/api/geodatabase/summary")
async def get_geodatabase_summary():
    """Get geodatabase summary"""
    summary = gdb.get_catalog_summary()
    return summary

@app.get("/api/geodatabase/datasets")
async def get_all_datasets(dataset_type: str = None, category: str = None):
    """List all datasets in geodatabase"""
    datasets = gdb.list_datasets(dataset_type=dataset_type, category=category)
    return {"datasets": datasets}

@app.get("/api/boundary")
async def get_city_boundary():
    """Get Fort Myers city boundary as GeoJSON"""
    try:
        # Get boundary from geodatabase
        boundary_dataset = gdb.get_dataset("fort_myers_boundary")
        
        if boundary_dataset:
            boundary = gpd.read_file(boundary_dataset['path'])
        else:
            # Fallback to original path
            boundary = gpd.read_file(CITY_BOUNDARY)
        
        # Convert to WGS84 (EPSG:4326) for web mapping
        boundary_wgs84 = boundary.to_crs("EPSG:4326")
        
        # Convert any datetime/timestamp columns to strings to avoid JSON serialization errors
        for col in boundary_wgs84.columns:
            if col != 'geometry' and pd.api.types.is_datetime64_any_dtype(boundary_wgs84[col]):
                boundary_wgs84[col] = boundary_wgs84[col].astype(str)
        
        return json.loads(boundary_wgs84.to_json())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading boundary: {str(e)}")

@app.get("/api/flood-extent")
async def get_flood_extent():
    """Get flood extent as GeoJSON"""
    try:
        # Get flood raster from geodatabase
        flood_dataset = gdb.get_dataset("flood_extent_helene_2024")
        
        if not flood_dataset:
            raise HTTPException(status_code=404, detail="Flood dataset not found in geodatabase")
        
        # Read flood raster
        with rasterio.open(flood_dataset['path']) as src:
            flood_data = src.read(1)
            
            # Convert raster to polygons
            from rasterio import features
            shapes = features.shapes(flood_data.astype(np.int16), transform=src.transform)
            
            # Filter for flood pixels (value = 1)
            flood_polygons = []
            for geom, value in shapes:
                if value == 1:
                    flood_polygons.append(shape(geom))
            
            if not flood_polygons:
                return {"type": "FeatureCollection", "features": []}
            
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(
                {"flood": [1] * len(flood_polygons)},
                geometry=flood_polygons,
                crs=src.crs
            )
            
            # Convert to WGS84
            gdf_wgs84 = gdf.to_crs("EPSG:4326")
            
            return json.loads(gdf_wgs84.to_json())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading flood extent: {str(e)}")

@app.get("/api/statistics")
async def get_statistics():
    """Get comprehensive flood exposure statistics from geodatabase"""
    try:
        stats = {}
        
        # Get exposure statistics
        exposure_stats = gdb.get_dataset("exposure_statistics")
        if exposure_stats:
            df = pd.read_csv(exposure_stats['path'])
            stats["exposure"] = df.to_dict(orient="records")
            
            # Calculate actual flood area from raster if it's 0 in the stats
            flood_area_stat = df[df['Metric'] == 'Flooded Area']
            if len(flood_area_stat) > 0:
                flood_area_km2 = float(flood_area_stat[flood_area_stat['Unit'] == 'square kilometers'].iloc[0]['Value'])
                
                # If flood area is 0, calculate it from the raster
                if flood_area_km2 == 0:
                    flood_dataset = gdb.get_dataset("flood_extent_helene_2024")
                    if flood_dataset:
                        with rasterio.open(flood_dataset['path']) as src:
                            data = src.read(1)
                            transform = src.transform
                            
                            # Get pixel size in meters
                            pixel_width = abs(transform[0])
                            pixel_height = abs(transform[4])
                            
                            # Count flood pixels (value = 1)
                            flood_pixels = np.sum(data == 1)
                            
                            # Calculate area
                            pixel_area_m2 = pixel_width * pixel_height
                            flood_area_m2 = flood_pixels * pixel_area_m2
                            flood_area_km2_calculated = flood_area_m2 / 1_000_000
                            
                            # Update the stats with calculated values
                            for item in stats["exposure"]:
                                if item['Metric'] == 'Flooded Area' and item['Unit'] == 'square kilometers':
                                    item['Value'] = f"{flood_area_km2_calculated:.4f}"
                                elif item['Metric'] == 'Flooded Area' and item['Unit'] == 'square meters':
                                    item['Value'] = f"{flood_area_m2:.2f}"
                                elif item['Metric'] == 'Pixel Area':
                                    item['Value'] = f"{pixel_area_m2:.2f}"
                                elif item['Metric'] == 'Pixel Resolution X':
                                    item['Value'] = f"{pixel_width:.2f}"
                                elif item['Metric'] == 'Pixel Resolution Y':
                                    item['Value'] = f"{pixel_height:.2f}"
        
        # Get coverage statistics
        coverage_stats = gdb.get_dataset("flood_coverage_statistics")
        if coverage_stats:
            df = pd.read_csv(coverage_stats['path'])
            stats["coverage_categories"] = df.to_dict(orient="records")
        
        # Get G2SFCA statistics for different bandwidths
        stats["g2sfca"] = {}
        for bw in [250, 500, 1000, 2500]:
            g2sfca_dataset = gdb.get_dataset(f"g2sfca_stats_{bw}m")
            if g2sfca_dataset:
                df = pd.read_csv(g2sfca_dataset['path'])
                stats["g2sfca"][f"{bw}m"] = df.to_dict(orient="records")
        
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading statistics: {str(e)}")

@app.get("/api/risk-layers")
async def get_risk_layers():
    """Get available risk analysis layers from geodatabase"""
    layers = {}
    
    # Flood layer
    flood_dataset = gdb.get_dataset("flood_extent_helene_2024")
    if flood_dataset:
        layers['flood'] = {
            "name": "Flood Extent",
            "file": flood_dataset['path'],
            "type": "binary",
            "colormap": "Blues",
            "description": flood_dataset.get('description', '')
        }
    
    # Population layer
    pop_dataset = gdb.get_dataset("population_worldpop")
    if pop_dataset:
        layers['population'] = {
            "name": "Population Density",
            "file": pop_dataset['path'],
            "type": "continuous",
            "colormap": "YlOrRd",
            "unit": "people/hectare",
            "description": pop_dataset.get('description', '')
        }
    
    # Flood coverage rate
    coverage_dataset = gdb.get_dataset("flood_coverage_rate")
    if coverage_dataset:
        layers['exposure'] = {
            "name": "Flood Coverage Rate",
            "file": coverage_dataset['path'],
            "type": "percentage",
            "colormap": "Blues",
            "unit": "%",
            "description": coverage_dataset.get('description', '')
        }
    
    # Exposed population
    exposed_dataset = gdb.get_dataset("exposed_population")
    if exposed_dataset:
        layers['exposed_population'] = {
            "name": "Exposed Population",
            "file": exposed_dataset['path'],
            "type": "continuous",
            "colormap": "Reds",
            "unit": "people/hectare",
            "description": exposed_dataset.get('description', '')
        }
    
    # G2SFCA risk layers
    for bw in [250, 500, 1000, 2500]:
        risk_dataset = gdb.get_dataset(f"g2sfca_risk_{bw}m")
        if risk_dataset:
            layers[f'g2sfca_{bw}m'] = {
                "name": f"G2SFCA Risk ({bw}m)",
                "file": risk_dataset['path'],
                "type": "risk_score",
                "colormap": "RdPu",
                "bandwidth": bw,
                "unit": "risk score",
                "description": risk_dataset.get('description', '')
            }
    
    return layers

@app.get("/api/raster-bounds/{layer}")
async def get_raster_bounds(layer: str):
    """Get bounds of a raster layer in WGS84"""
    try:
        layers = await get_risk_layers()
        if layer not in layers:
            raise HTTPException(status_code=404, detail=f"Layer {layer} not found")
        
        raster_file = layers[layer]["file"]
        
        with rasterio.open(raster_file) as src:
            bounds = src.bounds
            bounds_wgs84 = transform_bounds(
                src.crs,
                "EPSG:4326",
                bounds.left, bounds.bottom, bounds.right, bounds.top
            )
            
            return {
                "bounds": bounds_wgs84,
                "center": [
                    (bounds_wgs84[1] + bounds_wgs84[3]) / 2,  # lat
                    (bounds_wgs84[0] + bounds_wgs84[2]) / 2   # lon
                ],
                "crs": str(src.crs)
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting bounds: {str(e)}")

@app.get("/api/raster-png/{layer}")
async def get_raster_as_png(layer: str, width: int = 800):
    """
    Convert raster to PNG for overlay on web map
    Returns transparent PNG with color mapping
    """
    try:
        layers = await get_risk_layers()
        if layer not in layers:
            raise HTTPException(status_code=404, detail=f"Layer {layer} not found")
        
        raster_file = layers[layer]["file"]
        layer_info = layers[layer]
        
        with rasterio.open(raster_file) as src:
            data = src.read(1)
            
            # Mask invalid data
            if layer == "flood":
                data = np.ma.masked_where(data == 0, data)
            elif layer in ["population", "exposed_population"]:
                data = np.ma.masked_where(data <= 0, data)
            else:
                data = np.ma.masked_where(data <= 0, data)
            
            # Create colormap
            cmap_name = layer_info.get("colormap", "viridis")
            
            if cmap_name == "Blues":
                cmap = plt.cm.Blues
            elif cmap_name == "YlOrRd":
                cmap = plt.cm.YlOrRd
            elif cmap_name == "Reds":
                cmap = plt.cm.Reds
            elif cmap_name == "RdPu":
                cmap = plt.cm.RdPu
            else:
                cmap = plt.cm.viridis
            
            # Normalize data
            if len(data.compressed()) > 0:
                vmin = 0
                vmax = np.percentile(data.compressed(), 98)
                
                # Normalize to 0-1
                data_norm = (data - vmin) / (vmax - vmin)
                data_norm = np.clip(data_norm, 0, 1)
                
                # Apply colormap
                rgba = cmap(data_norm)
                
                # Set alpha channel for masked values
                rgba[:, :, 3] = np.where(data.mask, 0, 0.7)
                
                # Convert to uint8
                rgba_uint8 = (rgba * 255).astype(np.uint8)
                
                # Create PIL Image
                img = Image.fromarray(rgba_uint8, mode='RGBA')
                
                # Resize if needed
                if width != data.shape[1]:
                    aspect_ratio = data.shape[0] / data.shape[1]
                    height = int(width * aspect_ratio)
                    img = img.resize((width, height), Image.LANCZOS)
                
                # Save to bytes
                img_bytes = BytesIO()
                img.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                
                return Response(content=img_bytes.read(), media_type="image/png")
            else:
                raise HTTPException(status_code=404, detail="No valid data in raster")
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating PNG: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # In production, serve static files
    if PRODUCTION_MODE:
        app.mount("/", StaticFiles(directory=str(FRONTEND_BUILD), html=True), name="frontend")
        print(f"🚀 Running in PRODUCTION mode")
        print(f"📁 Serving frontend from: {FRONTEND_BUILD}")
    else:
        print(f"🔧 Running in DEVELOPMENT mode")
        print(f"⚠️  Frontend should run separately on http://localhost:3000")
    
    print(f"📊 File Geodatabase: {gdb.gdb_path}")
    summary = gdb.get_catalog_summary()
    print(f"📈 Datasets: {summary['counts']['rasters']} rasters, {summary['counts']['vectors']} vectors, {summary['counts']['tables']} tables")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

def main():
    """Entry point for uv run"""
    if __name__ == "__main__":
        pass
    else:
        import uvicorn
        if PRODUCTION_MODE:
            app.mount("/", StaticFiles(directory=str(FRONTEND_BUILD), html=True), name="frontend")
        uvicorn.run(app, host="0.0.0.0", port=8000)
