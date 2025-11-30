"""
FastAPI Backend for Flood Risk Analysis Dashboard
Serves geospatial data and statistics for interactive visualization
Uses FileDatabase for automatic data discovery and indexing

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
import sys
import os

# Add parent directory to path to import paths.py
sys.path.append(str(Path(__file__).parent.parent.parent))
from paths import DATA_DIR, CITY_BOUNDARY

# Import file database
from file_database import get_file_db

# Check if running in production mode (frontend build exists)
FRONTEND_BUILD = Path(__file__).parent.parent / "frontend" / "build"
PRODUCTION_MODE = FRONTEND_BUILD.exists()

app = FastAPI(
    title="Fort Myers Flood Risk Analysis API",
    description="Full-stack flood risk analysis dashboard",
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

# Initialize file database
file_db = get_file_db()

@app.get("/")
async def root():
    return {
        "message": "Fort Myers Flood Risk Analysis API",
        "version": "2.0.0",
        "features": ["File Database", "Auto-Discovery"],
        "endpoints": [
            "/api/boundary",
            "/api/flood-extent",
            "/api/statistics",
            "/api/risk-layers",
            "/api/database/info",
            "/api/database/search",
            "/api/raster-bounds/{layer}",
            "/api/raster-png/{layer}"
        ]
    }

@app.get("/api/database/info")
async def get_database_info():
    """Get file database information"""
    return {
        "total_rasters": len(file_db.list_all_rasters()),
        "total_vectors": len(file_db.list_all_vectors()),
        "categories": file_db.get_categories(),
        "last_scan": file_db.db.get('metadata', {}).get('last_scan'),
        "data_dir": str(file_db.data_dir)
    }

@app.get("/api/database/search")
async def search_database(category: str = None, pattern: str = None):
    """Search file database by category or pattern"""
    if category:
        results = file_db.search_by_category(category)
    elif pattern:
        results = file_db.search_by_pattern(pattern)
    else:
        results = {
            "rasters": file_db.list_all_rasters(),
            "vectors": file_db.list_all_vectors()
        }
    return results

@app.post("/api/database/refresh")
async def refresh_database():
    """Refresh the file database by re-scanning"""
    file_db.refresh()
    return {
        "status": "success",
        "message": "Database refreshed",
        "total_rasters": len(file_db.list_all_rasters()),
        "total_vectors": len(file_db.list_all_vectors())
    }

@app.get("/api/boundary")
async def get_city_boundary():
    """Get Fort Myers city boundary as GeoJSON"""
    try:
        # Use file database to find boundary
        boundary_files = file_db.search_by_category('boundary')
        
        if boundary_files['vectors']:
            boundary_path = boundary_files['vectors'][0]['path']
            boundary = gpd.read_file(boundary_path)
        else:
            # Fallback to hardcoded path
            boundary = gpd.read_file(CITY_BOUNDARY)
        
        # Convert to WGS84 (EPSG:4326) for web mapping
        boundary_wgs84 = boundary.to_crs("EPSG:4326")
        return json.loads(boundary_wgs84.to_json())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading boundary: {str(e)}")

@app.get("/api/flood-extent")
async def get_flood_extent():
    """Get flood extent as GeoJSON - auto-discovered from database"""
    try:
        # Search for flood raster using file database
        flood_files = file_db.search_by_category('flood')
        
        if not flood_files['rasters']:
            raise HTTPException(status_code=404, detail="No flood raster found in database")
        
        # Use the first clipped flood file if available
        flood_raster = None
        for raster in flood_files['rasters']:
            if 'clipped' in raster['relative_path']:
                flood_raster = raster
                break
        
        if not flood_raster:
            flood_raster = flood_files['rasters'][0]
        
        # Read flood raster
        with rasterio.open(flood_raster['path']) as src:
            flood_data = src.read(1)
            
            # Convert raster to polygons
            from rasterio import features
            shapes = features.shapes(flood_data.astype(np.int16), transform=src.transform)
            
            # Filter for flood pixels (value = 1)
            flood_polygons = []
            for geom, value in shapes:
                if value == 1:
                    flood_polygons.append(geom)
            
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
    """Get comprehensive flood exposure statistics"""
    try:
        stats = {}
        
        # Search for exposure files using database
        exposure_files = file_db.search_by_category('exposure')
        
        # Read exposure statistics
        for raster in exposure_files['rasters']:
            if 'exposure_statistics.csv' in raster['relative_path']:
                df = pd.read_csv(raster['path'])
                stats["exposure"] = df.to_dict(orient="records")
                break
        
        # Try direct path if not found via database
        if "exposure" not in stats:
            exposure_stats_file = DATA_DIR / "pop_exposure" / "exposure_statistics.csv"
            if exposure_stats_file.exists():
                df = pd.read_csv(exposure_stats_file)
                stats["exposure"] = df.to_dict(orient="records")
        
        # Read coverage statistics
        coverage_stats_file = DATA_DIR / "pop_exposure" / "flood_coverage_statistics.csv"
        if coverage_stats_file.exists():
            df = pd.read_csv(coverage_stats_file)
            stats["coverage_categories"] = df.to_dict(orient="records")
        
        # Read G2SFCA statistics for different bandwidths
        bandwidths = [250, 500, 1000, 2500]
        stats["g2sfca"] = {}
        
        for bw in bandwidths:
            g2sfca_file = DATA_DIR / "pop_exposure" / f"flood_risk_g2sfca_raster_{bw}m_summary.csv"
            if g2sfca_file.exists():
                df = pd.read_csv(g2sfca_file)
                stats["g2sfca"][f"{bw}m"] = df.to_dict(orient="records")
        
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading statistics: {str(e)}")

@app.get("/api/risk-layers")
async def get_risk_layers():
    """Get available risk analysis layers - auto-discovered from database"""
    layers = {}
    
    # Get flood layers
    flood_files = file_db.search_by_category('flood')
    for raster in flood_files['rasters']:
        if raster['relative_path'].endswith('_clipped.tif'):
            layers['flood'] = {
                "name": "Flood Extent",
                "file": raster['path'],
                "type": "binary",
                "colormap": "Blues",
                "metadata": raster
            }
            break
    
    # Get population layers
    pop_files = file_db.search_by_category('population')
    for raster in pop_files['rasters']:
        if 'worldpop' in raster['relative_path'].lower():
            layers['population'] = {
                "name": "Population Density",
                "file": raster['path'],
                "type": "continuous",
                "colormap": "YlOrRd",
                "unit": "people/hectare",
                "metadata": raster
            }
            break
    
    # Get exposure layers
    exposure_files = file_db.search_by_category('exposure')
    for raster in exposure_files['rasters']:
        rel_path = raster['relative_path']
        
        if 'coverage_rate' in rel_path:
            layers['exposure'] = {
                "name": "Flood Coverage Rate",
                "file": raster['path'],
                "type": "percentage",
                "colormap": "Blues",
                "unit": "%",
                "metadata": raster
            }
        elif 'population_flood_exposure' in rel_path:
            layers['exposed_population'] = {
                "name": "Exposed Population",
                "file": raster['path'],
                "type": "continuous",
                "colormap": "Reds",
                "unit": "people/hectare",
                "metadata": raster
            }
        elif 'g2sfca_raster' in rel_path and not 'distance' in rel_path:
            # Extract bandwidth from filename
            import re
            match = re.search(r'(\d+)m', rel_path)
            if match:
                bw = match.group(1)
                layers[f'g2sfca_{bw}m'] = {
                    "name": f"G2SFCA Risk ({bw}m)",
                    "file": raster['path'],
                    "type": "risk_score",
                    "colormap": "RdPu",
                    "bandwidth": int(bw),
                    "unit": "risk score",
                    "metadata": raster
                }
    
    return layers

@app.get("/api/raster-bounds/{layer}")
async def get_raster_bounds(layer: str):
    """Get bounds of a raster layer in WGS84"""
    try:
        layers = await get_risk_layers()
        if layer not in layers:
            raise HTTPException(status_code=404, detail=f"Layer {layer} not found")
        
        layer_info = layers[layer]
        metadata = layer_info.get('metadata')
        
        if metadata:
            # Use cached metadata from database
            return {
                "bounds": metadata['bounds']['wgs84'],
                "center": metadata['center_wgs84'],
                "crs": metadata['crs']
            }
        else:
            # Fallback to reading file
            raster_file = layer_info["file"]
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
                        (bounds_wgs84[1] + bounds_wgs84[3]) / 2,
                        (bounds_wgs84[0] + bounds_wgs84[2]) / 2
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
