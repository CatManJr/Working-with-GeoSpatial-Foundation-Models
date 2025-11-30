"""
File Database System for Geospatial Data
Automatically scans and indexes data files from the data directory
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import rasterio
import geopandas as gpd
from rasterio.warp import transform_bounds
import numpy as np
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from paths import DATA_DIR

class FileDatabase:
    """
    Manages geospatial data files and their metadata
    """
    
    def __init__(self, data_dir: Path = DATA_DIR, cache_file: str = "file_db_cache.json"):
        self.data_dir = data_dir
        self.cache_file = Path(__file__).parent / cache_file
        self.db: Dict[str, Any] = {}
        
        # File type patterns
        self.raster_extensions = ['.tif', '.tiff']
        self.vector_extensions = ['.shp', '.geojson', '.gpkg']
        
        # Initialize or load cache
        self._load_or_scan()
    
    def _load_or_scan(self):
        """Load cache or perform full scan"""
        if self.cache_file.exists():
            print(f"Loading file database cache from {self.cache_file}")
            with open(self.cache_file, 'r') as f:
                self.db = json.load(f)
            print(f"  Loaded {len(self.db)} files from cache")
        else:
            print("No cache found, performing full scan...")
            self.scan_all()
    
    def scan_all(self):
        """Scan all data directories and build database"""
        print(f"Scanning {self.data_dir} for geospatial files...")
        
        self.db = {
            'rasters': {},
            'vectors': {},
            'metadata': {
                'last_scan': datetime.now().isoformat(),
                'data_dir': str(self.data_dir)
            }
        }
        
        # Scan for raster files
        for ext in self.raster_extensions:
            for raster_file in self.data_dir.rglob(f'*{ext}'):
                try:
                    self._index_raster(raster_file)
                except Exception as e:
                    print(f"  Warning: Failed to index {raster_file}: {e}")
        
        # Scan for vector files
        for ext in self.vector_extensions:
            for vector_file in self.data_dir.rglob(f'*{ext}'):
                try:
                    self._index_vector(vector_file)
                except Exception as e:
                    print(f"  Warning: Failed to index {vector_file}: {e}")
        
        print(f"  Found {len(self.db['rasters'])} raster files")
        print(f"  Found {len(self.db['vectors'])} vector files")
        
        # Save cache
        self._save_cache()
    
    def _index_raster(self, file_path: Path):
        """Extract metadata from a raster file"""
        with rasterio.open(file_path) as src:
            # Get basic info
            data = src.read(1)
            
            # Calculate statistics (avoid reading entire large files)
            if data.size < 10_000_000:  # Only for files < 10M pixels
                valid_data = data[data != src.nodata] if src.nodata else data
                if len(valid_data) > 0:
                    stats = {
                        'min': float(np.min(valid_data)),
                        'max': float(np.max(valid_data)),
                        'mean': float(np.mean(valid_data)),
                        'std': float(np.std(valid_data))
                    }
                else:
                    stats = None
            else:
                stats = None
            
            # Get bounds in WGS84
            bounds_wgs84 = transform_bounds(
                src.crs,
                'EPSG:4326',
                *src.bounds
            )
            
            # Create file ID from relative path
            rel_path = file_path.relative_to(self.data_dir)
            file_id = str(rel_path).replace('/', '_').replace('\\', '_').replace('.', '_')
            
            # Determine category from path
            category = self._categorize_file(rel_path)
            
            metadata = {
                'id': file_id,
                'path': str(file_path),
                'relative_path': str(rel_path),
                'category': category,
                'type': 'raster',
                'crs': str(src.crs),
                'bounds': {
                    'original': list(src.bounds),
                    'wgs84': list(bounds_wgs84)
                },
                'center_wgs84': [
                    (bounds_wgs84[1] + bounds_wgs84[3]) / 2,  # lat
                    (bounds_wgs84[0] + bounds_wgs84[2]) / 2   # lon
                ],
                'shape': [src.height, src.width],
                'resolution': [src.res[0], src.res[1]],
                'nodata': src.nodata,
                'dtype': str(src.dtypes[0]),
                'count': src.count,
                'statistics': stats,
                'size_mb': file_path.stat().st_size / (1024 * 1024)
            }
            
            self.db['rasters'][file_id] = metadata
            print(f"  Indexed raster: {rel_path} -> {file_id}")
    
    def _index_vector(self, file_path: Path):
        """Extract metadata from a vector file"""
        gdf = gpd.read_file(file_path)
        
        # Get bounds in WGS84
        gdf_wgs84 = gdf.to_crs('EPSG:4326')
        bounds = gdf_wgs84.total_bounds  # [minx, miny, maxx, maxy]
        
        # Create file ID
        rel_path = file_path.relative_to(self.data_dir)
        file_id = str(rel_path).replace('/', '_').replace('\\', '_').replace('.', '_')
        
        # Determine category
        category = self._categorize_file(rel_path)
        
        metadata = {
            'id': file_id,
            'path': str(file_path),
            'relative_path': str(rel_path),
            'category': category,
            'type': 'vector',
            'crs': str(gdf.crs),
            'bounds': {
                'wgs84': list(bounds)
            },
            'center_wgs84': [
                (bounds[1] + bounds[3]) / 2,  # lat
                (bounds[0] + bounds[2]) / 2   # lon
            ],
            'geometry_type': gdf.geometry.type.iloc[0] if len(gdf) > 0 else None,
            'feature_count': len(gdf),
            'columns': list(gdf.columns),
            'size_mb': file_path.stat().st_size / (1024 * 1024)
        }
        
        self.db['vectors'][file_id] = metadata
        print(f"  Indexed vector: {rel_path} -> {file_id}")
    
    def _categorize_file(self, rel_path: Path) -> str:
        """Categorize file based on its path"""
        parts = rel_path.parts
        
        if 'flood' in parts:
            return 'flood'
        elif 'pop' in parts or 'population' in str(rel_path).lower():
            return 'population'
        elif 'exposure' in parts:
            return 'exposure'
        elif 'boundary' in str(rel_path).lower():
            return 'boundary'
        elif 'water' in str(rel_path).lower():
            return 'water'
        elif 'IBM' in parts:
            return 'satellite'
        else:
            return 'other'
    
    def _save_cache(self):
        """Save database to cache file"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.db, f, indent=2)
        print(f"Cache saved to {self.cache_file}")
    
    def get_raster(self, file_id: str) -> Optional[Dict]:
        """Get raster metadata by ID"""
        return self.db['rasters'].get(file_id)
    
    def get_vector(self, file_id: str) -> Optional[Dict]:
        """Get vector metadata by ID"""
        return self.db['vectors'].get(file_id)
    
    def search_by_category(self, category: str) -> Dict[str, List[Dict]]:
        """Search files by category"""
        results = {
            'rasters': [],
            'vectors': []
        }
        
        for file_id, metadata in self.db['rasters'].items():
            if metadata['category'] == category:
                results['rasters'].append(metadata)
        
        for file_id, metadata in self.db['vectors'].items():
            if metadata['category'] == category:
                results['vectors'].append(metadata)
        
        return results
    
    def search_by_pattern(self, pattern: str) -> Dict[str, List[Dict]]:
        """Search files by filename pattern"""
        results = {
            'rasters': [],
            'vectors': []
        }
        
        pattern_lower = pattern.lower()
        
        for file_id, metadata in self.db['rasters'].items():
            if pattern_lower in metadata['relative_path'].lower():
                results['rasters'].append(metadata)
        
        for file_id, metadata in self.db['vectors'].items():
            if pattern_lower in metadata['relative_path'].lower():
                results['vectors'].append(metadata)
        
        return results
    
    def list_all_rasters(self) -> List[Dict]:
        """List all indexed rasters"""
        return list(self.db['rasters'].values())
    
    def list_all_vectors(self) -> List[Dict]:
        """List all indexed vectors"""
        return list(self.db['vectors'].values())
    
    def get_categories(self) -> Dict[str, int]:
        """Get count of files per category"""
        categories = {}
        
        for metadata in self.db['rasters'].values():
            cat = metadata['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        for metadata in self.db['vectors'].values():
            cat = metadata['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        return categories
    
    def refresh(self):
        """Refresh the database by re-scanning"""
        print("Refreshing file database...")
        self.scan_all()


# Global instance
_db_instance = None

def get_file_db() -> FileDatabase:
    """Get or create the global FileDatabase instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = FileDatabase()
    return _db_instance


if __name__ == "__main__":
    # Test the database
    print("="*70)
    print("FILE DATABASE TEST")
    print("="*70)
    
    db = FileDatabase()
    
    print("\nCategories:")
    for cat, count in db.get_categories().items():
        print(f"  {cat}: {count} files")
    
    print("\nFlood-related files:")
    flood_files = db.search_by_category('flood')
    for raster in flood_files['rasters']:
        print(f"  [R] {raster['relative_path']}")
    for vector in flood_files['vectors']:
        print(f"  [V] {vector['relative_path']}")
    
    print("\nPopulation files:")
    pop_files = db.search_by_category('population')
    for raster in pop_files['rasters']:
        print(f"  [R] {raster['relative_path']} ({raster['size_mb']:.2f} MB)")
    
    print("\nBoundary files:")
    boundary_files = db.search_by_category('boundary')
    for vector in boundary_files['vectors']:
        print(f"  [V] {vector['relative_path']} ({vector['feature_count']} features)")
    
    print("\n" + "="*70)
