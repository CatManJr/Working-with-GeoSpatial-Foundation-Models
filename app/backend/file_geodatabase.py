"""
File Geodatabase Manager - Similar to ArcGIS File Geodatabase
Organizes and manages all geospatial data in a structured directory
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
import rasterio
from rasterio.warp import transform_bounds
import geopandas as gpd
import pandas as pd
from typing import Dict, List, Optional
import sqlite3


class FileGeodatabase:
    """
    File Geodatabase - Centralized geospatial data management
    Similar to ArcGIS File Geodatabase structure
    """
    
    def __init__(self, gdb_path: Path):
        """
        Initialize File Geodatabase
        
        Args:
            gdb_path: Path to the geodatabase directory
        """
        self.gdb_path = Path(gdb_path)
        self.rasters_dir = self.gdb_path / "rasters"
        self.vectors_dir = self.gdb_path / "vectors"
        self.tables_dir = self.gdb_path / "tables"
        self.metadata_dir = self.gdb_path / "metadata"
        
        # Create structure
        self._initialize_structure()
        
        # Initialize catalog
        self.catalog_file = self.metadata_dir / "catalog.json"
        self.index_db = self.metadata_dir / "index.db"
        self._initialize_catalog()
        self._initialize_index()
    
    def _initialize_structure(self):
        """Create geodatabase directory structure"""
        # Main directories
        self.rasters_dir.mkdir(parents=True, exist_ok=True)
        self.vectors_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Raster subdirectories
        (self.rasters_dir / "flood").mkdir(exist_ok=True)
        (self.rasters_dir / "population").mkdir(exist_ok=True)
        (self.rasters_dir / "exposure").mkdir(exist_ok=True)
        (self.rasters_dir / "risk").mkdir(exist_ok=True)
        
        # Vector subdirectories
        (self.vectors_dir / "boundaries").mkdir(exist_ok=True)
        (self.vectors_dir / "flood_polygons").mkdir(exist_ok=True)
        
        # Table subdirectories
        (self.tables_dir / "statistics").mkdir(exist_ok=True)
    
    def _initialize_catalog(self):
        """Initialize or load catalog"""
        if not self.catalog_file.exists():
            catalog = {
                "created": datetime.now().isoformat(),
                "version": "1.0",
                "description": "Fort Myers Flood Risk Analysis File Geodatabase",
                "datasets": {
                    "rasters": {},
                    "vectors": {},
                    "tables": {}
                }
            }
            self._save_catalog(catalog)
    
    def _initialize_index(self):
        """Initialize SQLite index database"""
        conn = sqlite3.connect(self.index_db)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rasters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                category TEXT,
                path TEXT,
                crs TEXT,
                bounds_wgs84 TEXT,
                width INTEGER,
                height INTEGER,
                added_date TEXT,
                description TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                category TEXT,
                path TEXT,
                crs TEXT,
                bounds_wgs84 TEXT,
                feature_count INTEGER,
                geometry_type TEXT,
                added_date TEXT,
                description TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tables (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                category TEXT,
                path TEXT,
                row_count INTEGER,
                column_count INTEGER,
                added_date TEXT,
                description TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _save_catalog(self, catalog: dict):
        """Save catalog to JSON"""
        with open(self.catalog_file, 'w') as f:
            json.dump(catalog, f, indent=2)
    
    def _load_catalog(self) -> dict:
        """Load catalog from JSON"""
        with open(self.catalog_file, 'r') as f:
            return json.load(f)
    
    def import_raster(self, source_path: Path, name: str, category: str, 
                     description: str = "", copy: bool = True) -> Path:
        """
        Import raster dataset into geodatabase
        
        Args:
            source_path: Source raster file path
            name: Dataset name
            category: Category (flood, population, exposure, risk)
            description: Dataset description
            copy: If True, copy file; if False, create symlink
            
        Returns:
            Path to the imported raster in geodatabase
        """
        # Determine target directory
        target_dir = self.rasters_dir / category
        target_dir.mkdir(exist_ok=True)
        
        # Target path
        target_path = target_dir / f"{name}.tif"
        
        # Copy or symlink
        if copy:
            shutil.copy2(source_path, target_path)
        else:
            if target_path.exists():
                target_path.unlink()
            target_path.symlink_to(source_path.absolute())
        
        # Read metadata
        with rasterio.open(target_path) as src:
            crs = str(src.crs)
            width = src.width
            height = src.height
            bounds = src.bounds
            
            # Transform bounds to WGS84
            bounds_wgs84 = transform_bounds(
                src.crs, "EPSG:4326",
                bounds.left, bounds.bottom, bounds.right, bounds.top
            )
        
        # Add to index
        conn = sqlite3.connect(self.index_db)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO rasters 
            (name, category, path, crs, bounds_wgs84, width, height, added_date, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            name, category, str(target_path), crs,
            json.dumps(bounds_wgs84), width, height,
            datetime.now().isoformat(), description
        ))
        conn.commit()
        conn.close()
        
        # Update catalog
        catalog = self._load_catalog()
        catalog["datasets"]["rasters"][name] = {
            "category": category,
            "path": str(target_path.relative_to(self.gdb_path)),
            "description": description,
            "added": datetime.now().isoformat()
        }
        self._save_catalog(catalog)
        
        print(f"✓ Imported raster: {name} -> {category}")
        return target_path
    
    def import_vector(self, source_path: Path, name: str, category: str,
                     description: str = "", copy: bool = True) -> Path:
        """
        Import vector dataset into geodatabase
        
        Args:
            source_path: Source vector file path (shapefile, GeoJSON, etc.)
            name: Dataset name
            category: Category (boundaries, flood_polygons, etc.)
            description: Dataset description
            copy: If True, copy file; if False, create symlink
            
        Returns:
            Path to the imported vector in geodatabase
        """
        # Determine target directory
        target_dir = self.vectors_dir / category
        target_dir.mkdir(exist_ok=True)
        
        # Read and save as GeoPackage (single file format)
        gdf = gpd.read_file(source_path)
        target_path = target_dir / f"{name}.gpkg"
        gdf.to_file(target_path, driver="GPKG")
        
        # Read metadata
        crs = str(gdf.crs)
        feature_count = len(gdf)
        geometry_type = gdf.geometry.type.iloc[0] if len(gdf) > 0 else "Unknown"
        bounds = gdf.total_bounds
        
        # Transform bounds to WGS84
        gdf_wgs84 = gdf.to_crs("EPSG:4326")
        bounds_wgs84 = gdf_wgs84.total_bounds
        
        # Add to index
        conn = sqlite3.connect(self.index_db)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO vectors 
            (name, category, path, crs, bounds_wgs84, feature_count, geometry_type, added_date, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            name, category, str(target_path), crs,
            json.dumps(bounds_wgs84.tolist()), feature_count, geometry_type,
            datetime.now().isoformat(), description
        ))
        conn.commit()
        conn.close()
        
        # Update catalog
        catalog = self._load_catalog()
        catalog["datasets"]["vectors"][name] = {
            "category": category,
            "path": str(target_path.relative_to(self.gdb_path)),
            "description": description,
            "added": datetime.now().isoformat()
        }
        self._save_catalog(catalog)
        
        print(f"✓ Imported vector: {name} -> {category}")
        return target_path
    
    def import_table(self, source_path: Path, name: str, category: str,
                    description: str = "") -> Path:
        """
        Import table (CSV, Excel) into geodatabase
        
        Args:
            source_path: Source table file path
            name: Dataset name
            category: Category (statistics, etc.)
            description: Dataset description
            
        Returns:
            Path to the imported table in geodatabase
        """
        # Determine target directory
        target_dir = self.tables_dir / category
        target_dir.mkdir(exist_ok=True)
        
        # Read and save as CSV
        if source_path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(source_path)
        else:
            df = pd.read_csv(source_path)
        
        target_path = target_dir / f"{name}.csv"
        df.to_csv(target_path, index=False)
        
        # Add to index
        conn = sqlite3.connect(self.index_db)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO tables 
            (name, category, path, row_count, column_count, added_date, description)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            name, category, str(target_path), len(df), len(df.columns),
            datetime.now().isoformat(), description
        ))
        conn.commit()
        conn.close()
        
        # Update catalog
        catalog = self._load_catalog()
        catalog["datasets"]["tables"][name] = {
            "category": category,
            "path": str(target_path.relative_to(self.gdb_path)),
            "description": description,
            "added": datetime.now().isoformat()
        }
        self._save_catalog(catalog)
        
        print(f"✓ Imported table: {name} -> {category}")
        return target_path
    
    def list_datasets(self, dataset_type: str = None, category: str = None) -> List[Dict]:
        """
        List datasets in geodatabase
        
        Args:
            dataset_type: Filter by type ('rasters', 'vectors', 'tables')
            category: Filter by category
            
        Returns:
            List of dataset information
        """
        conn = sqlite3.connect(self.index_db)
        cursor = conn.cursor()
        
        results = []
        
        # Query based on type
        types_to_query = [dataset_type] if dataset_type else ['rasters', 'vectors', 'tables']
        
        for dtype in types_to_query:
            query = f"SELECT * FROM {dtype}"
            params = []
            
            if category:
                query += " WHERE category = ?"
                params.append(category)
            
            cursor.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            
            for row in cursor.fetchall():
                item = dict(zip(columns, row))
                item['type'] = dtype[:-1]  # Remove 's' from end
                results.append(item)
        
        conn.close()
        return results
    
    def get_dataset(self, name: str) -> Optional[Dict]:
        """Get dataset information by name"""
        conn = sqlite3.connect(self.index_db)
        cursor = conn.cursor()
        
        # Try each table
        for table in ['rasters', 'vectors', 'tables']:
            cursor.execute(f"SELECT * FROM {table} WHERE name = ?", (name,))
            columns = [desc[0] for desc in cursor.description]
            row = cursor.fetchone()
            
            if row:
                conn.close()
                item = dict(zip(columns, row))
                item['type'] = table[:-1]
                return item
        
        conn.close()
        return None
    
    def get_catalog_summary(self) -> Dict:
        """Get geodatabase summary"""
        catalog = self._load_catalog()
        
        conn = sqlite3.connect(self.index_db)
        cursor = conn.cursor()
        
        summary = {
            "created": catalog["created"],
            "version": catalog["version"],
            "description": catalog["description"],
            "counts": {}
        }
        
        for table in ['rasters', 'vectors', 'tables']:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            summary["counts"][table] = count
        
        conn.close()
        return summary


# Global geodatabase instance
_gdb_instance = None

def get_geodatabase(gdb_path: Path = None) -> FileGeodatabase:
    """Get or create global geodatabase instance"""
    global _gdb_instance
    
    if _gdb_instance is None:
        if gdb_path is None:
            # Default path: app/file_database
            gdb_path = Path(__file__).parent.parent / "file_database"
        _gdb_instance = FileGeodatabase(gdb_path)
    
    return _gdb_instance
