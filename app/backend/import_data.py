"""
Import data from ../data directory into File Geodatabase
One-time setup script to populate the geodatabase
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from paths import DATA_DIR

from file_geodatabase import get_geodatabase


def import_all_data(clean_first=True):
    """Import all data from data directory into geodatabase
    
    Args:
        clean_first: If True, remove all existing datasets before importing
    """
    
    print("=" * 60)
    print("IMPORTING DATA INTO FILE GEODATABASE")
    print("=" * 60)
    print()
    
    # Initialize geodatabase
    gdb = get_geodatabase()
    
    print(f"Geodatabase location: {gdb.gdb_path}")
    print()
    
    # Clean existing data if requested
    if clean_first:
        print("🗑️  Cleaning existing datasets...")
        print("-" * 60)
        
        import sqlite3
        import glob
        
        # Clear SQLite index
        conn = sqlite3.connect(gdb.index_db)
        cursor = conn.cursor()
        
        for table in ['rasters', 'vectors', 'tables']:
            cursor.execute(f"DELETE FROM {table}")
            print(f"  Cleared {table} index")
        
        conn.commit()
        conn.close()
        
        # Clear catalog
        catalog = gdb._load_catalog()
        catalog["datasets"] = {
            "rasters": {},
            "vectors": {},
            "tables": {}
        }
        gdb._save_catalog(catalog)
        print(f"  Cleared catalog")
        
        # Remove physical files
        for raster_file in gdb.rasters_dir.rglob("*.tif"):
            raster_file.unlink()
            
        for vector_file in gdb.vectors_dir.rglob("*.gpkg"):
            vector_file.unlink()
            
        for table_file in gdb.tables_dir.rglob("*.csv"):
            table_file.unlink()
        
        print(f"  ✓ Cleaned all existing datasets")
        print()
    
    # --- Import Rasters ---
    print("Importing Rasters...")
    print("-" * 60)
    
    # Flood raster
    flood_raster = DATA_DIR / "flood" / "FortMyersHelene_2024T269_flood_clipped.tif"
    if flood_raster.exists():
        gdb.import_raster(
            flood_raster,
            name="flood_extent_helene_2024",
            category="flood",
            description="Hurricane Helene 2024 flood extent (clipped to Fort Myers)",
            copy=True  # Changed to True for portability
        )
    
    # Population raster
    pop_raster = DATA_DIR / "pop" / "fort_myers_worldpop.tif"
    if pop_raster.exists():
        gdb.import_raster(
            pop_raster,
            name="population_worldpop",
            category="population",
            description="WorldPop population density for Fort Myers",
            copy=True  # Changed to True for portability
        )
    
    # Exposure rasters
    exposure_dir = DATA_DIR / "pop_exposure"
    
    coverage_rate = exposure_dir / "flood_coverage_rate.tif"
    if coverage_rate.exists():
        gdb.import_raster(
            coverage_rate,
            name="flood_coverage_rate",
            category="exposure",
            description="Flood coverage rate per 100m grid cell",
            copy=True  # Changed to True for portability
        )
    
    exposed_pop = exposure_dir / "population_flood_exposure.tif"
    if exposed_pop.exists():
        gdb.import_raster(
            exposed_pop,
            name="exposed_population",
            category="exposure",
            description="Population exposed to flooding (weighted by coverage)",
            copy=True  # Changed to True for portability
        )
    
    # G2SFCA influence rasters (updated naming)
    for bandwidth in [250, 500, 1000, 2500]:
        # Try new naming first (flood_influence_*)
        influence_file = exposure_dir / f"flood_influence_g2sfca_raster_{bandwidth}m.tif"
        
        # Fallback to old naming (flood_risk_*) for backward compatibility
        if not influence_file.exists():
            influence_file = exposure_dir / f"flood_risk_g2sfca_raster_{bandwidth}m.tif"
        
        if influence_file.exists():
            gdb.import_raster(
                influence_file,
                name=f"g2sfca_influence_{bandwidth}m",
                category="influence",
                description=f"G2SFCA flood influence assessment (bandwidth={bandwidth}m)",
                copy=True
            )
        
        # Distance raster
        dist_file = exposure_dir / f"flood_distance_raster_{bandwidth}m.tif"
        if dist_file.exists():
            gdb.import_raster(
                dist_file,
                name=f"flood_distance_{bandwidth}m",
                category="influence",
                description=f"Distance to nearest flood pixel (bandwidth={bandwidth}m)",
                copy=True
            )
    
    print()
    
    # --- Import Vectors ---
    print("Importing Vectors...")
    print("-" * 60)
    
    # City boundary
    boundary_shp = DATA_DIR / "Fort_Myers_City_Boundary" / "City_Boundary.shp"
    if boundary_shp.exists():
        gdb.import_vector(
            boundary_shp,
            name="fort_myers_boundary",
            category="boundaries",
            description="Fort Myers city boundary",
            copy=True
        )
    
    # Permanent water
    water_shp = DATA_DIR / "permanent_water" / "permanent_water.shp"
    if water_shp.exists():
        gdb.import_vector(
            water_shp,
            name="permanent_water",
            category="boundaries",
            description="Permanent water bodies",
            copy=True
        )
    
    print()
    
    # --- Import Tables ---
    print("Importing Tables...")
    print("-" * 60)
    
    # Exposure statistics
    exposure_stats = exposure_dir / "exposure_statistics.csv"
    if exposure_stats.exists():
        gdb.import_table(
            exposure_stats,
            name="exposure_statistics",
            category="statistics",
            description="Overall flood exposure statistics"
        )
    
    # Coverage statistics
    coverage_stats = exposure_dir / "flood_coverage_statistics.csv"
    if coverage_stats.exists():
        gdb.import_table(
            coverage_stats,
            name="flood_coverage_statistics",
            category="statistics",
            description="Flood coverage by category"
        )
    
    # G2SFCA statistics (updated naming)
    for bandwidth in [250, 500, 1000, 2500]:
        # Try new naming first (flood_influence_*)
        g2sfca_stats = exposure_dir / f"flood_influence_g2sfca_raster_{bandwidth}m_summary.csv"
        
        # Fallback to old naming (flood_risk_*) for backward compatibility
        if not g2sfca_stats.exists():
            g2sfca_stats = exposure_dir / f"flood_risk_g2sfca_raster_{bandwidth}m_summary.csv"
        
        if g2sfca_stats.exists():
            gdb.import_table(
                g2sfca_stats,
                name=f"g2sfca_stats_{bandwidth}m",
                category="statistics",
                description=f"G2SFCA influence statistics (bandwidth={bandwidth}m)"
            )
    
    print()
    
    # --- Print Summary ---
    print("=" * 60)
    print("IMPORT COMPLETE")
    print("=" * 60)
    
    summary = gdb.get_catalog_summary()
    print(f"\nGeodatabase Summary:")
    print(f"  Created: {summary['created']}")
    print(f"  Version: {summary['version']}")
    print(f"\nDataset Counts:")
    print(f"  Rasters: {summary['counts']['rasters']}")
    print(f"  Vectors: {summary['counts']['vectors']}")
    print(f"  Tables:  {summary['counts']['tables']}")
    
    print(f"\nGeodatabase location: {gdb.gdb_path}")
    print()
    
    # List all datasets
    print("All Datasets:")
    print("-" * 60)
    datasets = gdb.list_datasets()
    for ds in datasets:
        print(f"  [{ds['type']}] {ds['name']} ({ds['category']})")
    
    print()
    print("✓ File Geodatabase ready to use!")


if __name__ == "__main__":
    import_all_data()
