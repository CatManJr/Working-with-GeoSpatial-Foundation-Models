"""
Calculate population exposure to flooding using coverage rate method
Treats 100m population grid as community atomic units and calculates flood coverage rate
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import geopandas as gpd
from paths import DATA_DIR, FLOOD_CLIPPED, CITY_BOUNDARY, ensure_dir

# Nature journal style configuration
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.0,
})


def create_visualization(pop_data, flood_coverage, exposure_weighted, 
                         pop_transform, pop_crs, coverage_stats, 
                         total_population, exposed_population,
                         output_dir):
    """
    Create nature-style minimalist visualization maps
    """
    print(f"\nCreating visualizations...")
    
    # Load city boundary for context
    try:
        boundary = gpd.read_file(CITY_BOUNDARY)
        boundary = boundary.to_crs(pop_crs)
        bounds = boundary.total_bounds
    except:
        bounds = None
    
    # Get raster extent
    height, width = pop_data.shape
    left = pop_transform[2]
    top = pop_transform[5]
    right = left + width * pop_transform[0]
    bottom = top + height * pop_transform[4]
    extent = [left, right, bottom, top]
    
    # === Figure 1: Three-panel overview ===
    fig = plt.figure(figsize=(14, 4.5))
    
    # Panel A: Population Density
    ax1 = plt.subplot(131)
    pop_masked = np.ma.masked_where(pop_data <= 0, pop_data)
    im1 = ax1.imshow(pop_masked, cmap='YlOrRd', interpolation='nearest', 
                     extent=extent, vmin=0, vmax=np.percentile(pop_masked.compressed(), 98))
    if bounds is not None:
        boundary.boundary.plot(ax=ax1, color='black', linewidth=1, alpha=0.8)
    ax1.set_title('A. Population Density', fontweight='bold', loc='left')
    ax1.set_xlabel('Easting (m)')
    ax1.set_ylabel('Northing (m)')
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('People per hectare', rotation=270, labelpad=15, fontsize=8)
    ax1.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
    
    # Panel B: Flood Coverage Rate
    ax2 = plt.subplot(132)
    coverage_masked = np.ma.masked_where(pop_data <= 0, flood_coverage)
    colors_coverage = ['#ffffff', '#e0f3db', '#a8ddb5', '#43a2ca', '#0868ac']
    cmap_coverage = LinearSegmentedColormap.from_list('flood_coverage', colors_coverage, N=256)
    im2 = ax2.imshow(coverage_masked, cmap=cmap_coverage, interpolation='nearest',
                     extent=extent, vmin=0, vmax=1)
    if bounds is not None:
        boundary.boundary.plot(ax=ax2, color='black', linewidth=1, alpha=0.8)
    ax2.set_title('B. Flood Coverage Rate', fontweight='bold', loc='left')
    ax2.set_xlabel('Easting (m)')
    ax2.set_ylabel('Northing (m)')
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Coverage (%)', rotation=270, labelpad=15, fontsize=8)
    cbar2.ax.set_yticklabels([f'{int(x*100)}' for x in cbar2.get_ticks()])
    ax2.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
    
    # Panel C: Exposed Population
    ax3 = plt.subplot(133)
    exposure_masked = np.ma.masked_where(exposure_weighted <= 0, exposure_weighted)
    im3 = ax3.imshow(exposure_masked, cmap='Reds', interpolation='nearest',
                     extent=extent, vmin=0, vmax=np.percentile(exposure_masked.compressed(), 98))
    if bounds is not None:
        boundary.boundary.plot(ax=ax3, color='black', linewidth=1, alpha=0.8)
    ax3.set_title('C. Exposed Population', fontweight='bold', loc='left')
    ax3.set_xlabel('Easting (m)')
    ax3.set_ylabel('Northing (m)')
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label('People per hectare', rotation=270, labelpad=15, fontsize=8)
    ax3.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
    
    plt.suptitle('Population Flood Exposure Analysis - Fort Myers, FL (Hurricane Helene 2024)',
                 fontsize=11, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    output_file1 = output_dir / "Fig1_exposure_overview.png"
    plt.savefig(output_file1, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file1}")
    plt.close()
    
    # === Figure 2: Risk Classification Map ===
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Classify coverage into risk categories
    risk_zones = np.zeros_like(flood_coverage, dtype=np.int8)
    risk_zones[pop_data <= 0] = -1  # No data
    risk_zones[(pop_data > 0) & (flood_coverage == 0)] = 0  # No flood
    risk_zones[(pop_data > 0) & (flood_coverage > 0) & (flood_coverage <= 0.25)] = 1  # Minor
    risk_zones[(pop_data > 0) & (flood_coverage > 0.25) & (flood_coverage <= 0.5)] = 2  # Moderate
    risk_zones[(pop_data > 0) & (flood_coverage > 0.5) & (flood_coverage <= 0.75)] = 3  # Severe
    risk_zones[(pop_data > 0) & (flood_coverage > 0.75)] = 4  # Extreme
    
    risk_zones_masked = np.ma.masked_where(risk_zones < 0, risk_zones)
    
    # Nature-style color scheme: muted, professional
    colors_risk = ['#f7f7f7', '#fee5d9', '#fcae91', '#fb6a4a', '#cb181d']
    cmap_risk = LinearSegmentedColormap.from_list('risk', colors_risk, N=5)
    bounds_risk = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm_risk = BoundaryNorm(bounds_risk, cmap_risk.N)
    
    im = ax.imshow(risk_zones_masked, cmap=cmap_risk, norm=norm_risk, 
                   interpolation='nearest', extent=extent)
    
    if bounds is not None:
        boundary.boundary.plot(ax=ax, color='black', linewidth=1.5, alpha=0.9)
    
    # Custom legend
    legend_elements = [
        mpatches.Patch(facecolor='#f7f7f7', edgecolor='black', linewidth=0.5, label='No Flood (0%)'),
        mpatches.Patch(facecolor='#fee5d9', edgecolor='black', linewidth=0.5, label='Minor (1–25%)'),
        mpatches.Patch(facecolor='#fcae91', edgecolor='black', linewidth=0.5, label='Moderate (26–50%)'),
        mpatches.Patch(facecolor='#fb6a4a', edgecolor='black', linewidth=0.5, label='Severe (51–75%)'),
        mpatches.Patch(facecolor='#cb181d', edgecolor='black', linewidth=0.5, label='Extreme (76–100%)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.95, 
              edgecolor='black', title='Flood Coverage')
    
    ax.set_title('Flood Risk Classification by Coverage Rate', fontweight='bold', pad=15)
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
    
    # Add scale bar (approximate)
    from matplotlib.patches import Rectangle
    scale_length = 1000  # 1 km in meters
    scale_x = extent[0] + (extent[1] - extent[0]) * 0.05
    scale_y = extent[2] + (extent[3] - extent[2]) * 0.05
    ax.add_patch(Rectangle((scale_x, scale_y), scale_length, 50, 
                           facecolor='black', edgecolor='white', linewidth=1))
    ax.text(scale_x + scale_length/2, scale_y + 200, '1 km', 
            ha='center', va='bottom', fontsize=8, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    output_file2 = output_dir / "Fig2_risk_classification.png"
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file2}")
    plt.close()
    
    # === Figure 3: Statistical Summary ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Bar chart: Coverage categories
    categories = [stat['category'] for stat in coverage_stats]
    populations = [stat['total_population'] for stat in coverage_stats]
    exposed = [stat['exposed_population'] for stat in coverage_stats]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, populations, width, label='Total Population',
                    color='#4292c6', edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, exposed, width, label='Exposed Population',
                    color='#cb181d', edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('Flood Coverage Category', fontweight='bold')
    ax1.set_ylabel('Population', fontweight='bold')
    ax1.set_title('Population Distribution by Flood Coverage', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.split('(')[0].strip() for c in categories], rotation=15, ha='right')
    ax1.legend(frameon=True, edgecolor='black')
    ax1.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Pie chart: Overall exposure
    exposure_pct = (exposed_population / total_population * 100)
    safe_pct = 100 - exposure_pct
    
    sizes = [exposed_population, total_population - exposed_population]
    labels = [f'Exposed\n{exposed_population:,.0f}\n({exposure_pct:.1f}%)',
              f'Safe\n{total_population - exposed_population:,.0f}\n({safe_pct:.1f}%)']
    colors_pie = ['#cb181d', '#41ab5d']
    explode = (0.05, 0)
    
    ax2.pie(sizes, labels=labels, colors=colors_pie, autopct='',
            startangle=90, explode=explode, textprops={'fontsize': 9, 'fontweight': 'bold'},
            wedgeprops={'edgecolor': 'black', 'linewidth': 1})
    ax2.set_title('Overall Population Exposure', fontweight='bold')
    
    plt.tight_layout()
    output_file3 = output_dir / "Fig3_statistical_summary.png"
    plt.savefig(output_file3, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file3}")
    plt.close()
    
    print(f"\n✓ All visualizations saved to: {output_dir}")


def calculate_exposure():
    """
    Calculate flood coverage rate for each 100m population cell
    Core concept: 100m grid = community atomic unit, calculate flood coverage for each unit
    """
    population_raster = DATA_DIR / "pop" / "fort_myers_worldpop.tif"
    flood_raster = FLOOD_CLIPPED
    output_dir = ensure_dir(DATA_DIR / "pop_exposure")
    
    print("="*60)
    print("FLOOD EXPOSURE ANALYSIS - Coverage Rate Method")
    print("="*60)
    print("Concept: 100m population grid = Community atomic unit")
    print("Method: Calculate flood coverage % for each unit")
    print("="*60)
    
    # Read population data (100m resolution)
    with rasterio.open(population_raster) as pop_src:
        pop_data = pop_src.read(1)
        pop_transform = pop_src.transform
        pop_crs = pop_src.crs
        pop_nodata = pop_src.nodata
        
        print(f"\nPopulation Grid (Community Units):")
        print(f"  Resolution: {pop_transform[0]:.0f}m × {abs(pop_transform[4]):.0f}m")
        print(f"  Grid size: {pop_data.shape}")
        print(f"  Each cell: ~1 hectare (100m × 100m)")
        
        # Read flood data and calculate coverage rate
        with rasterio.open(flood_raster) as flood_src:
            print(f"\nFlood Grid:")
            print(f"  Resolution: {flood_src.transform[0]:.0f}m × {abs(flood_src.transform[4]):.0f}m")
            print(f"  Grid size: {flood_src.shape}")
            print(f"  Sub-grids per 100m unit: 10×10 = 100 cells")
            
            # Calculate coverage rate using AVERAGE resampling
            # This gives us the percentage of each 100m cell that is flooded
            flood_coverage = np.empty(pop_data.shape, dtype=np.float32)
            
            print(f"\nCalculating flood coverage rate...")
            print(f"  Method: Average aggregation (10m → 100m)")
            
            reproject(
                source=rasterio.band(flood_src, 1),
                destination=flood_coverage,
                src_transform=flood_src.transform,
                src_crs=flood_src.crs,
                dst_transform=pop_transform,
                dst_crs=pop_crs,
                resampling=Resampling.average  # Key: gives coverage rate
            )
    
    # Data cleaning
    pop_data = np.where(pop_data >= 0, pop_data, 0)
    flood_coverage = np.clip(flood_coverage, 0, 1)
    
    # Valid data mask
    valid_mask = pop_data > 0
    
    # Calculate weighted exposure
    # Formula: Exposed population = Community population × Flood coverage rate
    exposure_weighted = pop_data * flood_coverage
    
    # Statistics
    total_population = np.sum(pop_data[valid_mask])
    exposed_population = np.sum(exposure_weighted[valid_mask])
    exposure_percentage = (exposed_population / total_population * 100) if total_population > 0 else 0
    
    # Calculate flooded area
    pixel_area = abs(pop_transform[0] * pop_transform[4])  # m²
    # Total flooded area considering coverage rate
    flooded_area_m2 = np.sum(flood_coverage[valid_mask]) * pixel_area
    flooded_area_km2 = flooded_area_m2 / 1_000_000
    
    print(f"\n{'='*60}")
    print(f"FLOOD EXPOSURE ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"Analysis Resolution: {pop_transform[0]:.0f}m × {abs(pop_transform[4]):.0f}m")
    print(f"Community Unit Area: {pixel_area:,.0f} m² (1 hectare)")
    print(f"\nFlooded Area:")
    print(f"  Total flooded area: {flooded_area_m2:,.2f} m² ({flooded_area_km2:.4f} km²)")
    print(f"\nPopulation:")
    print(f"  Total population: {total_population:,.0f}")
    print(f"  Exposed population: {exposed_population:,.0f}")
    print(f"  Exposure percentage: {exposure_percentage:.2f}%")
    
    # Coverage rate classification
    print(f"\n{'='*60}")
    print("COMMUNITY UNITS BY FLOOD COVERAGE")
    print(f"{'='*60}")
    
    coverage_bins = [0, 0.01, 0.25, 0.5, 0.75, 1.0]
    coverage_labels = ['No Flood (0%)', 'Minor (1-25%)', 'Moderate (25-50%)', 
                       'Severe (50-75%)', 'Extreme (75-100%)']
    
    coverage_stats = []
    for i in range(len(coverage_bins) - 1):
        lower = coverage_bins[i]
        upper = coverage_bins[i + 1]
        
        if i == 0:
            mask = (flood_coverage == 0) & valid_mask
        else:
            mask = (flood_coverage > lower) & (flood_coverage <= upper) & valid_mask
        
        unit_count = np.sum(mask)
        unit_population = np.sum(pop_data[mask])
        unit_exposed = np.sum(exposure_weighted[mask])
        avg_coverage = np.mean(flood_coverage[mask]) * 100 if unit_count > 0 else 0
        
        coverage_stats.append({
            'category': coverage_labels[i],
            'coverage_range': f"{lower*100:.0f}-{upper*100:.0f}%",
            'community_units': unit_count,
            'total_population': unit_population,
            'exposed_population': unit_exposed,
            'avg_coverage': avg_coverage
        })
        
        print(f"\n{coverage_labels[i]}:")
        print(f"  Units: {unit_count:,}")
        print(f"  Population: {unit_population:,.0f}")
        print(f"  Exposed: {unit_exposed:,.0f}")
        if unit_count > 0:
            print(f"  Avg coverage: {avg_coverage:.1f}%")
    
    print(f"\n{'='*60}")
    
    # Save outputs
    print(f"\nSaving results...")
    
    # 1. Coverage rate raster
    coverage_raster = output_dir / "flood_coverage_rate.tif"
    pop_meta = {
        'driver': 'GTiff',
        'height': pop_data.shape[0],
        'width': pop_data.shape[1],
        'count': 1,
        'dtype': rasterio.float32,
        'crs': pop_crs,
        'transform': pop_transform,
        'nodata': -9999,
        'compress': 'lzw'
    }
    
    with rasterio.open(coverage_raster, 'w', **pop_meta) as dst:
        dst.write(flood_coverage.astype(np.float32), 1)
    print(f"✓ Coverage rate raster: {coverage_raster}")
    
    # 2. Weighted exposure raster
    exposure_raster = output_dir / "population_flood_exposure.tif"
    with rasterio.open(exposure_raster, 'w', **pop_meta) as dst:
        dst.write(exposure_weighted.astype(np.float32), 1)
    print(f"✓ Exposure raster: {exposure_raster}")
    
    # Save summary statistics
    stats_file = output_dir / "exposure_statistics.csv"
    with open(stats_file, 'w') as f:
        f.write("Metric,Value,Unit\n")
        f.write(f"Analysis Method,Coverage Rate Weighted,method\n")
        f.write(f"Community Unit Size,{pop_transform[0]:.0f},meters\n")
        f.write(f"Pixel Resolution X,{pop_transform[0]:.2f},meters\n")
        f.write(f"Pixel Resolution Y,{abs(pop_transform[4]):.2f},meters\n")
        f.write(f"Pixel Area,{pixel_area:.2f},square meters\n")
        f.write(f"Community Units Analyzed,{np.sum(valid_mask)},count\n")
        f.write(f"Units with Any Flood,{np.sum((flood_coverage > 0) & valid_mask)},count\n")
        f.write(f"Flooded Area,{flooded_area_m2:.2f},square meters\n")
        f.write(f"Flooded Area,{flooded_area_km2:.4f},square kilometers\n")
        f.write(f"Total Population,{total_population:.0f},people\n")
        f.write(f"Exposed Population,{exposed_population:.0f},people\n")
        f.write(f"Exposure Percentage,{exposure_percentage:.2f},percent\n")
    
    print(f"✓ Statistics saved to: {stats_file}")
    
    # Save coverage classification statistics
    coverage_stats_file = output_dir / "flood_coverage_statistics.csv"
    pd.DataFrame(coverage_stats).to_csv(coverage_stats_file, index=False)
    print(f"✓ Coverage statistics: {coverage_stats_file}")
    
    # Create visualizations
    create_visualization(
        pop_data=pop_data,
        flood_coverage=flood_coverage,
        exposure_weighted=exposure_weighted,
        pop_transform=pop_transform,
        pop_crs=pop_crs,
        coverage_stats=coverage_stats,
        total_population=total_population,
        exposed_population=exposed_population,
        output_dir=output_dir
    )
    
    return exposure_raster, {
        'pixel_size_m': pop_transform[0],
        'pixel_area_m2': pixel_area,
        'flooded_area_m2': flooded_area_m2,
        'flooded_area_km2': flooded_area_km2,
        'total_population': total_population,
        'exposed_population': exposed_population,
        'exposure_percentage': exposure_percentage,
        'coverage_stats': coverage_stats
    }


if __name__ == "__main__":
    calculate_exposure()