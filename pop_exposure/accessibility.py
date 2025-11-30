"""
Calculate flood risk index using adapted G2SFCA method
Raster-based implementation for compound flooding scenarios
Uses euclidean distance for planar spreading in flat coastal areas
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import distance_transform_edt, gaussian_filter
from shapely.geometry import Point
import gc

# Set non-interactive backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import matplotlib.patches as mpatches
from paths import DATA_DIR, CITY_BOUNDARY, ensure_dir

# Nature journal style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def calculate_distance_raster(flood_mask, pixel_size):
    """
    Calculate Euclidean distance from each pixel to nearest flood pixel
    
    Args:
        flood_mask: Binary array (1=flood, 0=no flood)
        pixel_size: Size of each pixel in meters (assumed square pixels)
    
    Returns:
        Distance array in meters
    """
    # Distance transform gives distance in pixels
    distance_pixels = distance_transform_edt(~flood_mask.astype(bool))
    
    # Convert to meters
    return distance_pixels * pixel_size


def calculate_g2sfca_raster(pop_raster, flood_raster, bandwidth=500, pixel_size=100):
    """
    Calculate G2SFCA flood risk using efficient Gaussian filtering
    
    This vectorized implementation avoids O(N^2) loops by using:
    1. Gaussian filtering for weighted population summation
    2. Convolution for risk score calculation
    
    Args:
        pop_raster: Population density array (people per hectare)
        flood_raster: Binary flood extent array (1=flooded, 0=not flooded)
        bandwidth: Search radius in meters (Gaussian kernel standard deviation)
        pixel_size: Raster pixel size in meters (default 100m)
    
    Returns:
        risk_raster: Flood risk score for each pixel
        distance_raster: Distance to nearest flood pixel (for visualization)
    """
    print(f"\nCalculating optimized G2SFCA (bandwidth={bandwidth}m)...")
    print("  Using vectorized Gaussian filtering approach")
    
    # Convert bandwidth to pixels for Gaussian filter
    sigma_pixels = bandwidth / pixel_size
    print(f"  Gaussian sigma: {sigma_pixels:.2f} pixels")
    
    # Step 1: Calculate weighted population sum around each pixel
    print("  Step 1: Computing weighted population density...")
    # Gaussian filter applies normalized kernel - gives weighted average
    weighted_pop_avg = gaussian_filter(
        pop_raster, 
        sigma=sigma_pixels,
        mode='reflect',  # Better boundary handling than 'constant'
        cval=0.0
    )
    
    # Step 2: Calculate flood supply ratios (R_j)
    print("  Step 2: Calculating flood supply ratios...")
    # Initialize ratio array
    R = np.zeros_like(pop_raster, dtype=np.float32)
    
    # Valid flood pixels with positive population influence
    valid_flood = (flood_raster == 1) & (weighted_pop_avg > 0)
    valid_count = np.sum(valid_flood)
    print(f"    Processing {valid_count} valid flood pixels")
    
    if valid_count > 0:
        # Pixel area in square meters (100m x 100m = 10,000 m²)
        pixel_area = pixel_size * pixel_size
        
        # Calculate supply ratio: area / weighted population
        R[valid_flood] = pixel_area / weighted_pop_avg[valid_flood]
    
    # Step 3: Calculate risk scores via convolution
    print("  Step 3: Computing risk scores via convolution...")
    risk_raster = gaussian_filter(
        R,
        sigma=sigma_pixels,
        mode='reflect',
        cval=0.0
    )
    
    # Step 4: Calculate distance to nearest flood for visualization
    print("  Step 4: Generating distance raster...")
    distance_raster = calculate_distance_raster(flood_raster, pixel_size)
    
    # Mask risk scores to populated areas only
    print("  Masking risk scores to populated areas...")
    risk_raster = np.where(pop_raster > 0, risk_raster, 0.0)
    
    print("  ✓ G2SFCA calculation complete")
    return risk_raster, distance_raster


def load_and_align_data(pop_raster_path, flood_raster_path):
    """
    Load and align population and flood rasters
    
    Returns:
        pop_data, flood_data, transform, crs, meta
    """
    print("Loading and aligning raster data...")
    
    # Load population raster
    with rasterio.open(pop_raster_path) as src:
        pop_data = src.read(1)
        pop_transform = src.transform
        pop_crs = src.crs
        pop_shape = src.shape
        pop_meta = src.meta.copy()
        
        print(f"  Population raster: {pop_shape}, CRS: {pop_crs}")
    
    # Load flood raster
    with rasterio.open(flood_raster_path) as src:
        flood_crs = src.crs
        
        # Reproject flood to match population grid
        flood_data = np.empty(pop_shape, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=flood_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=pop_transform,
            dst_crs=pop_crs,
            resampling=Resampling.nearest
        )
        
        print(f"  Flood raster reprojected to match population grid")
    
    # Clean data
    pop_data = np.where(pop_data >= 0, pop_data, 0)
    flood_data = np.where(flood_data == 1, 1, 0).astype(np.uint8)
    
    print(f"  Valid population pixels: {np.sum(pop_data > 0)}")
    print(f"  Flooded pixels: {np.sum(flood_data == 1)}")
    
    return pop_data, flood_data, pop_transform, pop_crs, pop_meta


def create_visualization(pop_data, flood_data, risk_raster, distance_raster,
                         transform, crs, bandwidth, output_dir):
    """Create visualizations with proper macOS resource cleanup"""
    print(f"\nCreating visualizations...")
    
    try:
        boundary = gpd.read_file(CITY_BOUNDARY).to_crs(crs)
    except Exception as e:
        print(f"  Warning: Could not load boundary: {e}")
        boundary = None
    
    # Calculate extent
    height, width = risk_raster.shape
    left = transform[2]
    top = transform[5]
    right = left + width * transform[0]
    bottom = top + height * transform[4]
    extent = [left, right, bottom, top]
    
    # Figure 1: Four-panel overview
    fig1_success = False
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        
        # Panel A: Population (top-left)
        ax1 = axes[0, 0]
        pop_masked = np.ma.masked_where(pop_data <= 0, pop_data)
        im1 = ax1.imshow(pop_masked, cmap='YlOrRd', extent=extent, 
                         vmin=0, vmax=np.percentile(pop_masked.compressed(), 98))
        if boundary is not None:
            boundary.boundary.plot(ax=ax1, color='black', linewidth=1)
        ax1.set_title('A. Population Density', fontweight='bold', loc='left')
        ax1.set_xlabel('Easting (m)')
        ax1.set_ylabel('Northing (m)')
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('People/hectare', rotation=270, labelpad=15)
        ax1.ticklabel_format(style='sci', scilimits=(0,0))
        
        # Panel B: Flood extent (top-right)
        ax2 = axes[0, 1]
        ax2.imshow(pop_masked, cmap='gray', alpha=0.2, extent=extent)
        flood_masked = np.ma.masked_where(flood_data == 0, flood_data)
        ax2.imshow(flood_masked, cmap='Blues', alpha=0.7, extent=extent)
        if boundary is not None:
            boundary.boundary.plot(ax=ax2, color='black', linewidth=1)
        ax2.set_title('B. Flood Extent', fontweight='bold', loc='left')
        ax2.set_xlabel('Easting (m)')
        ax2.set_ylabel('Northing (m)')
        ax2.ticklabel_format(style='sci', scilimits=(0,0))
        
        # Panel C: Distance to flood (bottom-left)
        ax3 = axes[1, 0]
        dist_masked = np.ma.masked_where(pop_data <= 0, distance_raster)
        im3 = ax3.imshow(dist_masked, cmap='RdYlGn', extent=extent,
                         vmin=0, vmax=bandwidth)
        if boundary is not None:
            boundary.boundary.plot(ax=ax3, color='black', linewidth=1)
        ax3.set_title('C. Distance to Flood', fontweight='bold', loc='left')
        ax3.set_xlabel('Easting (m)')
        ax3.set_ylabel('Northing (m)')
        cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        cbar3.set_label('Distance (m)', rotation=270, labelpad=15)
        ax3.ticklabel_format(style='sci', scilimits=(0,0))
        
        # Panel D: G2SFCA Risk (bottom-right)
        ax4 = axes[1, 1]
        risk_masked = np.ma.masked_where(risk_raster <= 0, risk_raster)
        if len(risk_masked.compressed()) > 0:
            im4 = ax4.imshow(risk_masked, cmap='RdPu', extent=extent,
                            vmin=0, vmax=np.percentile(risk_masked.compressed(), 98))
            if boundary is not None:
                boundary.boundary.plot(ax=ax4, color='black', linewidth=1)
            ax4.set_title('D. G2SFCA Risk Index', fontweight='bold', loc='left')
            ax4.set_xlabel('Easting (m)')
            ax4.set_ylabel('Northing (m)')
            cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
            cbar4.set_label('Risk Score', rotation=270, labelpad=15)
            ax4.ticklabel_format(style='sci', scilimits=(0,0))
        
        plt.suptitle(f'G2SFCA Flood Risk Analysis - Fort Myers (Hurricane Helene 2024)\n'
                     f'Raster-based, Euclidean Distance, Bandwidth={bandwidth}m',
                     fontsize=13, fontweight='bold', y=0.98)
        
        fig1_path = output_dir / f"Fig1_g2sfca_raster_{bandwidth}m.png"
        
        # Force rendering and save
        fig.canvas.draw()
        plt.savefig(fig1_path, dpi=300, bbox_inches='tight', format='png')
        
        # Critical: Close figure and clear all references
        plt.close(fig)
        del fig, axes, ax1, ax2, ax3, ax4
        if 'im1' in locals(): del im1
        if 'im3' in locals(): del im3
        if 'im4' in locals(): del im4
        if 'cbar1' in locals(): del cbar1
        if 'cbar3' in locals(): del cbar3
        if 'cbar4' in locals(): del cbar4
        
        plt.close('all')
        gc.collect()  # Force garbage collection
        
        print(f"  ✓ Saved: {fig1_path}")
        fig1_success = True
        
    except Exception as e:
        print(f"  ⚠ Error creating Fig1: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')
        gc.collect()
    
    # Figure 2: Risk classification
    fig2_success = False
    try:
        risk_positive = risk_raster[risk_raster > 0]
        if len(risk_positive) > 0:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            p33 = np.percentile(risk_positive, 33)
            p66 = np.percentile(risk_positive, 66)
            
            risk_zones = np.zeros_like(risk_raster, dtype=np.int8)
            risk_zones[risk_raster <= 0] = 0
            risk_zones[(risk_raster > 0) & (risk_raster <= p33)] = 1
            risk_zones[(risk_raster > p33) & (risk_raster <= p66)] = 2
            risk_zones[risk_raster > p66] = 3
            
            risk_zones_masked = np.ma.masked_where(pop_data <= 0, risk_zones)
            
            colors = ['#f7f7f7', '#fc9272', '#de2d26', '#a50f15']
            cmap = LinearSegmentedColormap.from_list('risk', colors, N=4)
            norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
            
            im = ax.imshow(risk_zones_masked, cmap=cmap, norm=norm, extent=extent)
            
            if boundary is not None:
                boundary.boundary.plot(ax=ax, color='black', linewidth=1.5)
            
            legend = [
                mpatches.Patch(facecolor=c, edgecolor='black', linewidth=0.5, 
                              label=l) for c, l in zip(colors, 
                              ['No Risk', 'Low Risk', 'Medium Risk', 'High Risk'])
            ]
            ax.legend(handles=legend, loc='upper right', title='Risk Category',
                     framealpha=0.95, edgecolor='black')
            
            ax.set_title(f'Risk Classification (G2SFCA)\nBandwidth={bandwidth}m',
                        fontweight='bold', pad=15)
            ax.set_xlabel('Easting (m)')
            ax.set_ylabel('Northing (m)')
            ax.ticklabel_format(style='sci', scilimits=(0,0))
            
            # Add scale bar
            from matplotlib.patches import Rectangle
            scale_length = 1000
            scale_x = extent[0] + (extent[1] - extent[0]) * 0.05
            scale_y = extent[2] + (extent[3] - extent[2]) * 0.05
            scale_bar = Rectangle((scale_x, scale_y), scale_length, 50, 
                                  facecolor='black', edgecolor='white', linewidth=1)
            ax.add_patch(scale_bar)
            ax.text(scale_x + scale_length/2, scale_y + 200, '1 km', 
                    ha='center', va='bottom', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            fig2_path = output_dir / f"Fig2_g2sfca_raster_class_{bandwidth}m.png"
            
            # Force rendering and save
            fig.canvas.draw()
            plt.savefig(fig2_path, dpi=300, bbox_inches='tight', format='png')
            
            # Critical: Close figure and clear all references
            plt.close(fig)
            del fig, ax, im, scale_bar, legend
            if 'cmap' in locals(): del cmap
            if 'norm' in locals(): del norm
            
            plt.close('all')
            gc.collect()  # Force garbage collection
            
            print(f"  ✓ Saved: {fig2_path}")
            fig2_success = True
        else:
            print(f"  ⚠ No positive risk values, skipping Fig2")
    except Exception as e:
        print(f"  ⚠ Error creating Fig2: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')
        gc.collect()
    
    if fig1_success and fig2_success:
        print(f"  ✓ All visualizations complete")
    elif fig1_success or fig2_success:
        print(f"  ⚠ Partial visualization success")
    else:
        print(f"  ✗ Visualization failed")


def run_analysis(bandwidth=500):
    """Run G2SFCA analysis"""
    pop_raster_path = DATA_DIR / "pop" / "fort_myers_worldpop.tif"
    flood_raster_path = DATA_DIR / "flood" / "FortMyersHelene_2024T269_flood_clipped.tif"
    output_dir = ensure_dir(DATA_DIR / "pop_exposure")
    
    print(f"{'='*70}")
    print(f"G2SFCA FLOOD RISK ANALYSIS (OPTIMIZED)")
    print(f"{'='*70}")
    print(f"Distance: Euclidean (planar spreading)")
    print(f"Bandwidth: {bandwidth}m")
    print(f"Method: Vectorized Gaussian filtering")
    print(f"{'='*70}\n")
    
    try:
        # Load and align data
        pop_data, flood_data, transform, crs, meta = load_and_align_data(
            pop_raster_path, flood_raster_path
        )
        
        # Calculate G2SFCA risk scores (optimized version)
        risk_raster, distance_raster = calculate_g2sfca_raster(
            pop_data, flood_data, bandwidth, pixel_size=100
        )
        
        # Save risk raster
        raster_file = output_dir / f"flood_risk_g2sfca_raster_{bandwidth}m.tif"
        meta.update(dtype=rasterio.float32, nodata=-9999, compress='lzw')
        
        with rasterio.open(raster_file, 'w', **meta) as dst:
            dst.write(risk_raster.astype(np.float32), 1)
        
        print(f"\n✓ Risk raster saved: {raster_file}")
        
        # Save distance raster
        dist_file = output_dir / f"flood_distance_raster_{bandwidth}m.tif"
        with rasterio.open(dist_file, 'w', **meta) as dst:
            dst.write(distance_raster.astype(np.float32), 1)
        
        print(f"✓ Distance raster saved: {dist_file}")
        
        # Calculate statistics
        pop_valid = pop_data > 0
        risk_positive = risk_raster > 0
        
        # Classify risk
        risk_values = risk_raster[pop_valid & risk_positive]
        if len(risk_values) > 0:
            p33 = np.percentile(risk_values, 33)
            p66 = np.percentile(risk_values, 66)
            
            low_risk = (risk_raster > 0) & (risk_raster <= p33) & pop_valid
            med_risk = (risk_raster > p33) & (risk_raster <= p66) & pop_valid
            high_risk = (risk_raster > p66) & pop_valid
            
            stats_data = {
                'risk_category': ['Low', 'Medium', 'High'],
                'pixel_count': [
                    np.sum(low_risk),
                    np.sum(med_risk),
                    np.sum(high_risk)
                ],
                'total_population': [
                    np.sum(pop_data[low_risk]),
                    np.sum(pop_data[med_risk]),
                    np.sum(pop_data[high_risk])
                ],
                'mean_risk_score': [
                    np.mean(risk_raster[low_risk]),
                    np.mean(risk_raster[med_risk]),
                    np.mean(risk_raster[high_risk])
                ],
                'mean_distance_m': [
                    np.mean(distance_raster[low_risk]),
                    np.mean(distance_raster[med_risk]),
                    np.mean(distance_raster[high_risk])
                ]
            }
            
            stats_df = pd.DataFrame(stats_data)
            stats_file = output_dir / f"flood_risk_g2sfca_raster_{bandwidth}m_summary.csv"
            stats_df.to_csv(stats_file, index=False)
            print(f"✓ Statistics saved: {stats_file}")
        
        # Create visualizations
        try:
            create_visualization(
                pop_data, flood_data, risk_raster, distance_raster,
                transform, crs, bandwidth, output_dir
            )
        except Exception as e:
            print(f"⚠ Error in visualization (continuing anyway): {e}")
            import traceback
            traceback.print_exc()
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}")
        total_pop = np.sum(pop_data[pop_valid])
        
        if len(risk_values) > 0:
            for _, row in stats_df.iterrows():
                cat = row['risk_category']
                pop = row['total_population']
                pct = pop / total_pop * 100
                dist = row['mean_distance_m']
                print(f"  {cat:8s}: {pop:10,.0f} people ({pct:5.1f}%) | "
                      f"Avg dist to flood: {dist:6.1f}m")
        
        print(f"\nDistance Statistics:")
        valid_distances = distance_raster[pop_valid]
        print(f"  Min: {valid_distances.min():.1f}m")
        print(f"  Max: {valid_distances.max():.1f}m")
        print(f"  Mean: {valid_distances.mean():.1f}m")
        print(f"  Median: {np.median(valid_distances):.1f}m")
        
        print(f"{'='*70}\n")
        print(f"✓ Analysis for bandwidth={bandwidth}m COMPLETE\n")
        
        return risk_raster, distance_raster
        
    except Exception as e:
        print(f"✗ ERROR in analysis for bandwidth={bandwidth}m: {e}")
        import traceback
        traceback.print_exc()
        print(f"Continuing to next bandwidth...\n")
        return None, None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='G2SFCA Flood Risk Analysis')
    parser.add_argument('--bandwidth', type=int, default=500,
                       help='Search radius in meters (default: 500)')
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"STARTING G2SFCA ANALYSIS")
    print(f"Bandwidth: {args.bandwidth}m")
    print(f"{'='*70}\n")
    
    try:
        run_analysis(bandwidth=args.bandwidth)
        print(f"\n{'='*70}")
        print(f"✓ ANALYSIS COMPLETE FOR BANDWIDTH={args.bandwidth}m")
        print(f"{'='*70}\n")
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"✗ ANALYSIS FAILED FOR BANDWIDTH={args.bandwidth}m")
        print(f"Error: {e}")
        print(f"{'='*70}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Final cleanup
        plt.close('all')
        gc.collect()