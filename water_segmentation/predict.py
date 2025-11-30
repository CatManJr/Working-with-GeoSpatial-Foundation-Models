"""
Flood Segmentation Prediction Workflow
Uses Prithvi model via HuggingFace Gradio Space API

Since terratorch doesn't support macOS, we use the Gradio Space directly.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
import rasterio
from rasterio.merge import merge as rio_merge
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import IBM_TILES_DIR, IBM_PREDICT_DIR, PRITHVI_PREDICTION, get_hf_token


class GradioSpaceClient:
    """Client to call Prithvi flood model via Gradio Space"""
    
    def __init__(self, api_token: Optional[str] = None, space_url: Optional[str] = None):
        """Initialize Gradio client"""
        self.api_token = api_token or os.getenv("HF_TOKEN")
        # Use IBM's official Space (they have model access)
        self.space_url = space_url or "ibm-nasa-geospatial/Prithvi-EO-2.0-Sen1Floods11-demo"
        
        print(f"Connecting to Gradio Space: {self.space_url}")
        
        try:
            from gradio_client import Client
            
            # Connect to Space
            self.client = Client(self.space_url)
            print("✓ Connected successfully")
            
        except Exception as e:
            print(f"\n{'=' * 80}")
            print("ERROR: Failed to connect to Gradio Space")
            print(f"{'=' * 80}")
            print(f"Error: {e}")
            print(f"\nMake sure gradio_client is installed:")
            print("  uv add gradio-client")
            print(f"{'=' * 80}")
            sys.exit(1)
    
    def predict(self, image_path: str) -> np.ndarray:
        """
        Call Gradio Space to predict flood on image
        
        Args:
            image_path: Path to input GeoTIFF
            
        Returns:
            Prediction mask (H, W)
        """
        from gradio_client import handle_file
        
        try:
            # Call the prediction API
            # The API name is "/partial" not "/predict"
            # Returns [rgb_orig, pred_rgb, overlay] - 3 image outputs
            result = self.client.predict(
                data_file=handle_file(image_path),
                api_name="/partial"
            )
            
            # Result is a tuple of 3 outputs: (value_12, value_13, value_14)
            # value_12 = original RGB image
            # value_13 = prediction RGB (white=flood, black=no flood)
            # value_14 = overlay image
            
            if isinstance(result, (tuple, list)) and len(result) >= 2:
                pred_output = result[1]  # Prediction image (second output)
            else:
                raise Exception(f"Unexpected result format: {type(result)}")
            
            # pred_output is a dict with 'path' key
            if isinstance(pred_output, dict) and 'path' in pred_output:
                pred_path = pred_output['path']
            elif isinstance(pred_output, str):
                pred_path = pred_output
            else:
                raise Exception(f"Unexpected output format: {type(pred_output)}")
            
            # Read the prediction image
            # It's an RGB image where white=flood, black=no flood
            from PIL import Image
            img = Image.open(pred_path).convert('L')  # Convert to grayscale
            pred_array = np.array(img)
            
            # Convert to binary mask (>128 = flood)
            pred_mask = (pred_array > 128).astype(np.uint8)
            
            return pred_mask
            
        except Exception as e:
            print(f"\n{'=' * 80}")
            print("ERROR: Prediction failed")
            print(f"{'=' * 80}")
            print(f"File: {image_path}")
            print(f"Error: {e}")
            print(f"{'=' * 80}")
            raise


class FloodSegmentationWorkflow:
    """Flood Segmentation Workflow using Gradio Space"""
    
    def __init__(
        self,
        tiles_dir: str = "IBM/tiles_2024T269",
        output_dir: str = "IBM/predict",
        api_token: Optional[str] = None,
    ):
        self.tiles_dir = Path(tiles_dir)
        self.output_dir = Path(output_dir)
        self.api_token = api_token
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir = self.output_dir / "tmp_predictions"
        self.tmp_dir.mkdir(exist_ok=True)
        
        self.client = None
        
    def init_client(self):
        """Initialize Gradio client"""
        print("=" * 80)
        print("Initializing Gradio Space client...")
        print("=" * 80)
        self.client = GradioSpaceClient(api_token=self.api_token)
        print()
        
    def get_tile_paths(self) -> pd.DataFrame:
        """Get all tile paths"""
        tiles_csv = self.tiles_dir / "tiles.csv"
        
        if tiles_csv.exists():
            print(f"Loading tile metadata from {tiles_csv}...")
            tiles_df = pd.read_csv(tiles_csv)
            tiles_df['path'] = tiles_df.apply(
                lambda row: str(self.tiles_dir / f"FortMyersHelene_2024T269_r{row['ri']}_c{row['ci']}.tif"),
                axis=1
            )
        else:
            print("tiles.csv not found, scanning directory...")
            tile_files = sorted(self.tiles_dir.glob("*.tif"))
            tile_files = [f for f in tile_files if not f.name.endswith('.aux.xml')]
            
            tiles_data = []
            for tile_file in tile_files:
                parts = tile_file.stem.split('_')
                ri = int(parts[-2][1:])
                ci = int(parts[-1][1:])
                
                with rasterio.open(tile_file) as src:
                    tiles_data.append({
                        'ri': ri,
                        'ci': ci,
                        'path': str(tile_file),
                        'height': src.height,
                        'width': src.width,
                    })
            
            tiles_df = pd.DataFrame(tiles_data)
        
        print(f"Found {len(tiles_df)} tiles")
        print(f"  Rows: {tiles_df['ri'].max() + 1}")
        print(f"  Cols: {tiles_df['ci'].max() + 1}")
        print()
        
        return tiles_df
    
    def predict_tile(self, tile_path: str) -> Tuple[np.ndarray, dict]:
        """Predict on single tile"""
        with rasterio.open(tile_path) as src:
            meta = src.meta.copy()
        
        pred_mask = self.client.predict(tile_path)
        
        return pred_mask, meta
    
    def predict_all_tiles(self, tiles_df: pd.DataFrame) -> List[str]:
        """Batch prediction"""
        print("=" * 80)
        print(f"Starting flood segmentation on {len(tiles_df)} tiles...")
        print("=" * 80)
        print(f"Using Gradio Space API (cloud-based)")
        print()
        
        pred_paths = []
        
        for idx, row in tqdm(tiles_df.iterrows(), total=len(tiles_df), desc="Inference"):
            tile_path = row['path']
            tile_name = Path(tile_path).stem
            
            try:
                pred_mask, meta = self.predict_tile(tile_path)
                
                pred_path = self.tmp_dir / f"pred_{tile_name}.tif"
                meta.update(count=1, dtype='uint8', compress='lzw', nodata=0)
                
                with rasterio.open(pred_path, 'w', **meta) as dst:
                    dst.write(pred_mask, 1)
                
                pred_paths.append(str(pred_path))
                
            except Exception as e:
                print(f"\nError on {tile_name}: {e}")
                continue
        
        print(f"\n✓ Completed {len(pred_paths)}/{len(tiles_df)} tiles")
        print()
        
        return pred_paths
    
    def mosaic_predictions(self, pred_paths: List[str], output_path: str):
        """Mosaic predictions"""
        print("=" * 80)
        print("Mosaicking results...")
        print("=" * 80)
        
        src_datasets = [rasterio.open(p) for p in pred_paths]
        
        try:
            mosaic, out_transform = rio_merge(src_datasets, method='max', nodata=0)
            
            ref = src_datasets[0]
            out_meta = ref.meta.copy()
            out_meta.update(
                driver='GTiff',
                height=mosaic.shape[1],
                width=mosaic.shape[2],
                transform=out_transform,
                count=1,
                dtype='uint8',
                compress='lzw',
                nodata=0
            )
            
            with rasterio.open(output_path, 'w', **out_meta) as dst:
                dst.write(mosaic[0].astype(np.uint8), 1)
            
            inundated = np.sum(mosaic[0] == 1)
            total = mosaic.shape[1] * mosaic.shape[2]
            
            print(f"  Output: {mosaic.shape[1]} × {mosaic.shape[2]}")
            print(f"  Inundated: {inundated:,} pixels ({inundated/total*100:.2f}%)")
            
        finally:
            for src in src_datasets:
                src.close()
        
        print()
    
    def cleanup(self, keep_tmp: bool = False):
        """Clean up"""
        if not keep_tmp and self.tmp_dir.exists():
            import shutil
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
    
    def run(self, keep_tmp: bool = False):
        """Execute workflow"""
        print("\n" + "=" * 80)
        print("Fort Myers Flood Segmentation (via Gradio Space)")
        print("=" * 80 + "\n")
        
        self.init_client()
        tiles_df = self.get_tile_paths()
        pred_paths = self.predict_all_tiles(tiles_df)
        
        if not pred_paths:
            print("❌ No predictions generated")
            return None
        
        output_path = self.output_dir / "FortMyersHelene_2024T269_inundated.tif"
        self.mosaic_predictions(pred_paths, str(output_path))
        self.cleanup(keep_tmp=keep_tmp)
        
        print("=" * 80)
        print(f"✓ Complete! Output: {output_path}")
        print("=" * 80)
        
        return output_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles_dir", default="IBM/tiles_2024T269")
    parser.add_argument("--output_dir", default="IBM/predict")
    parser.add_argument("--api_token", default=None)
    parser.add_argument("--keep_tmp", action="store_true")
    
    args = parser.parse_args()
    
    # Try to read token
    if not args.api_token and not os.getenv("HF_TOKEN"):
        token_file = Path("hf_token.txt")
        if token_file.exists():
            args.api_token = token_file.read_text().strip()
            print(f"Loaded token from {token_file}\n")
    
    workflow = FloodSegmentationWorkflow(
        tiles_dir=args.tiles_dir,
        output_dir=args.output_dir,
        api_token=args.api_token,
    )
    
    workflow.run(keep_tmp=args.keep_tmp)


if __name__ == "__main__":
    main()
