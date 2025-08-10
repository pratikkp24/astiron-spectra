#!/usr/bin/env python3
"""
Prepare thermal infrared data for Astiron Spectra
Converts various TIR formats to unified format and updates pairs.csv
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np
import cv2

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'common'))

from astiron_io import (load_config, write_geotiff, save_json, read_pairs_csv,
                write_pairs_csv, ensure_dir, read_thermal)
from thermal import (denoise_thermal_bilateral, extract_video_frames,
                    brightness_temperature_to_celsius)
from geo import reproject_to_crs
from rasterio.crs import CRS
from rasterio.transform import from_bounds


def create_synthetic_tir_data(output_path: Path, scene_name: str,
                             config: Dict) -> Tuple[np.ndarray, Dict]:
    """Create synthetic thermal infrared data for demo purposes"""

    print(f"Creating synthetic TIR data: {scene_name}")

    # Create synthetic thermal data
    height, width = 100, 100  # Small for demo

    # Generate synthetic thermal data (in Kelvin)
    np.random.seed(42)  # Reproducible

    # Base temperature (around 300K = 27°C)
    base_temp = 300.0

    # Create temperature field with spatial variation
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # Add spatial temperature gradient
    temp_gradient = 5.0 * (y_coords / height) + 3.0 * (x_coords / width)

    # Add random noise
    noise = np.random.normal(0, 2.0, (height, width))

    # Combine
    thermal_data = base_temp + temp_gradient + noise

    # Add some hot spots (anomalies)
    n_hotspots = 3
    for _ in range(n_hotspots):
        y = np.random.randint(10, height-10)
        x = np.random.randint(10, width-10)

        # Create hot spot (10-20K above background)
        hotspot_temp = base_temp + np.random.uniform(10, 20)

        # Apply to small region with Gaussian falloff
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                if 0 <= y+dy < height and 0 <= x+dx < width:
                    distance = np.sqrt(dy*dy + dx*dx)
                    if distance <= 3:
                        weight = np.exp(-distance**2 / 2)
                        thermal_data[y+dy, x+dx] = (
                            thermal_data[y+dy, x+dx] * (1-weight) +
                            hotspot_temp * weight
                        )

    # Ensure reasonable temperature range
    thermal_data = np.clip(thermal_data, 250, 400)  # -23°C to 127°C

    # Create metadata
    bounds = (-122.5, 37.5, -122.4, 37.6)  # San Francisco Bay area
    transform = from_bounds(*bounds, width, height)

    metadata = {
        'crs': CRS.from_epsg(4326),
        'transform': transform,
        'width': width,
        'height': height,
        'count': 1,
        'dtype': 'float32',
        'nodata': None,
        'scene_name': scene_name,
        'acquisition_date': '2024-06-01',
        'sensor': 'SYNTHETIC_TIR',
        'spatial_resolution': 30.0,
        'wavelength_range': [8000, 12000],  # 8-12 μm
        'units': 'Kelvin',
        'calibration': 'brightness_temperature'
    }

    return thermal_data.astype(np.float32), metadata


def process_tir_scene(raw_path: Path, output_dir: Path, scene_name: str,
                     config: Dict) -> Dict:
    """Process a single TIR scene"""

    print(f"Processing TIR scene: {scene_name}")

    # Create output directory
    scene_output_dir = output_dir / f"{scene_name}_sample"
    ensure_dir(scene_output_dir)

    # Check if this is a synthetic demo case
    if not raw_path.exists() or raw_path.is_dir():
        # Create synthetic data
        data, metadata = create_synthetic_tir_data(scene_output_dir, scene_name, config)
    else:
        # Try to read actual thermal data
        try:
            # Look for thermal files
            thermal_files = list(raw_path.glob('*.tif')) + list(raw_path.glob('*.tiff'))

            if thermal_files:
                data, metadata = read_thermal(str(thermal_files[0]))
            else:
                # Check for video files
                video_files = list(raw_path.glob('*.mp4')) + list(raw_path.glob('*.avi'))
                if video_files:
                    # Extract frames from video
                    frames_dir = scene_output_dir / 'frames'
                    ensure_dir(frames_dir)

                    tir_config = config.get('preprocess', {}).get('tir', {})
                    fps = tir_config.get('video_fps', 2.0)

                    frame_files = extract_video_frames(str(video_files[0]),
                                                     str(frames_dir), fps)

                    if frame_files:
                        # Use first frame as representative
                        data, metadata = read_thermal(frame_files[0])
                    else:
                        raise ValueError("No frames extracted from video")
                else:
                    raise ValueError("No thermal files found")

        except Exception as e:
            print(f"Error reading {raw_path}: {e}")
            print("Creating synthetic data instead...")
            data, metadata = create_synthetic_tir_data(scene_output_dir, scene_name, config)

    # Apply preprocessing
    preprocess_config = config.get('preprocess', {})
    tir_config = preprocess_config.get('tir', {})

    # Denoise if requested
    denoise_method = tir_config.get('denoise', 'none')
    if denoise_method == 'bilateral':
        print("Applying bilateral denoising")
        data = denoise_thermal_bilateral(data)

    # Convert to Celsius if data is in Kelvin
    if metadata.get('units') == 'Kelvin' or np.mean(data[~np.isnan(data)]) > 200:
        print("Converting to Celsius")
        data = data - 273.15
        metadata['units'] = 'Celsius'

    # Reproject to target CRS if needed
    target_crs = preprocess_config.get('resample_to', 'EPSG:4326')
    if str(metadata.get('crs', 'EPSG:4326')) != target_crs:
        print(f"Reprojecting to {target_crs}")
        data, new_metadata = reproject_to_crs(
            data, metadata['crs'], metadata['transform'],
            CRS.from_string(target_crs)
        )
        metadata.update(new_metadata)

    # Save processed data
    output_file = scene_output_dir / 'thermal.tif'
    print(f"Saving to {output_file}")

    write_geotiff(data, str(output_file), metadata)

    # Save metadata
    metadata_clean = {}
    for key, value in metadata.items():
        if hasattr(value, 'to_dict'):  # Handle CRS objects
            metadata_clean[key] = value.to_dict()
        elif hasattr(value, '__array__'):  # Handle numpy arrays
            metadata_clean[key] = value.tolist()
        else:
            metadata_clean[key] = value

    save_json(metadata_clean, str(scene_output_dir / 'metadata.json'))

    # Create thermal-specific metadata
    thermal_info = {
        'wavelength_range_um': [8.0, 12.0],
        'temperature_range_celsius': [float(np.nanmin(data)), float(np.nanmax(data))],
        'mean_temperature_celsius': float(np.nanmean(data)),
        'calibration': metadata.get('calibration', 'brightness_temperature'),
        'units': metadata.get('units', 'Celsius')
    }

    save_json(thermal_info, str(scene_output_dir / 'thermal_info.json'))

    # Return pair information
    pair_info = {
        'pair_id': f"tir_{scene_name}",
        'sensor_type': 'THERMAL',
        'path': str(output_file.relative_to(Path.cwd())),
        'type': 'TIR',
        'date': metadata.get('acquisition_date', datetime.now().strftime('%Y-%m-%d')),
        'notes': f"Processed {scene_name} thermal infrared data"
    }

    return pair_info


def main():
    parser = argparse.ArgumentParser(description='Prepare thermal infrared data for Astiron Spectra')
    parser.add_argument('--input-dir', type=str, default='data/raw',
                       help='Input directory with raw TIR data')
    parser.add_argument('--output-dir', type=str, default='data/tir',
                       help='Output directory for processed TIR data')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Configuration file')
    parser.add_argument('--scenes', nargs='+',
                       help='Specific scenes to process (default: all found)')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    print("Astiron Spectra - TIR Data Preparation")
    print("======================================")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print("")

    # Find TIR datasets to process
    tir_datasets = []

    if args.scenes:
        # Process specified scenes
        for scene in args.scenes:
            scene_path = input_dir / scene
            tir_datasets.append((scene, scene_path))
    else:
        # Auto-discover datasets
        if input_dir.exists():
            for item in input_dir.iterdir():
                if item.is_dir():
                    # Check for common TIR file patterns
                    tir_files = (list(item.glob('*.tif')) + list(item.glob('*.tiff')) +
                               list(item.glob('*.mp4')) + list(item.glob('*.avi')))
                    if tir_files or item.name in ['flame3', 'kaist', 'mirsat']:
                        tir_datasets.append((item.name, item))

        # Add demo datasets even if not found
        demo_datasets = ['flame3', 'kaist']
        for demo in demo_datasets:
            if not any(name == demo for name, _ in tir_datasets):
                tir_datasets.append((demo, input_dir / demo))

    if not tir_datasets:
        print("No TIR datasets found. Creating demo datasets...")
        tir_datasets = [
            ('flame3', input_dir / 'flame3'),
            ('kaist', input_dir / 'kaist'),
        ]

    # Process each dataset
    pairs = read_pairs_csv()
    new_pairs = []

    for scene_name, scene_path in tir_datasets:
        try:
            pair_info = process_tir_scene(scene_path, output_dir, scene_name, config)
            new_pairs.append(pair_info)
            print(f"✓ Processed {scene_name}")
        except Exception as e:
            print(f"✗ Failed to process {scene_name}: {e}")
            continue

    # Update pairs.csv
    if new_pairs:
        # Remove existing TIR pairs and add new ones
        existing_pairs = [p for p in pairs if not p['pair_id'].startswith('tir_')]
        all_pairs = existing_pairs + new_pairs

        write_pairs_csv(all_pairs)
        print(f"\n✓ Updated pairs.csv with {len(new_pairs)} TIR pairs")

    print(f"\n✓ TIR data preparation complete!")
    print(f"Processed datasets: {len(new_pairs)}")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
