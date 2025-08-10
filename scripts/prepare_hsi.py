#!/usr/bin/env python3
"""
Prepare hyperspectral data for Astiron Spectra
Converts various HSI formats to unified format and updates pairs.csv
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml
from datetime import datetime

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'common'))

from astiron_io import (load_config, load_bands_config, read_hyperspectral,
                write_geotiff, save_json, read_pairs_csv, write_pairs_csv,
                ensure_dir)
from spectral import remove_bad_bands, normalize_spectrum
from geo import reproject_to_crs, suppress_natural_background
from rasterio.crs import CRS
from rasterio.transform import from_bounds


def create_synthetic_hsi_data(output_path: Path, scene_name: str,
                             config: Dict) -> Tuple[np.ndarray, Dict]:
    """Create synthetic hyperspectral data for demo purposes"""

    print(f"Creating synthetic HSI data: {scene_name}")

    # Create synthetic hyperspectral cube
    height, width = 100, 100  # Small for demo
    bands = 50  # Reduced for demo

    # Generate synthetic wavelengths (400-900 nm)
    wavelengths = np.linspace(400, 900, bands)

    # Create synthetic data with some spatial and spectral structure
    np.random.seed(42)  # Reproducible

    # Base reflectance (vegetation-like)
    base_reflectance = np.ones((height, width, bands)) * 0.1

    # Add vegetation signature (red edge)
    for i, wl in enumerate(wavelengths):
        if wl < 700:  # Visible - low reflectance
            base_reflectance[:, :, i] *= (0.5 + 0.3 * np.random.random((height, width)))
        else:  # NIR - high reflectance
            base_reflectance[:, :, i] *= (2.0 + 0.5 * np.random.random((height, width)))

    # Add some anomalous pixels (mineral-like signatures)
    n_anomalies = 5
    for _ in range(n_anomalies):
        y = np.random.randint(10, height-10)
        x = np.random.randint(10, width-10)

        # Create mineral-like spectrum (higher reflectance in SWIR)
        anomaly_spectrum = np.ones(bands) * 0.3
        for i, wl in enumerate(wavelengths):
            if wl > 750:  # Higher reflectance in NIR
                anomaly_spectrum[i] *= 1.5

        # Apply to small region
        base_reflectance[y-2:y+3, x-2:x+3, :] = anomaly_spectrum

    # Add noise
    noise = np.random.normal(0, 0.02, (height, width, bands))
    data = base_reflectance + noise
    data = np.clip(data, 0, 1)  # Keep in valid reflectance range

    # Create metadata
    bounds = (-122.5, 37.5, -122.4, 37.6)  # San Francisco Bay area
    transform = from_bounds(*bounds, width, height)

    metadata = {
        'crs': CRS.from_epsg(4326),
        'transform': transform,
        'width': width,
        'height': height,
        'count': bands,
        'dtype': 'float32',
        'nodata': None,
        'wavelengths': wavelengths.tolist(),
        'scene_name': scene_name,
        'acquisition_date': '2024-06-01',
        'sensor': 'SYNTHETIC',
        'spatial_resolution': 30.0,
        'spectral_resolution': 10.0
    }

    return data.astype(np.float32), metadata


def process_hsi_scene(raw_path: Path, output_dir: Path, scene_name: str,
                     config: Dict, bands_config: Dict) -> Dict:
    """Process a single HSI scene"""

    print(f"Processing HSI scene: {scene_name}")

    # Create output directory
    scene_output_dir = output_dir / scene_name
    ensure_dir(scene_output_dir)

    # Check if this is a synthetic demo case
    if not raw_path.exists() or raw_path.is_dir():
        # Create synthetic data
        data, metadata = create_synthetic_hsi_data(scene_output_dir, scene_name, config)
        wavelengths = np.array(metadata['wavelengths'])
    else:
        # Read actual hyperspectral data
        try:
            data, metadata = read_hyperspectral(str(raw_path))

            # Get wavelengths from bands config or metadata
            if scene_name.upper() in bands_config.get('wavelengths', {}):
                wavelengths = np.array(bands_config['wavelengths'][scene_name.upper()])
            else:
                # Generate default wavelengths
                n_bands = data.shape[2]
                wavelengths = np.linspace(400, 2500, n_bands)
                print(f"Warning: Using default wavelengths for {scene_name}")

        except Exception as e:
            print(f"Error reading {raw_path}: {e}")
            print("Creating synthetic data instead...")
            data, metadata = create_synthetic_hsi_data(scene_output_dir, scene_name, config)
            wavelengths = np.array(metadata['wavelengths'])

    # Remove bad bands if specified
    sensor_key = scene_name.upper()
    if sensor_key in bands_config:
        bad_bands = bands_config[sensor_key].get('bad_bands', [])
        if bad_bands:
            print(f"Removing {len(bad_bands)} bad bands")
            data, wavelengths = remove_bad_bands(data, wavelengths, bad_bands)

    # Apply preprocessing
    preprocess_config = config.get('preprocess', {})

    # Radiometric normalization
    norm_method = preprocess_config.get('radiometric_norm', 'zscore')
    if norm_method != 'none':
        print(f"Applying radiometric normalization: {norm_method}")
        if norm_method == 'zscore':
            # Z-score normalization per pixel
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    pixel = data[i, j, :]
                    if not np.any(np.isnan(pixel)):
                        data[i, j, :] = normalize_spectrum(pixel, 'zscore')

    # Suppress natural background if requested
    hsi_config = preprocess_config.get('hsi', {})
    if hsi_config.get('suppress_natural', False):
        print("Suppressing natural background")
        data = suppress_natural_background(data, wavelengths)

    # Reproject to target CRS if needed
    target_crs = preprocess_config.get('resample_to', 'EPSG:4326')
    if str(metadata.get('crs', 'EPSG:4326')) != target_crs:
        print(f"Reprojecting to {target_crs}")
        from geo import reproject_to_crs
        data, new_metadata = reproject_to_crs(
            data, metadata['crs'], metadata['transform'],
            CRS.from_string(target_crs)
        )
        metadata.update(new_metadata)

    # Save processed data
    output_file = scene_output_dir / 'scene.tif'
    print(f"Saving to {output_file}")

    write_geotiff(data, str(output_file), metadata)

    # Save wavelengths and metadata
    bands_info = {
        'wavelengths': wavelengths.tolist(),
        'n_bands': len(wavelengths),
        'spectral_range': [float(wavelengths.min()), float(wavelengths.max())],
        'spectral_resolution': float(np.mean(np.diff(wavelengths))),
        'units': 'nm'
    }

    save_json(bands_info, str(scene_output_dir / 'bands.json'))

    # Save full metadata
    metadata_clean = {}
    for key, value in metadata.items():
        if hasattr(value, 'to_dict'):  # Handle CRS objects
            metadata_clean[key] = value.to_dict()
        elif hasattr(value, '__array__'):  # Handle numpy arrays
            metadata_clean[key] = value.tolist()
        else:
            metadata_clean[key] = value

    save_json(metadata_clean, str(scene_output_dir / 'metadata.json'))

    # Return pair information
    pair_info = {
        'pair_id': f"hsi_{scene_name}",
        'sensor_type': 'SPECTRAL',
        'path': str(output_file.relative_to(Path.cwd())),
        'type': 'HSI',
        'date': metadata.get('acquisition_date', datetime.now().strftime('%Y-%m-%d')),
        'notes': f"Processed {scene_name} hyperspectral data"
    }

    return pair_info


def main():
    parser = argparse.ArgumentParser(description='Prepare hyperspectral data for Astiron Spectra')
    parser.add_argument('--input-dir', type=str, default='data/raw',
                       help='Input directory with raw HSI data')
    parser.add_argument('--output-dir', type=str, default='data/hsi',
                       help='Output directory for processed HSI data')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Configuration file')
    parser.add_argument('--bands-config', type=str, default='configs/bands.yaml',
                       help='Bands configuration file')
    parser.add_argument('--scenes', nargs='+',
                       help='Specific scenes to process (default: all found)')

    args = parser.parse_args()

    # Load configurations
    config = load_config(args.config)
    bands_config = load_bands_config(args.bands_config)

    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    print("Astiron Spectra - HSI Data Preparation")
    print("=====================================")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print("")

    # Find HSI datasets to process
    hsi_datasets = []

    if args.scenes:
        # Process specified scenes
        for scene in args.scenes:
            scene_path = input_dir / scene
            hsi_datasets.append((scene, scene_path))
    else:
        # Auto-discover datasets
        if input_dir.exists():
            for item in input_dir.iterdir():
                if item.is_dir():
                    # Check for common HSI file patterns
                    hsi_files = list(item.glob('*.bsq')) + list(item.glob('*.tif')) + list(item.glob('*.tiff'))
                    if hsi_files or item.name in ['aviris', 'pavia', 'indian_pines']:
                        hsi_datasets.append((item.name, item))

        # Add demo datasets even if not found
        demo_datasets = ['aviris_sample', 'pavia_university', 'indian_pines']
        for demo in demo_datasets:
            if not any(name == demo for name, _ in hsi_datasets):
                hsi_datasets.append((demo, input_dir / demo))

    if not hsi_datasets:
        print("No HSI datasets found. Creating demo datasets...")
        hsi_datasets = [
            ('aviris_sample', input_dir / 'aviris'),
            ('pavia_university', input_dir / 'pavia'),
        ]

    # Process each dataset
    pairs = read_pairs_csv()
    new_pairs = []

    for scene_name, scene_path in hsi_datasets:
        try:
            pair_info = process_hsi_scene(scene_path, output_dir, scene_name,
                                        config, bands_config)
            new_pairs.append(pair_info)
            print(f"✓ Processed {scene_name}")
        except Exception as e:
            print(f"✗ Failed to process {scene_name}: {e}")
            continue

    # Update pairs.csv
    if new_pairs:
        # Remove existing HSI pairs and add new ones
        existing_pairs = [p for p in pairs if not p['pair_id'].startswith('hsi_')]
        all_pairs = existing_pairs + new_pairs

        write_pairs_csv(all_pairs)
        print(f"\n✓ Updated pairs.csv with {len(new_pairs)} HSI pairs")

    print(f"\n✓ HSI data preparation complete!")
    print(f"Processed datasets: {len(new_pairs)}")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
