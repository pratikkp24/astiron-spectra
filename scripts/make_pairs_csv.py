#!/usr/bin/env python3
"""
Create pairs.csv file for Astiron Spectra
Manages the pairing of HSI and TIR data for processing
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'common'))

from astiron_io import read_pairs_csv, write_pairs_csv


def create_demo_pairs() -> List[Dict]:
    """Create demo pairs for testing"""

    demo_pairs = [
        {
            'pair_id': 'hsi_aviris_sample',
            'sensor_type': 'SPECTRAL',
            'path': 'data/hsi/aviris_sample/scene.tif',
            'type': 'HSI',
            'date': '2024-06-01',
            'notes': 'Synthetic AVIRIS-like hyperspectral sample'
        },
        {
            'pair_id': 'tir_flame3',
            'sensor_type': 'THERMAL',
            'path': 'data/tir/flame3_sample/thermal.tif',
            'type': 'TIR',
            'date': '2024-06-01',
            'notes': 'Synthetic FLAME-3-like thermal sample'
        },
        {
            'pair_id': 'hsi_pavia_university',
            'sensor_type': 'SPECTRAL',
            'path': 'data/hsi/pavia_university/scene.tif',
            'type': 'HSI',
            'date': '2024-06-02',
            'notes': 'Synthetic Pavia University-like sample'
        },
        {
            'pair_id': 'tir_kaist',
            'sensor_type': 'THERMAL',
            'path': 'data/tir/kaist_sample/thermal.tif',
            'type': 'TIR',
            'date': '2024-06-02',
            'notes': 'Synthetic KAIST-like thermal sample'
        }
    ]

    return demo_pairs


def discover_data_pairs(hsi_dir: Path, tir_dir: Path) -> List[Dict]:
    """Discover HSI and TIR data pairs automatically"""

    pairs = []

    # Find HSI data
    if hsi_dir.exists():
        for scene_dir in hsi_dir.iterdir():
            if scene_dir.is_dir():
                scene_file = scene_dir / 'scene.tif'
                if scene_file.exists():
                    pair = {
                        'pair_id': f'hsi_{scene_dir.name}',
                        'sensor_type': 'SPECTRAL',
                        'path': str(scene_file.relative_to(Path.cwd())),
                        'type': 'HSI',
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'notes': f'Hyperspectral data from {scene_dir.name}'
                    }
                    pairs.append(pair)

    # Find TIR data
    if tir_dir.exists():
        for scene_dir in tir_dir.iterdir():
            if scene_dir.is_dir():
                thermal_file = scene_dir / 'thermal.tif'
                if thermal_file.exists():
                    pair = {
                        'pair_id': f'tir_{scene_dir.name}',
                        'sensor_type': 'THERMAL',
                        'path': str(thermal_file.relative_to(Path.cwd())),
                        'type': 'TIR',
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'notes': f'Thermal infrared data from {scene_dir.name}'
                    }
                    pairs.append(pair)

    return pairs


def validate_pairs(pairs: List[Dict]) -> List[Dict]:
    """Validate that pair files exist and are accessible"""

    valid_pairs = []

    for pair in pairs:
        pair_path = Path(pair['path'])

        if pair_path.exists():
            valid_pairs.append(pair)
            print(f"✓ {pair['pair_id']}: {pair['path']}")
        else:
            print(f"✗ {pair['pair_id']}: {pair['path']} (file not found)")

    return valid_pairs


def create_multimodal_pairs(pairs: List[Dict]) -> List[Dict]:
    """Create multimodal pairs by matching HSI and TIR data by date/location"""

    # Group pairs by type
    hsi_pairs = [p for p in pairs if p['type'] == 'HSI']
    tir_pairs = [p for p in pairs if p['type'] == 'TIR']

    multimodal_pairs = []

    # Simple matching by date (in a real system, this would be more sophisticated)
    for hsi_pair in hsi_pairs:
        for tir_pair in tir_pairs:
            if hsi_pair['date'] == tir_pair['date']:
                # Create multimodal pair
                multimodal_pair = {
                    'pair_id': f"multimodal_{hsi_pair['date'].replace('-', '')}",
                    'sensor_type': 'MULTIMODAL',
                    'path': f"{hsi_pair['path']};{tir_pair['path']}",
                    'type': 'HSI+TIR',
                    'date': hsi_pair['date'],
                    'notes': f"Multimodal pair: {hsi_pair['pair_id']} + {tir_pair['pair_id']}"
                }
                multimodal_pairs.append(multimodal_pair)
                break

    return multimodal_pairs


def main():
    parser = argparse.ArgumentParser(description='Create pairs.csv for Astiron Spectra')
    parser.add_argument('--demo', action='store_true',
                       help='Create demo pairs for testing')
    parser.add_argument('--hsi-dir', type=str, default='data/hsi',
                       help='HSI data directory')
    parser.add_argument('--tir-dir', type=str, default='data/tir',
                       help='TIR data directory')
    parser.add_argument('--output', type=str, default='pairs.csv',
                       help='Output CSV file')
    parser.add_argument('--multimodal', action='store_true',
                       help='Create multimodal pairs')
    parser.add_argument('--validate', action='store_true',
                       help='Validate that all pair files exist')

    args = parser.parse_args()

    print("Astiron Spectra - Pairs CSV Creation")
    print("====================================")

    if args.demo:
        print("Creating demo pairs...")
        pairs = create_demo_pairs()
    else:
        print("Discovering data pairs...")
        hsi_dir = Path(args.hsi_dir)
        tir_dir = Path(args.tir_dir)

        pairs = discover_data_pairs(hsi_dir, tir_dir)

        if args.multimodal:
            print("Creating multimodal pairs...")
            multimodal_pairs = create_multimodal_pairs(pairs)
            pairs.extend(multimodal_pairs)

    if args.validate:
        print("\nValidating pairs...")
        pairs = validate_pairs(pairs)

    # Read existing pairs and merge
    existing_pairs = read_pairs_csv()

    # Remove duplicates (keep new pairs)
    existing_ids = {p['pair_id'] for p in existing_pairs}
    new_pairs = [p for p in pairs if p['pair_id'] not in existing_ids]

    # Combine
    all_pairs = existing_pairs + new_pairs

    # Write pairs.csv
    write_pairs_csv(all_pairs)

    print(f"\n✓ Pairs CSV created: {args.output}")
    print(f"Total pairs: {len(all_pairs)}")
    print(f"New pairs: {len(new_pairs)}")

    # Summary by type
    type_counts = {}
    for pair in all_pairs:
        pair_type = pair['type']
        type_counts[pair_type] = type_counts.get(pair_type, 0) + 1

    print("\nPairs by type:")
    for pair_type, count in type_counts.items():
        print(f"  {pair_type}: {count}")

    print(f"\nPairs file ready for processing!")


if __name__ == '__main__':
    main()
