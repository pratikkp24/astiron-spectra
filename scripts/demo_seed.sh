#!/bin/bash
# Demo seed script for Astiron Spectra
# Downloads sample data and creates a working demo environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Astiron Spectra - Demo Seed Setup"
echo "=================================="
echo "Setting up demo environment with sample data..."
echo ""

cd "$PROJECT_ROOT"

# Step 1: Download sample datasets
echo "Step 1: Downloading sample datasets..."
echo "--------------------------------------"

# Download HSI sample data
echo "Downloading hyperspectral sample data..."
bash scripts/download_hsi.sh --subset aviris_sample

# Download thermal sample data
echo "Downloading thermal sample data..."
bash scripts/download_thermal.sh --subset flame3

echo "✓ Sample data download complete"
echo ""

# Step 2: Prepare datasets
echo "Step 2: Preparing datasets..."
echo "-----------------------------"

# Prepare HSI data
echo "Preparing hyperspectral data..."
python scripts/prepare_hsi.py --scenes aviris_sample

# Prepare thermal data (create the script first)
echo "Preparing thermal data..."
python scripts/prepare_tir.py --scenes flame3

echo "✓ Data preparation complete"
echo ""

# Step 3: Create demo pairs
echo "Step 3: Creating demo pairs..."
echo "------------------------------"

python scripts/make_pairs_csv.py --demo

echo "✓ Demo pairs created"
echo ""

# Step 4: Create demo output structure for UI
echo "Step 4: Setting up UI demo data..."
echo "----------------------------------"

# Create demo output directory structure
mkdir -p ui/public/outputs/demo_run/pair_001

# Create synthetic demo outputs for UI testing
python -c "
import numpy as np
import json
from pathlib import Path
import sys
sys.path.insert(0, 'common')
from astiron_io import write_geotiff, save_json
from rasterio.crs import CRS
from rasterio.transform import from_bounds

# Create demo output directory
output_dir = Path('ui/public/outputs/demo_run/pair_001')
output_dir.mkdir(parents=True, exist_ok=True)

# Create synthetic anomaly mask (small for demo)
height, width = 50, 50
mask = np.zeros((height, width), dtype=np.uint8)

# Add some anomalous regions
mask[10:15, 10:15] = 1  # Small square
mask[30:35, 20:30] = 1  # Rectangle
mask[40:45, 35:45] = 1  # Another square

# Create transform for San Francisco Bay area
bounds = (-122.5, 37.5, -122.4, 37.6)
transform = from_bounds(*bounds, width, height)
metadata = {
    'crs': CRS.from_epsg(4326),
    'transform': transform,
    'width': width,
    'height': height
}

# Save anomaly mask
write_geotiff(mask, str(output_dir / 'anomaly_mask.tif'), metadata)

# Create heatmap (anomaly scores)
heatmap = np.random.random((height, width)) * 0.3
heatmap[mask == 1] += 0.7  # Higher scores for anomalies
heatmap = np.clip(heatmap, 0, 1)

write_geotiff(heatmap, str(output_dir / 'anomaly_heatmap.tif'), metadata)

# Create GeoJSON polygons
geojson = {
    'type': 'FeatureCollection',
    'features': [
        {
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[
                    [-122.48, 37.52],
                    [-122.47, 37.52],
                    [-122.47, 37.53],
                    [-122.48, 37.53],
                    [-122.48, 37.52]
                ]]
            },
            'properties': {
                'anomaly_id': 1,
                'area_m2': 2500,
                'confidence': 0.85,
                'material': 'Unknown'
            }
        },
        {
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[
                    [-122.46, 37.54],
                    [-122.45, 37.54],
                    [-122.45, 37.55],
                    [-122.46, 37.55],
                    [-122.46, 37.54]
                ]]
            },
            'properties': {
                'anomaly_id': 2,
                'area_m2': 3600,
                'confidence': 0.72,
                'material': 'Mineral'
            }
        }
    ]
}

save_json(geojson, str(output_dir / 'anomaly_polygons.geojson'))

# Create characterization CSV
import csv
with open(output_dir / 'characterization.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['anomaly_id', 'material', 'confidence', 'category', 'area_m2'])
    writer.writerow([1, 'Quartz', 0.85, 'silicate', 2500])
    writer.writerow([2, 'Hematite', 0.72, 'oxide', 3600])

# Create manifest
manifest = {
    'pair_id': 'demo_pair_001',
    'run_id': 'demo_run',
    'timestamp': '2024-12-01T14:30:22Z',
    'processing_time_seconds': 45.2,
    'algorithms': {
        'hsi_detector': 'hybrid_rx_ae',
        'tir_detector': 'cnn_patch',
        'fusion': 'late_score'
    },
    'thresholds': {
        'hsi': 0.75,
        'tir': 0.68,
        'fusion': 0.71
    },
    'statistics': {
        'total_anomalies': 2,
        'total_area_m2': 6100,
        'coverage_percent': 0.24
    },
    'files': {
        'anomaly_mask': 'anomaly_mask.tif',
        'anomaly_heatmap': 'anomaly_heatmap.tif',
        'anomaly_polygons': 'anomaly_polygons.geojson',
        'characterization': 'characterization.csv'
    }
}

save_json(manifest, str(output_dir / 'manifest.json'))

print('✓ Demo UI data created')
"

# Create output index for UI
echo "Creating output index..."
python -c "
import json
from pathlib import Path

# Create outputs index
outputs_index = {
    'runs': [
        {
            'run_id': 'demo_run',
            'timestamp': '2024-12-01T14:30:22Z',
            'pairs': [
                {
                    'pair_id': 'pair_001',
                    'sensor_types': ['HSI', 'TIR'],
                    'has_fusion': True,
                    'anomaly_count': 2,
                    'total_area_m2': 6100
                }
            ]
        }
    ]
}

# Save index
with open('ui/public/outputs/index.json', 'w') as f:
    json.dump(outputs_index, f, indent=2)

print('✓ Output index created')
"

echo "✓ UI demo data setup complete"
echo ""

# Step 5: Verify setup
echo "Step 5: Verifying setup..."
echo "--------------------------"

# Check that key files exist
files_to_check=(
    "pairs.csv"
    "data/hsi/aviris_sample/scene.tif"
    "data/tir/flame3_sample/thermal.tif"
    "ui/public/outputs/demo_run/pair_001/manifest.json"
    "ui/public/outputs/index.json"
)

all_good=true
for file in "${files_to_check[@]}"; do
    if [[ -f "$file" ]]; then
        echo "✓ $file"
    else
        echo "✗ $file (missing)"
        all_good=false
    fi
done

if [[ "$all_good" == true ]]; then
    echo ""
    echo "🎉 Demo seed setup complete!"
    echo ""
    echo "Next steps:"
    echo "1. Build Docker services:    make build"
    echo "2. Run inference pipeline:   make run RUN_ID=demo_\$(date +%Y%m%d_%H%M%S)"
    echo "3. Launch UI:                make ui"
    echo ""
    echo "The demo includes:"
    echo "- Synthetic hyperspectral data (AVIRIS-like)"
    echo "- Synthetic thermal infrared data (FLAME-3-like)"
    echo "- Pre-generated UI demo data"
    echo "- Complete pairs.csv configuration"
else
    echo ""
    echo "❌ Demo seed setup incomplete - some files are missing"
    echo "Please check the errors above and try again"
    exit 1
fi
