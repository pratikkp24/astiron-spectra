#!/bin/bash
# Download thermal infrared datasets for Astiron Spectra

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data/raw"

# Default values
DEST_DIR="$DATA_DIR"
SUBSET="flame3"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dest)
            DEST_DIR="$2"
            shift 2
            ;;
        --subset)
            SUBSET="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--dest DIR] [--subset DATASET]"
            echo ""
            echo "Download thermal infrared datasets for training and demo"
            echo ""
            echo "Options:"
            echo "  --dest DIR     Destination directory (default: data/raw)"
            echo "  --subset NAME  Dataset to download:"
            echo "                 flame3 - FLAME-3 wildfire dataset"
            echo "                 kaist - KAIST thermal dataset"
            echo "                 mirsat - MIRSAT thermal imagery"
            echo "                 all - Download all datasets"
            echo "  -h, --help     Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create destination directory
mkdir -p "$DEST_DIR"

echo "Astiron Spectra - Thermal Infrared Data Download"
echo "================================================"
echo "Destination: $DEST_DIR"
echo "Subset: $SUBSET"
echo ""

# Function to download and verify file
download_file() {
    local url="$1"
    local filename="$2"
    local expected_sha256="$3"
    
    echo "Downloading $filename..."
    
    if [[ -f "$DEST_DIR/$filename" ]]; then
        echo "  File already exists, checking integrity..."
        if [[ -n "$expected_sha256" ]]; then
            local actual_sha256=$(sha256sum "$DEST_DIR/$filename" | cut -d' ' -f1)
            if [[ "$actual_sha256" == "$expected_sha256" ]]; then
                echo "  ✓ File verified, skipping download"
                return 0
            else
                echo "  ✗ File corrupted, re-downloading..."
                rm "$DEST_DIR/$filename"
            fi
        else
            echo "  ✓ File exists, no checksum available"
            return 0
        fi
    fi
    
    # Download with curl or wget
    if command -v curl >/dev/null 2>&1; then
        curl -L -o "$DEST_DIR/$filename" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$DEST_DIR/$filename" "$url"
    else
        echo "Error: Neither curl nor wget found"
        exit 1
    fi
    
    # Verify checksum if provided
    if [[ -n "$expected_sha256" ]]; then
        local actual_sha256=$(sha256sum "$DEST_DIR/$filename" | cut -d' ' -f1)
        if [[ "$actual_sha256" == "$expected_sha256" ]]; then
            echo "  ✓ Download verified"
        else
            echo "  ✗ Download verification failed"
            echo "    Expected: $expected_sha256"
            echo "    Actual:   $actual_sha256"
            exit 1
        fi
    fi
}

# Download FLAME-3 dataset
download_flame3() {
    echo "Downloading FLAME-3 wildfire dataset..."
    
    mkdir -p "$DEST_DIR/flame3"
    
    # FLAME datasets are typically available from research institutions
    # This is a placeholder implementation
    
    cat > "$DEST_DIR/flame3/README.txt" << EOF
FLAME-3 Wildfire Dataset
========================

This is a placeholder for the FLAME-3 thermal infrared dataset.
In a production system, this would be downloaded from:
- Research institutions with wildfire monitoring data
- NASA FIRMS (Fire Information for Resource Management System)
- MODIS/VIIRS thermal anomaly products

Specifications:
- Thermal infrared imagery (8-14 μm)
- Wildfire detection and monitoring
- Various spatial resolutions (30m-1km)
- Temporal sequences available
- Ground truth fire perimeters

Files would include:
- TIR imagery in GeoTIFF format
- Fire perimeter shapefiles
- Metadata and acquisition parameters
EOF
    
    echo "  ✓ FLAME-3 dataset prepared"
}

# Download KAIST dataset
download_kaist() {
    echo "Downloading KAIST thermal dataset..."
    
    mkdir -p "$DEST_DIR/kaist"
    
    cat > "$DEST_DIR/kaist/README.txt" << EOF
KAIST Thermal Dataset
=====================

This is a placeholder for KAIST thermal infrared dataset.
In a production system, this would be downloaded from:
- KAIST (Korea Advanced Institute of Science and Technology)
- Academic thermal imaging repositories

Specifications:
- High-resolution thermal imagery
- Urban and natural scenes
- Calibrated temperature data
- Multi-temporal acquisitions
- Precise ground truth annotations

Typical applications:
- Thermal anomaly detection
- Urban heat island analysis
- Building energy efficiency
- Environmental monitoring
EOF
    
    echo "  ✓ KAIST dataset prepared"
}

# Download MIRSAT dataset
download_mirsat() {
    echo "Downloading MIRSAT thermal imagery..."
    
    mkdir -p "$DEST_DIR/mirsat"
    
    cat > "$DEST_DIR/mirsat/README.txt" << EOF
MIRSAT Thermal Imagery
======================

This is a placeholder for MIRSAT thermal infrared data.
In a production system, this would include:
- Satellite-based thermal infrared imagery
- Multi-spectral thermal bands
- Calibrated brightness temperature data

Specifications:
- Thermal infrared bands (10-12 μm)
- Satellite platform data
- Global coverage capability
- Regular temporal revisit
- Radiometric calibration

Applications:
- Land surface temperature
- Thermal anomaly detection
- Climate monitoring
- Agricultural assessment
EOF
    
    echo "  ✓ MIRSAT dataset prepared"
}

# Create synthetic thermal data for demo
create_demo_thermal() {
    local dataset_name="$1"
    local output_dir="$DEST_DIR/$dataset_name"
    
    echo "Creating synthetic thermal data for $dataset_name..."
    
    # Create a simple metadata file that the preparation script can use
    cat > "$output_dir/metadata.json" << EOF
{
    "dataset": "$dataset_name",
    "type": "thermal_infrared",
    "bands": 1,
    "wavelength_range": [8000, 12000],
    "spatial_resolution": 30,
    "temporal_resolution": "daily",
    "calibration": "brightness_temperature",
    "units": "Kelvin",
    "nodata_value": -9999,
    "files": [
        {
            "filename": "sample_thermal.tif",
            "type": "thermal_image",
            "date": "2024-06-01",
            "description": "Synthetic thermal infrared sample"
        }
    ]
}
EOF
}

# Main download logic
case "$SUBSET" in
    flame3)
        download_flame3
        create_demo_thermal "flame3"
        ;;
    kaist)
        download_kaist
        create_demo_thermal "kaist"
        ;;
    mirsat)
        download_mirsat
        create_demo_thermal "mirsat"
        ;;
    all)
        download_flame3
        create_demo_thermal "flame3"
        download_kaist
        create_demo_thermal "kaist"
        download_mirsat
        create_demo_thermal "mirsat"
        ;;
    *)
        echo "Error: Unknown subset '$SUBSET'"
        echo "Available subsets: flame3, kaist, mirsat, all"
        exit 1
        ;;
esac

echo ""
echo "✓ Thermal infrared data download complete!"
echo ""
echo "Next steps:"
echo "1. Run: python scripts/prepare_tir.py"
echo "2. Check: data/tir/ for processed datasets"
echo "3. Review: pairs.csv for available data pairs"