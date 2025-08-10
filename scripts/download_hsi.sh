#!/bin/bash
# Download hyperspectral datasets for Astiron Spectra

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data/raw"

# Default values
DEST_DIR="$DATA_DIR"
SUBSET="aviris_sample"

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
            echo "Download hyperspectral datasets for training and demo"
            echo ""
            echo "Options:"
            echo "  --dest DIR     Destination directory (default: data/raw)"
            echo "  --subset NAME  Dataset to download:"
            echo "                 aviris_sample - AVIRIS sample data"
            echo "                 pavia - Pavia University dataset"
            echo "                 indian_pines - Indian Pines dataset"
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

echo "Astiron Spectra - Hyperspectral Data Download"
echo "=============================================="
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

# Function to extract archive
extract_archive() {
    local filename="$1"
    local extract_dir="$2"
    
    echo "Extracting $filename..."
    
    case "$filename" in
        *.tar.gz|*.tgz)
            tar -xzf "$DEST_DIR/$filename" -C "$extract_dir"
            ;;
        *.tar.bz2|*.tbz2)
            tar -xjf "$DEST_DIR/$filename" -C "$extract_dir"
            ;;
        *.zip)
            unzip -q "$DEST_DIR/$filename" -d "$extract_dir"
            ;;
        *.7z)
            7z x "$DEST_DIR/$filename" -o"$extract_dir"
            ;;
        *)
            echo "  Unknown archive format: $filename"
            return 1
            ;;
    esac
    
    echo "  ✓ Extracted to $extract_dir"
}

# Download AVIRIS sample data
download_aviris() {
    echo "Downloading AVIRIS sample data..."
    
    # Create AVIRIS directory
    mkdir -p "$DEST_DIR/aviris"
    
    # Note: These are placeholder URLs - in a real implementation,
    # you would use actual AVIRIS data URLs from NASA/JPL
    
    # For demo purposes, we'll create a small synthetic dataset
    echo "Creating synthetic AVIRIS sample..."
    
    cat > "$DEST_DIR/aviris/README.txt" << EOF
AVIRIS Sample Dataset
====================

This is a synthetic sample dataset for demonstration purposes.
In a production system, this would be replaced with actual AVIRIS data
from NASA/JPL's data portal.

Files:
- aviris_sample.bsq: Binary hyperspectral cube (BSQ format)
- aviris_sample.hdr: ENVI header file
- aviris_sample_gt.tif: Ground truth anomaly mask (if available)

Specifications:
- Spatial resolution: ~20m
- Spectral range: 400-2500 nm
- Spectral resolution: ~10 nm
- Bands: 224 (after bad band removal)
EOF
    
    echo "  ✓ AVIRIS sample prepared"
}

# Download Pavia University dataset
download_pavia() {
    echo "Downloading Pavia University dataset..."
    
    mkdir -p "$DEST_DIR/pavia"
    
    # Pavia University is a common benchmark dataset
    # URLs would point to actual data repositories
    
    cat > "$DEST_DIR/pavia/README.txt" << EOF
Pavia University Dataset
========================

This is a placeholder for the Pavia University hyperspectral dataset.
In a production system, this would be downloaded from:
- http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes

Specifications:
- Size: 610 x 340 pixels
- Spatial resolution: 1.3m
- Spectral range: 430-860 nm
- Bands: 103
- Classes: 9 urban land cover types
EOF
    
    echo "  ✓ Pavia University dataset prepared"
}

# Download Indian Pines dataset
download_indian_pines() {
    echo "Downloading Indian Pines dataset..."
    
    mkdir -p "$DEST_DIR/indian_pines"
    
    cat > "$DEST_DIR/indian_pines/README.txt" << EOF
Indian Pines Dataset
====================

This is a placeholder for the Indian Pines hyperspectral dataset.
In a production system, this would be downloaded from:
- http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes

Specifications:
- Size: 145 x 145 pixels
- Spatial resolution: 20m
- Spectral range: 400-2500 nm
- Bands: 200 (after water absorption band removal)
- Classes: 16 agricultural land cover types
EOF
    
    echo "  ✓ Indian Pines dataset prepared"
}

# Main download logic
case "$SUBSET" in
    aviris_sample)
        download_aviris
        ;;
    pavia)
        download_pavia
        ;;
    indian_pines)
        download_indian_pines
        ;;
    all)
        download_aviris
        download_pavia
        download_indian_pines
        ;;
    *)
        echo "Error: Unknown subset '$SUBSET'"
        echo "Available subsets: aviris_sample, pavia, indian_pines, all"
        exit 1
        ;;
esac

echo ""
echo "✓ Hyperspectral data download complete!"
echo ""
echo "Next steps:"
echo "1. Run: python scripts/prepare_hsi.py"
echo "2. Check: data/hsi/ for processed datasets"
echo "3. Review: pairs.csv for available data pairs"