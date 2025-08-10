# Astiron Spectra

**Spectral-Spatial Change Intelligence**

> Verification test - confirming repository access and workflow functionality.

AI/ML-based Spectral-Spatial Anomaly Detection in Hyperspectral & Thermal (IR) data for PS-11.

## Features

- **Offline-first**: Air-gapped runtime with no network dependencies
- **Reproducible**: Dockerized microservices with deterministic outputs
- **Apple-grade UI**: Astiron Spectra Studio with polished macOS aesthetic
- **Multi-modal**: Hyperspectral (HSI) and Thermal Infrared (TIR) fusion
- **Submission-ready**: Automated packaging with model MD5 hashes

## Quick Start

```bash
# 1. Download demo data and prepare
bash scripts/demo_seed.sh

# 2. Build all services
make build

# 3. Run inference pipeline
make run RUN_ID=$(date +%Y%m%d_%H%M%S)

# 4. Launch UI
make ui
```

## Architecture

### Inference Pipeline
```
preprocess → detect_hsi → detect_tir → characterize → fuse → postprocess
```

### Outputs per pair
- `anomaly_mask.tif` - GeoTIFF binary mask
- `anomaly_heatmap.png` - PNG visualization
- `anomaly_polygons.geojson` - Vector polygons
- `anomaly_polygons.shp` - Shapefile format
- `characterization.csv` - Material classification
- `fusion_mask.tif` - Multi-modal fusion (when both HSI/TIR available)
- `metrics.json` - Performance metrics (when ground truth available)
- `manifest.json` - Run metadata and checksums

## Data Requirements

### Official Data (Stage-1)
Place archives under `data/raw/official/`:
- **PRISMA/ENMAP**: Hyperspectral cubes
- **Landsat-8/9**: TIR/SWIR bands

### Demo Data (Public)
Automatically downloaded by `demo_seed.sh`:
- AVIRIS hyperspectral samples
- FLAME-3 thermal imagery
- Pavia University dataset

## UI - Astiron Spectra Studio

Offline-first React application with:
- **4 Comparison Modes**: Mask Overlay, Swipe, Split, Flicker
- **Layer Controls**: HSI/TIR masks, heatmaps, opacity
- **Statistics**: Anomaly count, area, intensity metrics
- **Inspector**: Click polygons for characterization details
- **Export Tools**: GeoJSON, Shapefile, submission packages

## Configuration

All parameters in `configs/config.yaml`:
- Runtime settings (threads, GPU)
- Preprocessing options
- Detection algorithms
- Fusion strategies
- Output formats

## Development

```bash
# Install dependencies
pip install -r requirements.txt
cd ui && npm install

# Run tests
pytest tests/
cd ui && npm test

# Build for production
make build
cd ui && npm run build
```

## Submission

```bash
# Generate model hashes
bash scripts/compute_md5.sh

# Package submission
bash scripts/pack_submission.sh --team "ASTIRON" --date $(date +%d-%b-%Y)
# Produces: PS11_<DATE>_ASTIRON.zip
```

## License

MIT License - see LICENSE file for details.
