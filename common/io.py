"""
I/O utilities for Astiron Spectra
Handles reading/writing of various geospatial formats
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
import yaml
from rasterio.crs import CRS
from rasterio.transform import from_bounds


def load_config(config_path: str = "configs/config.yaml") -> Dict:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_bands_config(bands_path: str = "configs/bands.yaml") -> Dict:
    """Load band configuration file"""
    with open(bands_path, 'r') as f:
        return yaml.safe_load(f)


def read_hyperspectral(file_path: str) -> Tuple[np.ndarray, Dict]:
    """
    Read hyperspectral data from various formats
    Returns: (data, metadata)
    """
    file_path = Path(file_path)
    
    if file_path.suffix.lower() in ['.tif', '.tiff']:
        return read_geotiff_cube(str(file_path))
    elif file_path.suffix.lower() in ['.bsq', '.bil', '.bip']:
        return read_envi_format(str(file_path))
    else:
        raise ValueError(f"Unsupported hyperspectral format: {file_path.suffix}")


def read_geotiff_cube(file_path: str) -> Tuple[np.ndarray, Dict]:
    """Read multi-band GeoTIFF as hyperspectral cube"""
    with rasterio.open(file_path) as src:
        data = src.read()  # Shape: (bands, height, width)
        metadata = {
            'crs': src.crs,
            'transform': src.transform,
            'width': src.width,
            'height': src.height,
            'count': src.count,
            'dtype': src.dtypes[0],
            'nodata': src.nodata
        }
    
    # Transpose to (height, width, bands) for processing
    data = np.transpose(data, (1, 2, 0))
    return data, metadata


def read_envi_format(file_path: str) -> Tuple[np.ndarray, Dict]:
    """Read ENVI format hyperspectral data"""
    # Look for header file
    hdr_path = file_path.replace('.bsq', '.hdr').replace('.bil', '.hdr').replace('.bip', '.hdr')
    
    if not os.path.exists(hdr_path):
        raise FileNotFoundError(f"Header file not found: {hdr_path}")
    
    # Parse ENVI header
    metadata = parse_envi_header(hdr_path)
    
    # Read binary data
    with open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=metadata['dtype'])
    
    # Reshape based on interleave format
    if metadata['interleave'].upper() == 'BSQ':
        data = data.reshape(metadata['bands'], metadata['lines'], metadata['samples'])
        data = np.transpose(data, (1, 2, 0))  # (height, width, bands)
    elif metadata['interleave'].upper() == 'BIL':
        data = data.reshape(metadata['lines'], metadata['bands'], metadata['samples'])
        data = np.transpose(data, (0, 2, 1))  # (height, width, bands)
    elif metadata['interleave'].upper() == 'BIP':
        data = data.reshape(metadata['lines'], metadata['samples'], metadata['bands'])
    
    return data, metadata


def parse_envi_header(hdr_path: str) -> Dict:
    """Parse ENVI header file"""
    metadata = {}
    
    with open(hdr_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip().lower()
                value = value.strip()
                
                # Convert numeric values
                if key in ['samples', 'lines', 'bands']:
                    metadata[key] = int(value)
                elif key == 'data type':
                    # ENVI data type to numpy dtype mapping
                    dtype_map = {
                        '1': np.uint8, '2': np.int16, '3': np.int32,
                        '4': np.float32, '5': np.float64, '12': np.uint16
                    }
                    metadata['dtype'] = dtype_map.get(value, np.float32)
                else:
                    metadata[key] = value
    
    return metadata


def write_geotiff(data: np.ndarray, file_path: str, metadata: Dict, 
                  compress: str = 'lzw') -> None:
    """Write numpy array as GeoTIFF"""
    if data.ndim == 2:
        # Single band
        height, width = data.shape
        count = 1
        data = data[np.newaxis, :, :]  # Add band dimension
    elif data.ndim == 3:
        # Multi-band - transpose to (bands, height, width)
        if data.shape[2] < data.shape[0]:  # Assume (height, width, bands)
            data = np.transpose(data, (2, 0, 1))
        count, height, width = data.shape
    else:
        raise ValueError(f"Unsupported data dimensions: {data.shape}")
    
    # Set default metadata if not provided
    crs = metadata.get('crs', CRS.from_epsg(4326))
    transform = metadata.get('transform', from_bounds(-180, -90, 180, 90, width, height))
    
    with rasterio.open(
        file_path, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=count,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        compress=compress,
        tiled=True,
        blockxsize=512,
        blockysize=512
    ) as dst:
        dst.write(data)


def read_thermal(file_path: str) -> Tuple[np.ndarray, Dict]:
    """Read thermal infrared data"""
    with rasterio.open(file_path) as src:
        data = src.read(1)  # Single band
        metadata = {
            'crs': src.crs,
            'transform': src.transform,
            'width': src.width,
            'height': src.height,
            'dtype': src.dtypes[0],
            'nodata': src.nodata
        }
    
    return data, metadata


def save_json(data: Dict, file_path: str, indent: int = 2) -> None:
    """Save dictionary as JSON file"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(file_path: str) -> Dict:
    """Load JSON file as dictionary"""
    with open(file_path, 'r') as f:
        return json.load(f)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if not"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_pairs_csv_path() -> str:
    """Get path to pairs.csv file"""
    return "pairs.csv"


def read_pairs_csv() -> List[Dict]:
    """Read pairs.csv file"""
    import csv
    
    pairs_path = get_pairs_csv_path()
    if not os.path.exists(pairs_path):
        return []
    
    pairs = []
    with open(pairs_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append(row)
    
    return pairs


def write_pairs_csv(pairs: List[Dict]) -> None:
    """Write pairs.csv file"""
    import csv
    
    if not pairs:
        return
    
    pairs_path = get_pairs_csv_path()
    fieldnames = ['pair_id', 'sensor_type', 'path', 'type', 'date', 'notes']
    
    with open(pairs_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairs)


def get_output_path(run_id: str, pair_id: str, suffix: str = "") -> Path:
    """Get output path for a specific run and pair"""
    output_dir = Path("output") / "runs" / run_id / pair_id
    ensure_dir(output_dir)
    
    if suffix:
        return output_dir / f"{pair_id}{suffix}"
    return output_dir