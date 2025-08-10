"""
Geospatial utilities for Astiron Spectra
Handles coordinate transformations, projections, and spatial operations
"""

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.ndimage import shift
from skimage.registration import phase_cross_correlation
from typing import Dict, Tuple, Optional


def coregister_images(reference: np.ndarray, target: np.ndarray, 
                     method: str = 'phase_corr') -> Tuple[np.ndarray, Dict]:
    """
    Coregister target image to reference image
    
    Args:
        reference: Reference image (2D array)
        target: Target image to be aligned (2D array)
        method: Coregistration method ('phase_corr' or 'none')
    
    Returns:
        Tuple of (aligned_target, shift_info)
    """
    if method == 'none':
        return target, {'shift': (0, 0), 'error': 0.0}
    
    elif method == 'phase_corr':
        # Use phase correlation for sub-pixel registration
        shift_yx, error, _ = phase_cross_correlation(
            reference, target, 
            upsample_factor=10,
            return_error=True
        )
        
        # Apply shift to target image
        aligned_target = shift(target, shift_yx, mode='constant', cval=0)
        
        shift_info = {
            'shift': tuple(shift_yx),
            'error': float(error),
            'method': method
        }
        
        return aligned_target, shift_info
    
    else:
        raise ValueError(f"Unknown coregistration method: {method}")


def reproject_to_crs(data: np.ndarray, src_crs: CRS, src_transform, 
                     dst_crs: CRS, dst_resolution: float = None) -> Tuple[np.ndarray, Dict]:
    """
    Reproject raster data to target CRS
    
    Args:
        data: Input raster data (2D or 3D array)
        src_crs: Source CRS
        src_transform: Source geotransform
        dst_crs: Target CRS
        dst_resolution: Target resolution in target CRS units
    
    Returns:
        Tuple of (reprojected_data, metadata)
    """
    if data.ndim == 2:
        height, width = data.shape
        count = 1
        data = data[np.newaxis, :, :]
    else:
        count, height, width = data.shape
    
    # Calculate target transform and dimensions
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, width, height, *rasterio.transform.array_bounds(height, width, src_transform),
        resolution=dst_resolution
    )
    
    # Create output array
    dst_data = np.zeros((count, dst_height, dst_width), dtype=data.dtype)
    
    # Reproject each band
    reproject(
        data, dst_data,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear
    )
    
    # Remove band dimension if input was 2D
    if count == 1:
        dst_data = dst_data[0]
    
    metadata = {
        'crs': dst_crs,
        'transform': dst_transform,
        'width': dst_width,
        'height': dst_height,
        'count': count
    }
    
    return dst_data, metadata


def get_pixel_coordinates(transform, height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get pixel coordinates in geographic space
    
    Returns:
        Tuple of (x_coords, y_coords) arrays
    """
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    return np.array(xs), np.array(ys)


def calculate_pixel_area(transform, crs: CRS) -> float:
    """
    Calculate area of a single pixel in square meters
    
    Args:
        transform: Rasterio transform
        crs: Coordinate reference system
    
    Returns:
        Pixel area in square meters
    """
    if crs.is_geographic:
        # For geographic CRS, approximate using middle latitude
        # This is a simplification - more accurate methods exist
        lat_center = (transform[5] + transform[5] + transform[4] * 100) / 2
        meters_per_degree_lat = 111320  # Approximate
        meters_per_degree_lon = 111320 * np.cos(np.radians(lat_center))
        
        pixel_width_m = abs(transform[0]) * meters_per_degree_lon
        pixel_height_m = abs(transform[4]) * meters_per_degree_lat
        
        return pixel_width_m * pixel_height_m
    else:
        # For projected CRS, use transform directly
        return abs(transform[0] * transform[4])


def bounds_to_transform(bounds: Tuple[float, float, float, float], 
                       width: int, height: int):
    """
    Create transform from bounds and dimensions
    
    Args:
        bounds: (left, bottom, right, top)
        width: Image width in pixels
        height: Image height in pixels
    
    Returns:
        Rasterio transform
    """
    left, bottom, right, top = bounds
    return from_bounds(left, bottom, right, top, width, height)


def create_water_mask(data: np.ndarray, wavelengths: np.ndarray, 
                     threshold: float = 0.05) -> np.ndarray:
    """
    Create water mask using spectral characteristics
    
    Args:
        data: Hyperspectral data (height, width, bands)
        wavelengths: Wavelength array in nm
        threshold: NDWI threshold for water detection
    
    Returns:
        Binary water mask (True = water)
    """
    # Find bands closest to 560nm (green) and 1240nm (NIR)
    green_idx = np.argmin(np.abs(wavelengths - 560))
    nir_idx = np.argmin(np.abs(wavelengths - 1240))
    
    green_band = data[:, :, green_idx]
    nir_band = data[:, :, nir_idx]
    
    # Calculate NDWI (Normalized Difference Water Index)
    with np.errstate(divide='ignore', invalid='ignore'):
        ndwi = (green_band - nir_band) / (green_band + nir_band)
        ndwi = np.nan_to_num(ndwi, nan=0.0)
    
    return ndwi > threshold


def suppress_natural_background(data: np.ndarray, wavelengths: np.ndarray,
                              vegetation_threshold: float = 0.3) -> np.ndarray:
    """
    Suppress natural background (vegetation, soil) to enhance anomalies
    
    Args:
        data: Hyperspectral data (height, width, bands)
        wavelengths: Wavelength array in nm
        vegetation_threshold: NDVI threshold for vegetation
    
    Returns:
        Background-suppressed data
    """
    # Find red and NIR bands for NDVI
    red_idx = np.argmin(np.abs(wavelengths - 670))
    nir_idx = np.argmin(np.abs(wavelengths - 800))
    
    red_band = data[:, :, red_idx]
    nir_band = data[:, :, nir_idx]
    
    # Calculate NDVI
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (nir_band - red_band) / (nir_band + red_band)
        ndvi = np.nan_to_num(ndvi, nan=0.0)
    
    # Create vegetation mask
    veg_mask = ndvi > vegetation_threshold
    
    # Suppress vegetation pixels by reducing their spectral contrast
    suppressed_data = data.copy()
    suppressed_data[veg_mask] *= 0.5  # Reduce vegetation signal
    
    return suppressed_data


def get_geographic_bounds(transform, height: int, width: int) -> Tuple[float, float, float, float]:
    """
    Get geographic bounds from transform and dimensions
    
    Returns:
        (left, bottom, right, top) bounds
    """
    left = transform[2]
    top = transform[5]
    right = left + width * transform[0]
    bottom = top + height * transform[4]
    
    return left, bottom, right, top