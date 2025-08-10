"""
Spectral analysis utilities for Astiron Spectra
Handles spectral matching, similarity metrics, and hyperspectral processing
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from scipy.spatial.distance import cosine
from sklearn.covariance import LedoitWolf


def spectral_angle_mapper(spectrum1: np.ndarray, spectrum2: np.ndarray) -> float:
    """
    Calculate Spectral Angle Mapper (SAM) between two spectra
    
    Args:
        spectrum1: First spectrum
        spectrum2: Second spectrum
        
    Returns:
        SAM angle in radians (0 = identical, π/2 = orthogonal)
    """
    # Normalize spectra
    norm1 = np.linalg.norm(spectrum1)
    norm2 = np.linalg.norm(spectrum2)
    
    if norm1 == 0 or norm2 == 0:
        return np.pi / 2  # Maximum angle for zero vectors
    
    # Calculate cosine of angle
    cos_angle = np.dot(spectrum1, spectrum2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1, 1)  # Handle numerical errors
    
    return np.arccos(cos_angle)


def spectral_information_divergence(spectrum1: np.ndarray, spectrum2: np.ndarray, 
                                  epsilon: float = 1e-10) -> float:
    """
    Calculate Spectral Information Divergence (SID) between two spectra
    
    Args:
        spectrum1: First spectrum (reference)
        spectrum2: Second spectrum (target)
        epsilon: Small value to avoid log(0)
        
    Returns:
        SID value (0 = identical, higher = more different)
    """
    # Normalize to probability distributions
    p = spectrum1 + epsilon
    q = spectrum2 + epsilon
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Calculate KL divergences
    kl_pq = np.sum(p * np.log(p / q))
    kl_qp = np.sum(q * np.log(q / p))
    
    return kl_pq + kl_qp


def cosine_similarity(spectrum1: np.ndarray, spectrum2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two spectra
    
    Args:
        spectrum1: First spectrum
        spectrum2: Second spectrum
        
    Returns:
        Cosine similarity (1 = identical, 0 = orthogonal, -1 = opposite)
    """
    return 1 - cosine(spectrum1, spectrum2)


def match_spectrum_to_library(spectrum: np.ndarray, library_df: pd.DataFrame,
                             wavelengths: np.ndarray, metric: str = 'sid',
                             top_k: int = 3) -> List[Dict]:
    """
    Match a spectrum against a spectral library
    
    Args:
        spectrum: Input spectrum to match
        library_df: DataFrame with spectral library
        wavelengths: Wavelength array for input spectrum
        metric: Similarity metric ('sid', 'sam', 'cosine')
        top_k: Number of top matches to return
        
    Returns:
        List of match dictionaries with material, score, confidence
    """
    if library_df.empty:
        return []
    
    # Group library by material
    materials = library_df['material_name'].unique()
    matches = []
    
    for material in materials:
        material_data = library_df[library_df['material_name'] == material]
        
        # Interpolate library spectrum to input wavelengths
        lib_wavelengths = material_data['wavelength_nm'].values
        lib_reflectance = material_data['reflectance'].values
        
        # Simple linear interpolation
        interp_reflectance = np.interp(wavelengths, lib_wavelengths, lib_reflectance)
        
        # Calculate similarity
        if metric == 'sid':
            score = 1.0 / (1.0 + spectral_information_divergence(spectrum, interp_reflectance))
        elif metric == 'sam':
            sam_angle = spectral_angle_mapper(spectrum, interp_reflectance)
            score = 1.0 - (sam_angle / (np.pi / 2))  # Normalize to [0, 1]
        elif metric == 'cosine':
            score = cosine_similarity(spectrum, interp_reflectance)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Get material metadata
        category = material_data['category'].iloc[0] if 'category' in material_data.columns else 'unknown'
        confidence = material_data['confidence'].iloc[0] if 'confidence' in material_data.columns else 0.5
        
        matches.append({
            'material': material,
            'score': float(score),
            'confidence': float(confidence),
            'category': category,
            'metric': metric
        })
    
    # Sort by score and return top-k
    matches.sort(key=lambda x: x['score'], reverse=True)
    return matches[:top_k]


def reed_xiaoli_detector(data: np.ndarray, shrinkage: str = 'ledoit_wolf') -> np.ndarray:
    """
    Reed-Xiaoli (RX) anomaly detector for hyperspectral data
    
    Args:
        data: Hyperspectral data (height, width, bands)
        shrinkage: Covariance shrinkage method ('ledoit_wolf', 'none')
        
    Returns:
        Anomaly score map (height, width)
    """
    height, width, bands = data.shape
    
    # Reshape to (pixels, bands)
    pixels = data.reshape(-1, bands)
    
    # Remove invalid pixels
    valid_mask = ~np.any(np.isnan(pixels) | np.isinf(pixels), axis=1)
    valid_pixels = pixels[valid_mask]
    
    if len(valid_pixels) < bands:
        print("Warning: Not enough valid pixels for RX detector")
        return np.zeros((height, width))
    
    # Calculate mean spectrum
    mean_spectrum = np.mean(valid_pixels, axis=0)
    
    # Calculate covariance matrix with optional shrinkage
    if shrinkage == 'ledoit_wolf':
        lw = LedoitWolf()
        cov_matrix = lw.fit(valid_pixels).covariance_
    else:
        cov_matrix = np.cov(valid_pixels.T)
    
    # Regularize covariance matrix
    cov_matrix += np.eye(bands) * 1e-6
    
    try:
        # Invert covariance matrix
        inv_cov = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        print("Warning: Singular covariance matrix, using pseudo-inverse")
        inv_cov = np.linalg.pinv(cov_matrix)
    
    # Calculate RX scores for all pixels
    rx_scores = np.zeros(height * width)
    
    for i, pixel in enumerate(pixels):
        if valid_mask[i]:
            diff = pixel - mean_spectrum
            rx_scores[i] = np.dot(diff, np.dot(inv_cov, diff))
        else:
            rx_scores[i] = 0
    
    return rx_scores.reshape(height, width)


def normalize_spectrum(spectrum: np.ndarray, method: str = 'l2') -> np.ndarray:
    """
    Normalize spectrum using various methods
    
    Args:
        spectrum: Input spectrum
        method: Normalization method ('l2', 'l1', 'minmax', 'zscore')
        
    Returns:
        Normalized spectrum
    """
    if method == 'l2':
        norm = np.linalg.norm(spectrum)
        return spectrum / norm if norm > 0 else spectrum
    elif method == 'l1':
        norm = np.sum(np.abs(spectrum))
        return spectrum / norm if norm > 0 else spectrum
    elif method == 'minmax':
        min_val, max_val = np.min(spectrum), np.max(spectrum)
        if max_val > min_val:
            return (spectrum - min_val) / (max_val - min_val)
        else:
            return spectrum
    elif method == 'zscore':
        mean_val, std_val = np.mean(spectrum), np.std(spectrum)
        if std_val > 0:
            return (spectrum - mean_val) / std_val
        else:
            return spectrum - mean_val
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def remove_bad_bands(data: np.ndarray, wavelengths: np.ndarray, 
                    bad_bands: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove bad bands from hyperspectral data
    
    Args:
        data: Hyperspectral data (height, width, bands)
        wavelengths: Wavelength array
        bad_bands: List of band indices to remove
        
    Returns:
        Tuple of (cleaned_data, cleaned_wavelengths)
    """
    if not bad_bands:
        return data, wavelengths
    
    # Convert negative indices to positive
    n_bands = data.shape[2]
    bad_bands = [b if b >= 0 else n_bands + b for b in bad_bands]
    
    # Create mask for good bands
    good_bands = [i for i in range(n_bands) if i not in bad_bands]
    
    # Filter data and wavelengths
    cleaned_data = data[:, :, good_bands]
    cleaned_wavelengths = wavelengths[good_bands]
    
    return cleaned_data, cleaned_wavelengths


def calculate_spectral_indices(data: np.ndarray, wavelengths: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Calculate common spectral indices
    
    Args:
        data: Hyperspectral data (height, width, bands)
        wavelengths: Wavelength array in nm
        
    Returns:
        Dictionary of spectral indices
    """
    indices = {}
    
    # Find band indices for common wavelengths
    def find_band(target_wl):
        return np.argmin(np.abs(wavelengths - target_wl))
    
    try:
        # NDVI (Normalized Difference Vegetation Index)
        red_idx = find_band(670)
        nir_idx = find_band(800)
        red = data[:, :, red_idx]
        nir = data[:, :, nir_idx]
        indices['ndvi'] = (nir - red) / (nir + red + 1e-8)
        
        # NDWI (Normalized Difference Water Index)
        green_idx = find_band(560)
        nir_idx = find_band(1240)
        green = data[:, :, green_idx]
        nir = data[:, :, nir_idx]
        indices['ndwi'] = (green - nir) / (green + nir + 1e-8)
        
        # SAVI (Soil Adjusted Vegetation Index)
        L = 0.5  # Soil brightness correction factor
        red = data[:, :, find_band(670)]
        nir = data[:, :, find_band(800)]
        indices['savi'] = ((nir - red) / (nir + red + L)) * (1 + L)
        
        # EVI (Enhanced Vegetation Index)
        blue_idx = find_band(470)
        blue = data[:, :, blue_idx]
        indices['evi'] = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))
        
    except Exception as e:
        print(f"Warning: Could not calculate some spectral indices: {e}")
    
    return indices