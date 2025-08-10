"""
Thermal infrared processing utilities for Astiron Spectra
Handles thermal data processing, temperature conversion, and TIR-specific operations
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List
from scipy import ndimage
from sklearn.cluster import DBSCAN


def brightness_temperature_to_celsius(bt_data: np.ndarray, 
                                    emissivity: float = 0.95) -> np.ndarray:
    """
    Convert brightness temperature to surface temperature in Celsius
    
    Args:
        bt_data: Brightness temperature data in Kelvin
        emissivity: Surface emissivity (0-1)
        
    Returns:
        Surface temperature in Celsius
    """
    # Stefan-Boltzmann constant
    sigma = 5.670374419e-8  # W⋅m⁻²⋅K⁻⁴
    
    # Convert brightness temperature to radiance
    # Then apply emissivity correction
    # This is a simplified approach - more sophisticated methods exist
    
    # Ensure emissivity is valid
    emissivity = np.clip(emissivity, 0.1, 1.0)
    
    # Apply emissivity correction (simplified)
    corrected_temp = bt_data * (emissivity ** 0.25)
    
    # Convert to Celsius
    celsius_temp = corrected_temp - 273.15
    
    return celsius_temp


def detect_thermal_anomalies_cnn_patch(thermal_data: np.ndarray, 
                                     patch_size: int = 64,
                                     stride: int = 32,
                                     threshold_method: str = 'adaptive') -> np.ndarray:
    """
    Detect thermal anomalies using CNN-based patch analysis
    
    Args:
        thermal_data: Thermal infrared data (2D array)
        patch_size: Size of patches for analysis
        stride: Stride for patch extraction
        threshold_method: Thresholding method ('adaptive', 'otsu', 'percentile')
        
    Returns:
        Anomaly score map
    """
    height, width = thermal_data.shape
    
    # Initialize anomaly score map
    anomaly_scores = np.zeros_like(thermal_data, dtype=np.float32)
    weight_map = np.zeros_like(thermal_data, dtype=np.float32)
    
    # Extract patches and compute local statistics
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = thermal_data[y:y+patch_size, x:x+patch_size]
            
            # Skip patches with too many invalid pixels
            valid_pixels = ~np.isnan(patch)
            if np.sum(valid_pixels) < patch_size * patch_size * 0.5:
                continue
            
            # Compute local statistics
            patch_mean = np.nanmean(patch)
            patch_std = np.nanstd(patch)
            
            if patch_std == 0:
                continue
            
            # Compute anomaly scores for this patch
            patch_scores = np.abs(patch - patch_mean) / patch_std
            
            # Add to global score map
            y_end = min(y + patch_size, height)
            x_end = min(x + patch_size, width)
            
            anomaly_scores[y:y_end, x:x_end] += patch_scores[:y_end-y, :x_end-x]
            weight_map[y:y_end, x:x_end] += 1.0
    
    # Normalize by weight map
    weight_map[weight_map == 0] = 1.0
    anomaly_scores = anomaly_scores / weight_map
    
    return anomaly_scores


def detect_hot_spots(thermal_data: np.ndarray, 
                    sigma: float = 2.0,
                    min_area: int = 4) -> np.ndarray:
    """
    Detect hot spots in thermal data using statistical thresholding
    
    Args:
        thermal_data: Thermal infrared data (2D array)
        sigma: Number of standard deviations above mean for threshold
        min_area: Minimum area (pixels) for hot spot detection
        
    Returns:
        Binary mask of hot spots
    """
    # Remove invalid pixels
    valid_mask = ~np.isnan(thermal_data)
    valid_data = thermal_data[valid_mask]
    
    if len(valid_data) == 0:
        return np.zeros_like(thermal_data, dtype=bool)
    
    # Compute statistics
    mean_temp = np.mean(valid_data)
    std_temp = np.std(valid_data)
    
    # Threshold for hot spots
    threshold = mean_temp + sigma * std_temp
    
    # Create binary mask
    hot_spots = thermal_data > threshold
    hot_spots = hot_spots & valid_mask
    
    # Remove small regions
    if min_area > 1:
        # Use morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        hot_spots = cv2.morphologyEx(hot_spots.astype(np.uint8), 
                                   cv2.MORPH_OPEN, kernel)
        
        # Remove small connected components
        labeled, num_features = ndimage.label(hot_spots)
        for i in range(1, num_features + 1):
            component = labeled == i
            if np.sum(component) < min_area:
                hot_spots[component] = 0
    
    return hot_spots.astype(bool)


def temporal_median_filter(thermal_sequence: np.ndarray, 
                         window_size: int = 5) -> np.ndarray:
    """
    Apply temporal median filter to thermal image sequence
    
    Args:
        thermal_sequence: Thermal data sequence (time, height, width)
        window_size: Size of temporal window for filtering
        
    Returns:
        Filtered thermal sequence
    """
    n_frames, height, width = thermal_sequence.shape
    filtered_sequence = np.zeros_like(thermal_sequence)
    
    half_window = window_size // 2
    
    for t in range(n_frames):
        # Define temporal window
        t_start = max(0, t - half_window)
        t_end = min(n_frames, t + half_window + 1)
        
        # Extract temporal window
        window = thermal_sequence[t_start:t_end, :, :]
        
        # Apply median filter
        filtered_sequence[t, :, :] = np.nanmedian(window, axis=0)
    
    return filtered_sequence


def denoise_thermal_bilateral(thermal_data: np.ndarray,
                            d: int = 9,
                            sigma_color: float = 75,
                            sigma_space: float = 75) -> np.ndarray:
    """
    Denoise thermal data using bilateral filtering
    
    Args:
        thermal_data: Thermal infrared data (2D array)
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in color space
        sigma_space: Filter sigma in coordinate space
        
    Returns:
        Denoised thermal data
    """
    # Handle NaN values
    valid_mask = ~np.isnan(thermal_data)
    
    if not np.any(valid_mask):
        return thermal_data
    
    # Normalize data to 0-255 for OpenCV
    valid_data = thermal_data[valid_mask]
    min_val, max_val = np.nanmin(valid_data), np.nanmax(valid_data)
    
    if max_val == min_val:
        return thermal_data
    
    normalized = ((thermal_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    
    # Apply bilateral filter
    denoised_normalized = cv2.bilateralFilter(normalized, d, sigma_color, sigma_space)
    
    # Convert back to original scale
    denoised = denoised_normalized.astype(np.float32) / 255.0 * (max_val - min_val) + min_val
    
    # Restore NaN values
    denoised[~valid_mask] = np.nan
    
    return denoised


def extract_video_frames(video_path: str, output_dir: str, 
                        fps: float = 2.0) -> List[str]:
    """
    Extract frames from thermal video at specified FPS
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        fps: Target frames per second for extraction
        
    Returns:
        List of extracted frame file paths
    """
    import os
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame interval
    frame_interval = max(1, int(video_fps / fps))
    
    extracted_files = []
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Extract frame at specified interval
        if frame_count % frame_interval == 0:
            # Convert to grayscale if needed (thermal videos are often single channel)
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Save frame
            frame_filename = f"frame_{extracted_count:06d}.tif"
            frame_path = output_path / frame_filename
            
            cv2.imwrite(str(frame_path), frame)
            extracted_files.append(str(frame_path))
            extracted_count += 1
        
        frame_count += 1
    
    cap.release()
    
    print(f"Extracted {extracted_count} frames from {total_frames} total frames")
    return extracted_files


def estimate_emissivity(thermal_data: np.ndarray, 
                       material_map: np.ndarray,
                       emissivity_lib: Dict[str, float]) -> np.ndarray:
    """
    Estimate emissivity map based on material classification
    
    Args:
        thermal_data: Thermal infrared data
        material_map: Material classification map (integer labels)
        emissivity_lib: Dictionary mapping material IDs to emissivity values
        
    Returns:
        Emissivity map
    """
    emissivity_map = np.ones_like(thermal_data) * 0.95  # Default emissivity
    
    # Apply material-specific emissivities
    for material_id, emissivity in emissivity_lib.items():
        if isinstance(material_id, str):
            continue  # Skip string keys
        
        mask = material_map == material_id
        emissivity_map[mask] = emissivity
    
    return emissivity_map


def calculate_temperature_difference(thermal_data: np.ndarray,
                                   background_percentile: float = 10) -> np.ndarray:
    """
    Calculate temperature difference from local background
    
    Args:
        thermal_data: Thermal infrared data
        background_percentile: Percentile to use for background temperature
        
    Returns:
        Temperature difference map
    """
    # Calculate background temperature
    valid_data = thermal_data[~np.isnan(thermal_data)]
    
    if len(valid_data) == 0:
        return np.zeros_like(thermal_data)
    
    background_temp = np.percentile(valid_data, background_percentile)
    
    # Calculate difference
    temp_diff = thermal_data - background_temp
    
    return temp_diff


def cluster_thermal_anomalies(anomaly_scores: np.ndarray,
                            threshold: float = 0.5,
                            eps: float = 3.0,
                            min_samples: int = 5) -> np.ndarray:
    """
    Cluster thermal anomalies using DBSCAN
    
    Args:
        anomaly_scores: Anomaly score map
        threshold: Threshold for anomaly detection
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN minimum samples parameter
        
    Returns:
        Clustered anomaly labels
    """
    # Threshold anomaly scores
    anomaly_mask = anomaly_scores > threshold
    
    # Get anomaly pixel coordinates
    anomaly_coords = np.column_stack(np.where(anomaly_mask))
    
    if len(anomaly_coords) == 0:
        return np.zeros_like(anomaly_scores, dtype=int)
    
    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = clustering.fit_predict(anomaly_coords)
    
    # Create output map
    clustered_map = np.zeros_like(anomaly_scores, dtype=int)
    
    for i, (y, x) in enumerate(anomaly_coords):
        clustered_map[y, x] = cluster_labels[i] + 1  # +1 to avoid 0 (background)
    
    return clustered_map