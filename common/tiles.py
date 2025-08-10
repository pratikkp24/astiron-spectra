"""
Tiling utilities for Astiron Spectra
Handles splitting large images into tiles and stitching results back
"""

import numpy as np
from typing import List, Tuple, Dict, Iterator
from dataclasses import dataclass


@dataclass
class TileInfo:
    """Information about a tile"""
    row: int
    col: int
    y_start: int
    y_end: int
    x_start: int
    x_end: int
    height: int
    width: int
    overlap_top: int
    overlap_bottom: int
    overlap_left: int
    overlap_right: int


class TileManager:
    """Manages tiling and stitching operations"""
    
    def __init__(self, tile_size: int = 512, overlap: int = 64):
        self.tile_size = tile_size
        self.overlap = overlap
    
    def calculate_tiles(self, height: int, width: int) -> List[TileInfo]:
        """
        Calculate tile positions for given image dimensions
        
        Args:
            height: Image height
            width: Image width
            
        Returns:
            List of TileInfo objects
        """
        tiles = []
        
        # Calculate number of tiles needed
        effective_tile_size = self.tile_size - 2 * self.overlap
        n_rows = int(np.ceil(height / effective_tile_size))
        n_cols = int(np.ceil(width / effective_tile_size))
        
        for row in range(n_rows):
            for col in range(n_cols):
                # Calculate tile boundaries
                y_start = max(0, row * effective_tile_size - self.overlap)
                y_end = min(height, y_start + self.tile_size)
                
                x_start = max(0, col * effective_tile_size - self.overlap)
                x_end = min(width, x_start + self.tile_size)
                
                # Calculate actual overlap for this tile
                overlap_top = self.overlap if row > 0 else 0
                overlap_bottom = self.overlap if row < n_rows - 1 else 0
                overlap_left = self.overlap if col > 0 else 0
                overlap_right = self.overlap if col < n_cols - 1 else 0
                
                tile = TileInfo(
                    row=row,
                    col=col,
                    y_start=y_start,
                    y_end=y_end,
                    x_start=x_start,
                    x_end=x_end,
                    height=y_end - y_start,
                    width=x_end - x_start,
                    overlap_top=overlap_top,
                    overlap_bottom=overlap_bottom,
                    overlap_left=overlap_left,
                    overlap_right=overlap_right
                )
                
                tiles.append(tile)
        
        return tiles
    
    def extract_tile(self, data: np.ndarray, tile: TileInfo) -> np.ndarray:
        """
        Extract a tile from the input data
        
        Args:
            data: Input data array (2D or 3D)
            tile: Tile information
            
        Returns:
            Extracted tile data
        """
        if data.ndim == 2:
            return data[tile.y_start:tile.y_end, tile.x_start:tile.x_end]
        elif data.ndim == 3:
            return data[tile.y_start:tile.y_end, tile.x_start:tile.x_end, :]
        else:
            raise ValueError(f"Unsupported data dimensions: {data.shape}")
    
    def stitch_tiles(self, tile_results: List[Tuple[TileInfo, np.ndarray]], 
                     output_height: int, output_width: int) -> np.ndarray:
        """
        Stitch tile results back into full image
        
        Args:
            tile_results: List of (TileInfo, result_array) tuples
            output_height: Height of output image
            output_width: Width of output image
            
        Returns:
            Stitched result array
        """
        if not tile_results:
            raise ValueError("No tile results provided")
        
        # Determine output shape from first tile result
        first_result = tile_results[0][1]
        if first_result.ndim == 2:
            output_shape = (output_height, output_width)
        elif first_result.ndim == 3:
            output_shape = (output_height, output_width, first_result.shape[2])
        else:
            raise ValueError(f"Unsupported result dimensions: {first_result.shape}")
        
        # Initialize output arrays
        output = np.zeros(output_shape, dtype=first_result.dtype)
        weight_map = np.zeros((output_height, output_width), dtype=np.float32)
        
        for tile, result in tile_results:
            # Calculate effective region (excluding overlap)
            y_start_eff = tile.y_start + tile.overlap_top
            y_end_eff = tile.y_end - tile.overlap_bottom
            x_start_eff = tile.x_start + tile.overlap_left
            x_end_eff = tile.x_end - tile.overlap_right
            
            # Calculate corresponding indices in tile result
            tile_y_start = tile.overlap_top
            tile_y_end = tile.height - tile.overlap_bottom
            tile_x_start = tile.overlap_left
            tile_x_end = tile.width - tile.overlap_right
            
            # Extract effective region from tile result
            if result.ndim == 2:
                effective_result = result[tile_y_start:tile_y_end, tile_x_start:tile_x_end]
            else:
                effective_result = result[tile_y_start:tile_y_end, tile_x_start:tile_x_end, :]
            
            # Add to output with proper bounds checking
            y_start_out = max(0, y_start_eff)
            y_end_out = min(output_height, y_end_eff)
            x_start_out = max(0, x_start_eff)
            x_end_out = min(output_width, x_end_eff)
            
            if y_start_out < y_end_out and x_start_out < x_end_out:
                # Adjust effective result size if needed
                result_h = y_end_out - y_start_out
                result_w = x_end_out - x_start_out
                
                if result.ndim == 2:
                    output[y_start_out:y_end_out, x_start_out:x_end_out] += \
                        effective_result[:result_h, :result_w]
                else:
                    output[y_start_out:y_end_out, x_start_out:x_end_out, :] += \
                        effective_result[:result_h, :result_w, :]
                
                weight_map[y_start_out:y_end_out, x_start_out:x_end_out] += 1.0
        
        # Normalize by weight map to handle overlaps
        weight_map[weight_map == 0] = 1.0  # Avoid division by zero
        
        if output.ndim == 2:
            output = output / weight_map
        else:
            output = output / weight_map[:, :, np.newaxis]
        
        return output
    
    def process_in_tiles(self, data: np.ndarray, process_func, 
                        min_valid_pixels: float = 0.8, **kwargs) -> np.ndarray:
        """
        Process data in tiles using provided function
        
        Args:
            data: Input data array
            process_func: Function to apply to each tile
            min_valid_pixels: Minimum fraction of valid pixels required
            **kwargs: Additional arguments for process_func
            
        Returns:
            Processed result array
        """
        height, width = data.shape[:2]
        tiles = self.calculate_tiles(height, width)
        
        tile_results = []
        
        for tile in tiles:
            # Extract tile
            tile_data = self.extract_tile(data, tile)
            
            # Check if tile has enough valid pixels
            if data.ndim == 2:
                valid_pixels = np.sum(~np.isnan(tile_data))
                total_pixels = tile_data.size
            else:
                # For 3D data, check first band
                valid_pixels = np.sum(~np.isnan(tile_data[:, :, 0]))
                total_pixels = tile_data.shape[0] * tile_data.shape[1]
            
            valid_fraction = valid_pixels / total_pixels
            
            if valid_fraction >= min_valid_pixels:
                # Process tile
                try:
                    result = process_func(tile_data, **kwargs)
                    tile_results.append((tile, result))
                except Exception as e:
                    print(f"Warning: Failed to process tile {tile.row},{tile.col}: {e}")
                    # Create empty result with same shape as expected
                    if hasattr(process_func, '__name__') and 'detect' in process_func.__name__:
                        # For detection functions, return zeros
                        result = np.zeros((tile.height, tile.width), dtype=np.float32)
                    else:
                        # For other functions, return input
                        result = tile_data
                    tile_results.append((tile, result))
            else:
                print(f"Skipping tile {tile.row},{tile.col}: insufficient valid pixels ({valid_fraction:.2f})")
                # Create empty result
                result = np.zeros((tile.height, tile.width), dtype=np.float32)
                tile_results.append((tile, result))
        
        # Stitch results
        if tile_results:
            return self.stitch_tiles(tile_results, height, width)
        else:
            # Return zeros if no tiles processed
            return np.zeros_like(data[:, :, 0] if data.ndim == 3 else data)


def create_tile_generator(data: np.ndarray, tile_size: int = 512, 
                         overlap: int = 64) -> Iterator[Tuple[TileInfo, np.ndarray]]:
    """
    Generator that yields tiles from input data
    
    Args:
        data: Input data array
        tile_size: Size of tiles
        overlap: Overlap between tiles
        
    Yields:
        Tuple of (TileInfo, tile_data)
    """
    tile_manager = TileManager(tile_size, overlap)
    height, width = data.shape[:2]
    tiles = tile_manager.calculate_tiles(height, width)
    
    for tile in tiles:
        tile_data = tile_manager.extract_tile(data, tile)
        yield tile, tile_data