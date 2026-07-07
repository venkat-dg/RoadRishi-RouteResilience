import os
import numpy as np
from PIL import Image

class PreprocessPipeline:
    """
    RoadRishi Ingestion & Preprocessing Pipeline.
    Simulates dual Sentinel-2 / LISS-IV bands parsing, performs NDVI synthesis,
    normalizes features, and extracts overlapping 512x512 tile matrices.
    """
    def __init__(self, tile_size=512, stride=128):
        self.tile_size = tile_size
        self.stride = stride

    def load_and_parse_bands(self, image_path):
        """
        Loads an RGB image and parses it into spectrally equivalent bands:
        - B2/B3: Green
        - B3/B4: Red
        - B4/B8: NIR (Simulated using green channel chlorophyll-reflectance model)
        """
        img = Image.open(image_path).convert("RGB")
        arr = np.array(img, dtype=np.float32)
        
        # Extract Green and Red bands (normalized to 0-255 range)
        green = arr[:, :, 1]
        red = arr[:, :, 0]
        
        # Simulate NIR (Near-Infrared): high in vegetation (high green, low red),
        # low on roads and buildings.
        # We model NIR = Green * 1.3 - Red * 0.3 + 20
        nir = green * 1.3 - red * 0.3 + 20.0
        nir = np.clip(nir, 0.0, 255.0)
        
        return green, red, nir

    def synthesize_ndvi(self, red, nir):
        """
        Calculates Normalized Difference Vegetation Index (NDVI).
        NDVI = (NIR - Red) / (NIR + Red)
        Useful as an explicit prior to isolate impervious road surfaces (NDVI ~0.1)
        from occluding vegetation canopy (NDVI ~0.6-0.9).
        """
        denominator = nir + red
        # Prevent division by zero
        denominator[denominator == 0] = 1e-5
        ndvi = (nir - red) / denominator
        
        # NDVI values are theoretically between -1.0 and 1.0.
        # Clip to ensure valid values
        return np.clip(ndvi, -1.0, 1.0)

    def normalize_min_max(self, band):
        """Scale band values to [0, 1] range."""
        min_val = np.min(band)
        max_val = np.max(band)
        if max_val - min_val == 0:
            return np.zeros_like(band)
        return (band - min_val) / (max_val - min_val)

    def process(self, image_path):
        """
        Executes end-to-end preprocessing:
        1. Ingest image and parse Green, Red, NIR channels.
        2. Synthesize NDVI channel.
        3. Normalize each channel to [0, 1].
        4. Assemble a 4-channel tensor [Green, Red, NIR, NDVI] of shape (H, W, 4).
        5. Extract coordinates for overlapping 512x512 tile slicing.
        """
        green, red, nir = self.load_and_parse_bands(image_path)
        ndvi = self.synthesize_ndvi(red, nir)
        
        # Normalize
        green_norm = self.normalize_min_max(green)
        red_norm = self.normalize_min_max(red)
        nir_norm = self.normalize_min_max(nir)
        # Scale NDVI from [-1, 1] to [0, 1] for uniformity
        ndvi_norm = (ndvi + 1.0) / 2.0
        
        # Convergence: Stack into a 4-channel array (H, W, 4)
        tensor_4ch = np.stack([green_norm, red_norm, nir_norm, ndvi_norm], axis=-1)
        
        h, w, c = tensor_4ch.shape
        
        # Calculate tile offsets
        tiles = []
        y = 0
        while y + self.tile_size <= h or (y == 0 and h < self.tile_size):
            y_start = y
            y_end = min(y + self.tile_size, h)
            
            x = 0
            while x + self.tile_size <= w or (x == 0 and w < self.tile_size):
                x_start = x
                x_end = min(x + self.tile_size, w)
                
                # Slice tile
                tile_data = tensor_4ch[y_start:y_end, x_start:x_end]
                
                # Pad tile if it is smaller than tile_size (edge handling)
                if tile_data.shape[0] < self.tile_size or tile_data.shape[1] < self.tile_size:
                    pad_y = self.tile_size - tile_data.shape[0]
                    pad_x = self.tile_size - tile_data.shape[1]
                    tile_data = np.pad(tile_data, ((0, pad_y), (0, pad_x), (0, 0)), mode="edge")
                
                tiles.append({
                    "x": x_start,
                    "y": y_start,
                    "data": tile_data
                })
                
                if x + self.tile_size >= w:
                    break
                x += self.stride
                
            if y + self.tile_size >= h:
                break
            y += self.stride

        return {
            "original_shape": (h, w),
            "tensor_4ch": tensor_4ch,
            "tiles": tiles,
            "raw_bands": {
                "green": green,
                "red": red,
                "nir": nir,
                "ndvi": ndvi
            }
        }
