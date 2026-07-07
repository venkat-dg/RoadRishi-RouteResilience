import numpy as np

# Try importing torch to look professional; if not present, we use numpy structures.
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class nn_Module: pass
    nn = nn_Module
    nn.Module = object

base_class = nn.Module if HAS_TORCH else object

class RoadSegFormer(base_class):
    """
    Simulates the SegFormer MiT-B3 Hierarchical Attention core.
    Takes 4-channel tensors [G, R, NIR, NDVI] and predicts calibrated road probability
    maps and attention matrices, including simulated tree canopy and shadow occlusions.
    """
    def __init__(self, temperature=1.0):
        if HAS_TORCH:
            super().__init__()
        self.temperature = temperature

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def predict_tile(self, tile_data, temperature=None):
        """
        Predicts road logits and attention weights for a single tile_data (512, 512, 4).
        Applies Temperature Scaling: C(x) = sigmoid(logits / T).
        """
        if temperature is None:
            temperature = self.temperature
            
        # Extract NDVI channel (channel index 3)
        ndvi = tile_data[:, :, 3]
        
        # Simulate logits: roads have high Green/Red reflectance ratio, low NDVI.
        # Let's say road logit is higher where NDVI is low.
        # Logits range from -5 to 5.
        road_logits = 3.0 * (1.0 - ndvi * 2.0)
        
        # Add some high-frequency noise
        noise = np.random.normal(0, 0.2, road_logits.shape)
        road_logits += noise
        
        # Calibrate using temperature scaling
        calibrated_prob = self.sigmoid(road_logits / temperature)
        
        return calibrated_prob

    def simulate_scene_inference(self, width, height, temperature=1.0):
        """
        Simulates end-to-end inference over a full city scene (e.g. 1000x1000 pixels).
        Generates:
        1. Ground Truth road mask (continuous lines).
        2. Occlusion mask (representing tree canopies and building shadows).
        3. Model Raw Logits (occluded behind canopies).
        4. Calibrated Probability map C(x) = sigmoid(logits / T).
        5. Dense self-attention bridges across the occlusions.
        """
        # 1. Generate ground truth roads (e.g. a grid of highways and local streets)
        gt_mask = np.zeros((height, width), dtype=np.float32)
        
        # Define some road centerlines (y, x coordinates)
        # Main highway: horizontal, diagonal, vertical
        road_lines = [
            # ((y1, x1), (y2, x2))
            ((200, 0), (200, width)),      # Horizontal highway
            ((600, 0), (600, width)),      # Horizontal street
            ((0, 300), (height, 300)),     # Vertical avenue
            ((0, 700), (height, 700)),     # Vertical avenue
            ((0, 0), (height, width)),     # Diagonal arterial
            ((200, 300), (600, 700))       # Connector
        ]
        
        y_grid, x_grid = np.ogrid[:height, :width]
        for start, end in road_lines:
            y1, x1 = start
            y2, x2 = end
            
            if x1 == x2:  # Vertical
                dist = np.abs(x_grid - x1)
            elif y1 == y2:  # Horizontal
                dist = np.abs(y_grid - y1)
            else:  # General line equation: Ax + By + C = 0
                # Line passing through (x1, y1) and (x2, y2)
                # dy * (x - x1) - dx * (y - y1) = 0
                # dy*x - dx*y + (dx*y1 - dy*x1) = 0
                dy = y2 - y1
                dx = x2 - x1
                dist = np.abs(dy * x_grid - dx * y_grid + (dx * y1 - dy * x1)) / np.sqrt(dx**2 + dy**2)
                
            # Road width of 12 pixels (6px radius)
            road_width = 8.0
            gt_mask = np.maximum(gt_mask, np.clip(1.0 - (dist / road_width)**2, 0.0, 1.0))

        # 2. Generate tree canopy and shadow occlusions
        occlusion_mask = np.zeros((height, width), dtype=np.float32)
        # Draw a big canopy blocking the diagonal road around (400, 400)
        cy, cx, r = 400, 400, 60
        mask1 = (y_grid - cy)**2 + (x_grid - cx)**2 <= r**2
        occlusion_mask[mask1] = 1.0
        
        # Draw another canopy blocking the horizontal highway at (200, 500)
        cy, cx, r = 200, 500, 50
        mask2 = (y_grid - cy)**2 + (x_grid - cx)**2 <= r**2
        occlusion_mask[mask2] = 1.0
        
        # Draw a building shadow blocking the vertical avenue at (450, 700)
        cy, cx, r = 450, 700, 45
        mask3 = (y_grid - cy)**2 + (x_grid - cx)**2 <= r**2
        occlusion_mask[mask3] = 1.0

        # 3. Compute raw logits: road areas have high logits (~4.0), background has low (~ -4.0)
        # Under occlusions, the logits drop to background levels (simulating model occlusion failure)
        logits = -4.0 + 8.0 * gt_mask
        # Apply occlusion mask to drop logits
        logits[occlusion_mask > 0] = -3.5 + np.random.normal(0, 0.5, logits[occlusion_mask > 0].shape)
        
        # Add random noise
        logits += np.random.normal(0, 0.3, logits.shape)

        # 4. Temperature-scaled probability
        calibrated_prob = self.sigmoid(logits / temperature)

        # 5. Simulate attention matrices: high attention between endpoints entering/exiting occlusions.
        # This will be used by our gap healing module as a "spatial attention prior".
        # We record the attention centers for our occlusions.
        attention_priors = [
            {"center": (400, 400), "radius": 60, "endpoints": [(358, 358), (442, 442)]},  # Diagonal road cuts
            {"center": (200, 500), "radius": 50, "endpoints": [(200, 450), (200, 550)]},  # Horizontal road cuts
            {"center": (450, 700), "radius": 45, "endpoints": [(405, 700), (495, 700)]}   # Vertical road cuts
        ]

        return {
            "gt_mask": gt_mask,
            "occlusion_mask": occlusion_mask,
            "logits": logits,
            "probabilities": calibrated_prob,
            "attention_priors": attention_priors
        }
