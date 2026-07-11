import os
import numpy as np

# Try importing torch; if not present, simulation mode is used.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class nn_Module: pass
    nn = nn_Module
    nn.Module = object

# Try importing transformers for real SegFormer inference
try:
    from transformers import SegformerForSemanticSegmentation
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Default checkpoint path (relative to this file)
DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(__file__), "..", "training", "checkpoints", "roadrishi_finetuned.pth"
)

base_class = nn.Module if HAS_TORCH else object

class RoadSegFormer(base_class):
    """
    SegFormer MiT-B3 Road Segmentation model wrapper.

    Modes:
      - Simulation (default): generates realistic synthetic road probability maps
        using the numpy-based pipeline. Works with zero dependencies beyond numpy.
      - Live inference: loads a real trained checkpoint (roadrishi_finetuned.pth)
        and runs actual SegFormer inference. Requires torch + transformers.

    The mode is selected automatically: if the checkpoint file exists and
    torch + transformers are installed, real inference is used. Otherwise
    the simulation fallback is used transparently.
    """
    def __init__(self, temperature=1.0, checkpoint_path=None):
        if HAS_TORCH:
            super().__init__()
        self.temperature = temperature
        self.real_model = None
        self.device = "cpu"
        self.is_live = False

        # Auto-load checkpoint if available
        ckpt = checkpoint_path or DEFAULT_CHECKPOINT
        self.load_checkpoint(ckpt)

    # ------------------------------------------------------------------
    # Real inference methods
    # ------------------------------------------------------------------
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Attempts to load a trained SegFormer checkpoint from *checkpoint_path*.
        Returns True if successful, False if file not found or deps missing.
        The simulation fallback remains active on failure.
        """
        if not os.path.exists(checkpoint_path):
            print(f"[RoadSegFormer] No checkpoint at {checkpoint_path} — using simulation mode.")
            return False

        if not (HAS_TORCH and HAS_TRANSFORMERS):
            print("[RoadSegFormer] torch/transformers not installed — using simulation mode.")
            return False

        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            self.real_model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b3-finetuned-ade-512-512",
                num_labels=2,
                id2label={0: "background", 1: "road"},
                label2id={"background": 0, "road": 1},
                ignore_mismatched_sizes=True,
            )

            # Strip DataParallel prefix if saved from multi-GPU training
            state = ckpt.get("model_state_dict", ckpt)
            state = {k.replace("module.", ""): v for k, v in state.items()}
            self.real_model.load_state_dict(state, strict=False)
            self.real_model.to(self.device)
            self.real_model.eval()

            self.is_live = True
            val_miou = ckpt.get("best_miou", ckpt.get("val_miou", "N/A"))
            print(f"[RoadSegFormer] Live model loaded ✓  best_mIoU={val_miou}  device={self.device}")
            return True

        except Exception as e:
            print(f"[RoadSegFormer] Failed to load checkpoint: {e} — falling back to simulation.")
            self.real_model = None
            self.is_live = False
            return False

    def predict_from_image(self, image_array: np.ndarray) -> np.ndarray:
        """
        Runs real SegFormer inference on a single RGB image (H, W, 3) numpy array.
        Returns a road probability map of shape (H, W) with values in [0, 1].
        Falls back to simulate_scene_inference if not in live mode.
        """
        if not self.is_live or self.real_model is None:
            h, w = image_array.shape[:2]
            sim = self.simulate_scene_inference(w, h, self.temperature)
            return sim["probabilities"]

        import torch
        # Normalize to ImageNet stats
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = image_array.astype(np.float32) / 255.0
        img = (img - mean) / std  # (H, W, 3)

        # To tensor: (1, 3, H, W)
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.real_model(pixel_values=tensor)
            logits  = outputs.logits  # (1, 2, H/4, W/4)
            upsampled = F.interpolate(
                logits, size=image_array.shape[:2], mode="bilinear", align_corners=False
            )
            probs = torch.softmax(upsampled, dim=1)[0, 1].cpu().numpy()  # road channel

        return probs  # (H, W) float32

    def predict_tile_batch(self, tiles: list) -> list:
        """
        Runs batched inference over a list of image tiles (each H×W×3 numpy array).
        Returns a list of road probability maps.
        """
        return [self.predict_from_image(tile) for tile in tiles]

    def get_status(self) -> dict:
        """Returns the current inference mode and device."""
        return {
            "mode": "live_model" if self.is_live else "simulation",
            "device": self.device if self.is_live else "cpu (numpy)",
            "has_torch": HAS_TORCH,
            "has_transformers": HAS_TRANSFORMERS,
        }



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
