import json
import os
import shutil
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

# Import the tiling function
from tile_image import tile_image


class TestImageTiler(unittest.TestCase):
    def setUp(self):
        # Create a workspace-local directory for test outputs and inputs
        self.scratch_dir = Path(__file__).parent
        self.test_io_dir = self.scratch_dir / "test_scratch"
        self.test_io_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths for test images
        self.rgb_img_path = self.test_io_dir / "test_rgb.png"
        self.gray_img_path = self.test_io_dir / "test_gray.png"
        self.out_dir = self.test_io_dir / "tiles_out"
        
        # Create standard test images
        # 1. RGB Image (300 x 300)
        rgb_data = np.zeros((300, 300, 3), dtype=np.uint8)
        # Add some patterns so we can identify content
        for i in range(300):
            rgb_data[i, :, 0] = i % 256  # Red gradient
            rgb_data[:, i, 1] = i % 256  # Green gradient
            rgb_data[i, i, 2] = 255      # Blue diagonal line
        Image.fromarray(rgb_data).save(self.rgb_img_path)
        
        # 2. Grayscale Image (100 x 150)
        gray_data = np.zeros((100, 150), dtype=np.uint8)
        gray_data[20:80, 20:130] = 128
        Image.fromarray(gray_data).save(self.gray_img_path)

    def tearDown(self):
        # Clean up all created test inputs and outputs
        if self.test_io_dir.exists():
            shutil.rmtree(self.test_io_dir)

    def test_no_pad_cropping(self):
        """Test tiling without padding (should drop incomplete edge tiles)."""
        # Image is 300x300, tile_size 128, stride 128
        # Fits:
        # x offsets: 0, 128. Next is 256 (256 + 128 = 384 > 300) -> dropped.
        # y offsets: 0, 128. Next is 256 (256 + 128 = 384 > 300) -> dropped.
        # Total tiles: 2 x 2 = 4
        num_tiles = tile_image(
            image_path=str(self.rgb_img_path),
            tile_size=128,
            stride=128,
            out_dir=str(self.out_dir),
            pad=False,
            meta_format="json",
            quiet=True
        )
        self.assertEqual(num_tiles, 4)
        
        # Check files exist
        tile_files = list(self.out_dir.glob("test_rgb_y*_x*.png"))
        self.assertEqual(len(tile_files), 4)
        
        # Check metadata file
        meta_path = self.out_dir / "metadata.json"
        self.assertTrue(meta_path.exists())
        with open(meta_path, "r") as f:
            meta = json.load(f)
        self.assertEqual(len(meta), 4)
        self.assertEqual(meta[0]["width"], 128)
        self.assertEqual(meta[0]["height"], 128)
        self.assertFalse(meta[0]["is_padded"])

    def test_with_padding_constant(self):
        """Test tiling with constant padding."""
        # Image is 300x300, tile_size 128, stride 128
        # With padding:
        # x offsets: 0, 128, 256. (256 + 128 = 384 > 300, so 256 is padded)
        # y offsets: 0, 128, 256. (256 + 128 = 384 > 300, so 256 is padded)
        # Total tiles: 3 x 3 = 9
        num_tiles = tile_image(
            image_path=str(self.rgb_img_path),
            tile_size=128,
            stride=128,
            out_dir=str(self.out_dir),
            pad=True,
            pad_mode="constant",
            pad_val=0,
            meta_format="json",
            quiet=True
        )
        self.assertEqual(num_tiles, 9)
        
        # Check files
        tile_files = list(self.out_dir.glob("test_rgb_y*_x*.png"))
        self.assertEqual(len(tile_files), 9)
        
        # Check metadata file
        meta_path = self.out_dir / "metadata.json"
        self.assertTrue(meta_path.exists())
        with open(meta_path, "r") as f:
            meta = json.load(f)
            
        padded_tiles = [m for m in meta if m["is_padded"]]
        self.assertEqual(len(padded_tiles), 5)  # columns at 256 (3), rows at 256 (3) -> overlap at (256, 256) is 1. 3 + 3 - 1 = 5.
        
        # Verify a padded tile has correct dimensions and constant padding value
        padded_tile_path = self.out_dir / "test_rgb_y256_x256.png"
        self.assertTrue(padded_tile_path.exists())
        with Image.open(padded_tile_path) as tile:
            self.assertEqual(tile.size, (128, 128))
            arr = np.array(tile)
            # The padded region in x starts from (300 - 256) = 44px onwards
            # The padded region in y starts from (300 - 256) = 44px onwards
            # Check if padded area is indeed 0
            self.assertTrue(np.all(arr[44:, :, :] == 0))
            self.assertTrue(np.all(arr[:, 44:, :] == 0))

    def test_non_square_tiles_and_stride(self):
        """Test tiling with non-square tiles and stride sizes."""
        # Grayscale image: 100x150 (height x width)
        # Tile size: 60x80 (height x width)
        # Stride: 40x50 (stride_y x stride_x)
        # Without padding:
        # y indices: y=0 (0+60=60<=100), y=40 (40+60=100<=100), y=80 (80+60=140 > 100 -> Out of bounds)
        # x indices: x=0 (0+80=80<=150), x=50 (50+80=130<=150), x=100 (100+80=180 > 150 -> Out of bounds)
        # Total tiles: 2 * 2 = 4
        num_tiles = tile_image(
            image_path=str(self.gray_img_path),
            tile_size=(60, 80),  # height, width
            stride=(40, 50),   # stride_y, stride_x
            out_dir=str(self.out_dir),
            pad=False,
            meta_format="csv",
            quiet=True
        )
        self.assertEqual(num_tiles, 4)
        
        # Verify CSV metadata
        meta_path = self.out_dir / "metadata.csv"
        self.assertTrue(meta_path.exists())
        with open(meta_path, "r") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 5)  # header + 4 rows

    def test_different_pad_modes(self):
        """Test different padding modes like reflect and edge."""
        num_tiles = tile_image(
            image_path=str(self.rgb_img_path),
            tile_size=200,
            stride=150,
            out_dir=str(self.out_dir),
            pad=True,
            pad_mode="reflect",
            quiet=True
        )
        self.assertEqual(num_tiles, 4)  # (0, 150) x (0, 150) -> 2 x 2
        
        padded_tile_path = self.out_dir / "test_rgb_y150_x150.png"
        self.assertTrue(padded_tile_path.exists())
        with Image.open(padded_tile_path) as tile:
            self.assertEqual(tile.size, (200, 200))


if __name__ == "__main__":
    unittest.main()
