import os
import json
from pathlib import Path
import numpy as np
from PIL import Image

# Import the tiling function
from tile_image import tile_image, HAS_RASTERIO

def main():
    print("=" * 60)
    print("           Image Tiling Script Demo Run")
    print("=" * 60)
    
    script_dir = Path(__file__).parent
    output_dir = script_dir / "demo_output"
    
    # 1. Create a beautiful synthetic RGB image (512x512)
    # We will generate a nice color grid with shapes so tiles are visually distinct.
    img_size = 512
    arr = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
    # Fill with color gradient
    for y in range(img_size):
        for x in range(img_size):
            arr[y, x, 0] = int(y / img_size * 255)  # Red increases vertically
            arr[y, x, 1] = int(x / img_size * 255)  # Green increases horizontally
            arr[y, x, 2] = 128                      # Constant blue
            
    # Draw some shapes: a white circle in the middle
    cy, cx, r = 256, 256, 100
    y_indices, x_indices = np.ogrid[:img_size, :img_size]
    mask = (y_indices - cy)**2 + (x_indices - cx)**2 <= r**2
    arr[mask] = [255, 255, 255]
    
    # Draw a grid pattern
    arr[::64, :, :] = 50
    arr[:, ::64, :] = 50

    src_img_path = script_dir / "demo_source.png"
    Image.fromarray(arr).save(src_img_path)
    print(f"Created standard sample image at: {src_img_path}")
    
    # 2. Run standard tiling
    print("\n--- Running Standard Tiling (Tile Size: 200x200, Stride: 150x150, Pad: True) ---")
    tile_dir = output_dir / "standard_tiles"
    count = tile_image(
        image_path=str(src_img_path),
        tile_size=200,
        stride=150,
        out_dir=str(tile_dir),
        pad=True,
        pad_mode="constant",
        pad_val=0,
        img_format="png",
        parallel=True,
        meta_format="json"
    )
    
    # Read and print a subset of the metadata
    meta_file = tile_dir / "metadata.json"
    if meta_file.exists():
        with open(meta_file, "r") as f:
            metadata = json.load(f)
        print(f"Successfully generated {count} tiles. Sample metadata entry:")
        print(json.dumps(metadata[0], indent=2))
        
    # 3. If rasterio is installed, demonstrate geospatial tiling
    if HAS_RASTERIO:
        print("\n--- Running Geospatial Tiling ---")
        import rasterio
        from rasterio.transform import from_origin
        
        # Create a mock GeoTIFF (WGS84, centered over Bengaluru, India - as in RoadRishi presentation!)
        # Coordinate bounds centered near Bengaluru: Lat ~12.97, Lon ~77.59
        geotiff_path = script_dir / "demo_geospatial.tif"
        crs = "EPSG:4326"
        # Affine transform: west, north, xsize, ysize
        # 1 pixel = 0.0001 degrees (~11 meters)
        transform = from_origin(77.59, 12.97, 0.0001, 0.0001)
        
        # We will write the 3-band RGB image created earlier as a GeoTIFF
        # rasterio expects shape (bands, height, width)
        geodata = arr.transpose(2, 0, 1)
        
        with rasterio.open(
            geotiff_path,
            "w",
            driver="GTiff",
            height=img_size,
            width=img_size,
            count=3,
            dtype=geodata.dtype,
            crs=crs,
            transform=transform
        ) as dst:
            dst.write(geodata)
            
        print(f"Created geospatial sample GeoTIFF at: {geotiff_path}")
        
        geo_tile_dir = output_dir / "geospatial_tiles"
        geo_count = tile_image(
            image_path=str(geotiff_path),
            tile_size=256,
            stride=128,
            out_dir=str(geo_tile_dir),
            pad=True,
            pad_mode="reflect",
            img_format="tif",
            parallel=False,
            meta_format="json"
        )
        
        geo_meta_file = geo_tile_dir / "metadata.json"
        if geo_meta_file.exists():
            with open(geo_meta_file, "r") as f:
                geo_metadata = json.load(f)
            print(f"Successfully generated {geo_count} geospatial tiles. Sample metadata entry:")
            print(json.dumps(geo_metadata[0], indent=2))
    else:
        print("\n--- Geospatial Tiling Skipped ('rasterio' not installed) ---")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
