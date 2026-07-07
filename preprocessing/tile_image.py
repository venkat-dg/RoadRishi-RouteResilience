"""
Image Tiling Script
--------------------
Breaks an image (standard or geospatial raster) into smaller tiles of a given size,
sliding a window across the image with a given stride. Handles edges by either
dropping incomplete tiles or padding them (configurable).

Usage:
    python tile_image.py <image_path> --tile_size 256 --stride 128 --out_dir tiles

If --tile_size / --stride are omitted, the script will prompt for them
interactively.
"""

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from PIL import Image

# Try importing rasterio for geospatial support
try:
    import rasterio
    from rasterio.windows import Window
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# Try importing tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def get_positive_int(prompt_text, default=None):
    """Prompt the user until a valid positive integer is entered."""
    suffix = f" (default: {default})" if default is not None else ""
    while True:
        val = input(prompt_text + suffix + ": ").strip()
        if not val and default is not None:
            return default
        try:
            val_int = int(val)
            if val_int <= 0:
                print("Please enter a positive integer.")
                continue
            return val_int
        except ValueError:
            print("Invalid input, please enter an integer.")


def get_choice(prompt_text, choices, default=None):
    """Prompt the user to choose from a list of options."""
    suffix = f" (default: {default})" if default is not None else ""
    choices_str = "/".join(choices)
    while True:
        val = input(f"{prompt_text} [{choices_str}]{suffix}: ").strip().lower()
        if not val and default is not None:
            return default
        if val in choices:
            return val
        print(f"Invalid choice. Please choose from {choices}.")


def save_metadata(metadata, out_dir, format_type):
    """Save tiling metadata as JSON or CSV."""
    if format_type == "json":
        meta_path = os.path.join(out_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
    elif format_type == "csv":
        meta_path = os.path.join(out_dir, "metadata.csv")
        if not metadata:
            return
        headers = list(metadata[0].keys())
        with open(meta_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(metadata)


def process_standard_tile(args):
    """
    Worker function to process and save a single standard image tile.
    Arguments are passed in a dictionary to facilitate multiprocessing.
    """
    image_path = args["image_path"]
    out_dir = args["out_dir"]
    x = args["x"]
    y = args["y"]
    tile_w = args["tile_w"]
    tile_h = args["tile_h"]
    pad = args["pad"]
    pad_mode = args["pad_mode"]
    pad_val = args["pad_val"]
    img_format = args["format"]
    base_name = args["base_name"]
    img_w = args["img_w"]
    img_h = args["img_h"]

    fname = f"{base_name}_y{y}_x{x}.{img_format}"
    save_path = os.path.join(out_dir, fname)

    # Calculate exact cropping box
    box_w = min(tile_w, img_w - x)
    box_h = min(tile_h, img_h - y)
    is_padded = (box_w < tile_w) or (box_h < tile_h)

    with Image.open(image_path) as img:
        cropped = img.crop((x, y, x + box_w, y + box_h))
        
        if pad and is_padded:
            arr = np.array(cropped)
            pad_y = tile_h - arr.shape[0]
            pad_x = tile_w - arr.shape[1]
            
            # Form pad widths based on channel dimension
            if len(arr.shape) == 3:
                pad_width = ((0, pad_y), (0, pad_x), (0, 0))
                # For RGB/RGBA constant padding, fill channels correctly
                kw = {"constant_values": pad_val} if pad_mode == "constant" else {}
            else:
                pad_width = ((0, pad_y), (0, pad_x))
                kw = {"constant_values": pad_val} if pad_mode == "constant" else {}
                
            padded = np.pad(arr, pad_width, mode=pad_mode, **kw)
            tile_img = Image.fromarray(padded)
        else:
            tile_img = cropped

        tile_img.save(save_path)

    meta_entry = {
        "tile_name": fname,
        "original_image": Path(image_path).name,
        "y_offset": y,
        "x_offset": x,
        "height": tile_h,
        "width": tile_w,
        "is_padded": is_padded
    }
    return meta_entry


def process_geospatial_tile(args):
    """
    Worker function to process and save a single geospatial tile.
    """
    image_path = args["image_path"]
    out_dir = args["out_dir"]
    x = args["x"]
    y = args["y"]
    tile_w = args["tile_w"]
    tile_h = args["tile_h"]
    pad = args["pad"]
    pad_mode = args["pad_mode"]
    pad_val = args["pad_val"]
    img_format = args["format"]
    base_name = args["base_name"]
    img_w = args["img_w"]
    img_h = args["img_h"]

    fname = f"{base_name}_y{y}_x{x}.{img_format}"
    save_path = os.path.join(out_dir, fname)

    box_w = min(tile_w, img_w - x)
    box_h = min(tile_h, img_h - y)
    is_padded = (box_w < tile_w) or (box_h < tile_h)

    with rasterio.open(image_path) as src:
        # Read the subset using Window
        win = Window(x, y, box_w, box_h)
        data = src.read(window=win)  # shape: (bands, height, width)
        
        # Compute tile transform (affine transform of the top-left of the tile)
        # Even if padded, the pixel grid origin is at x, y
        tile_transform = rasterio.windows.transform(Window(x, y, tile_w, tile_h), src.transform)
        
        if pad and is_padded:
            bands, curr_h, curr_w = data.shape
            pad_y = tile_h - curr_h
            pad_x = tile_w - curr_w
            pad_width = ((0, 0), (0, pad_y), (0, pad_x))
            kw = {"constant_values": pad_val} if pad_mode == "constant" else {}
            data = np.pad(data, pad_width, mode=pad_mode, **kw)
        
        # Write tile as georeferenced raster
        meta = src.meta.copy()
        meta.update({
            "driver": "GTiff" if img_format in ["tif", "tiff"] else meta.get("driver", "GTiff"),
            "height": tile_h,
            "width": tile_w,
            "transform": tile_transform
        })
        
        with rasterio.open(save_path, "w", **meta) as dst:
            dst.write(data)
            
        # Compute bounding box coordinates
        bounds = rasterio.transform.array_bounds(tile_h, tile_w, tile_transform)
        # bounds is (left, bottom, right, top)

    meta_entry = {
        "tile_name": fname,
        "original_image": Path(image_path).name,
        "y_offset": y,
        "x_offset": x,
        "height": tile_h,
        "width": tile_w,
        "is_padded": is_padded,
        "crs": str(src.crs),
        "bounds_left": bounds[0],
        "bounds_bottom": bounds[1],
        "bounds_right": bounds[2],
        "bounds_top": bounds[3]
    }
    return meta_entry


def tile_image(image_path, tile_size, stride, out_dir="tiles", pad=False, 
               pad_mode="constant", pad_val=0, img_format=None, parallel=False,
               meta_format="json", quiet=False):
    """
    Tiles an image (standard or geospatial) into smaller parts.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image path '{image_path}' does not exist.")
        return 0

    # Parse tile size and stride
    if isinstance(tile_size, int):
        tile_h = tile_w = tile_size
    else:
        tile_h, tile_w = tile_size

    if isinstance(stride, int):
        stride_y = stride_x = stride
    else:
        stride_y, stride_x = stride

    # Detect geospatial raster
    is_geospatial = False
    if HAS_RASTERIO:
        try:
            with rasterio.open(image_path) as src:
                is_geospatial = True
                img_w = src.width
                img_h = src.height
                if not quiet:
                    print(f"Geospatial raster detected (CRS: {src.crs}, size: {img_w}x{img_h})")
        except rasterio.errors.RasterioIOError:
            pass

    if not is_geospatial:
        try:
            with Image.open(image_path) as img:
                img_w, img_h = img.size
                if not quiet:
                    print(f"Standard image detected (size: {img_w}x{img_h})")
        except Exception as e:
            print(f"Error opening image '{image_path}': {e}")
            return 0

    # Set default format if not provided
    if img_format is None:
        if is_geospatial:
            img_format = "tif"
        else:
            ext = Path(image_path).suffix.lower().lstrip(".")
            img_format = ext if ext in ["png", "jpg", "jpeg", "bmp"] else "png"

    os.makedirs(out_dir, exist_ok=True)
    base_name = Path(image_path).stem

    # Generate grid coordinates
    jobs = []
    y = 0
    while True:
        # Check out of bounds for y
        if y + tile_h > img_h and not pad:
            break
        if y >= img_h:
            break
            
        x = 0
        while True:
            # Check out of bounds for x
            if x + tile_w > img_w and not pad:
                break
            if x >= img_w:
                break

            jobs.append({
                "image_path": image_path,
                "out_dir": out_dir,
                "x": x,
                "y": y,
                "tile_w": tile_w,
                "tile_h": tile_h,
                "pad": pad,
                "pad_mode": pad_mode,
                "pad_val": pad_val,
                "format": img_format,
                "base_name": base_name,
                "img_w": img_w,
                "img_h": img_h
            })
            
            x += stride_x
            if x >= img_w:
                break
                
        y += stride_y
        if y >= img_h:
            break

    if not quiet:
        print(f"Generating {len(jobs)} tiles ({tile_w}x{tile_h}) with stride ({stride_x},{stride_y})...")

    metadata = []
    worker_fn = process_geospatial_tile if is_geospatial else process_standard_tile

    if parallel and len(jobs) > 1:
        if not quiet:
            print(f"Running in parallel mode...")
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(worker_fn, job) for job in jobs]
            if HAS_TQDM and not quiet:
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Tiling"):
                    metadata.append(fut.result())
            else:
                for fut in as_completed(futures):
                    metadata.append(fut.result())
                    if not quiet and len(metadata) % max(1, len(jobs)//10) == 0:
                        print(f"Progress: {len(metadata)}/{len(jobs)} tiles saved.")
    else:
        if HAS_TQDM and not quiet:
            for job in tqdm(jobs, desc="Tiling"):
                metadata.append(worker_fn(job))
        else:
            for i, job in enumerate(jobs):
                metadata.append(worker_fn(job))
                if not quiet and (i + 1) % max(1, len(jobs)//10) == 0:
                    print(f"Progress: {i + 1}/{len(jobs)} tiles saved.")

    # Sort metadata by tile name to keep it neat
    metadata.sort(key=lambda item: item["tile_name"])
    
    if meta_format != "none":
        save_metadata(metadata, out_dir, meta_format)
        if not quiet:
            print(f"Metadata saved to '{out_dir}/metadata.{meta_format}'")

    if not quiet:
        print(f"Done. {len(metadata)} tiles successfully saved to '{out_dir}'.")
        
    return len(metadata)


def main():
    parser = argparse.ArgumentParser(description="Tile an image into smaller overlapping/non-overlapping patches.")
    parser.add_argument("image_path", nargs="?", help="Path to the input image")
    parser.add_argument("--tile_size", type=str, help="Size of each tile: width,height (e.g. 256 or 256,512)")
    parser.add_argument("--stride", type=str, help="Stride between tiles: stride_x,stride_y (e.g. 128 or 128,256)")
    parser.add_argument("--out_dir", default="tiles", help="Output directory for tiles (default: 'tiles')")
    parser.add_argument("--pad", action="store_true", help="Pad image so edge content isn't dropped")
    parser.add_argument("--pad_mode", default="constant", choices=["constant", "edge", "reflect", "symmetric"],
                        help="Numpy padding mode to use (default: 'constant')")
    parser.add_argument("--pad_value", type=float, default=0.0, help="Value to use for constant padding (default: 0)")
    parser.add_argument("--format", help="Output format for tiles (e.g. png, jpg, tif)")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel processing for speedup")
    parser.add_argument("--meta_format", default="json", choices=["json", "csv", "none"],
                        help="Format of metadata file (default: 'json')")

    args = parser.parse_args()

    image_path = args.image_path
    if not image_path:
        while True:
            image_path = input("Enter path to input image: ").strip()
            if os.path.exists(image_path):
                break
            print(f"File '{image_path}' not found. Please enter a valid path.")

    # Parse or prompt tile_size
    if args.tile_size:
        parts = args.tile_size.split(",")
        if len(parts) == 1:
            tile_size = int(parts[0])
        else:
            tile_size = (int(parts[0]), int(parts[1]))
    else:
        size = get_positive_int("Enter tile size (pixels)", default=256)
        tile_size = size

    # Parse or prompt stride
    if args.stride:
        parts = args.stride.split(",")
        if len(parts) == 1:
            stride = int(parts[0])
        else:
            stride = (int(parts[0]), int(parts[1]))
    else:
        strd = get_positive_int("Enter stride (pixels)", default=tile_size)
        stride = strd

    pad = args.pad
    pad_mode = args.pad_mode
    pad_value = args.pad_value
    
    # Prompt interactively if args are omitted
    if not args.tile_size and not args.stride:
        pad_choice = get_choice("Pad edge tiles?", ["y", "n"], default="y")
        pad = (pad_choice == "y")
        if pad:
            pad_mode = get_choice("Select pad mode", ["constant", "edge", "reflect", "symmetric"], default="constant")
            if pad_mode == "constant":
                val_str = input("Enter pad value [default: 0]: ").strip()
                pad_value = float(val_str) if val_str else 0.0

    tile_image(
        image_path=image_path,
        tile_size=tile_size,
        stride=stride,
        out_dir=args.out_dir,
        pad=pad,
        pad_mode=pad_mode,
        pad_val=pad_value,
        img_format=args.format,
        parallel=args.parallel,
        meta_format=args.meta_format
    )


if __name__ == "__main__":
    main()
