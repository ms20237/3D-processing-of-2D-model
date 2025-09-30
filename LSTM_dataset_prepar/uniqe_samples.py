import os
import shutil
import argparse

from tqdm import tqdm


def init():
    parser = argparse.ArgumentParser(description="Select unique images from groups and copy to a new folder.")
    
    parser.add_argument("--input_path",
                        type=str,
                        required=True,
                        help="Path to the folder containing original images.")
    
    parser.add_argument("--output_path",
                        type=str,
                        required=True,
                        help="Path to save the selected images.")
    
    parser.add_argument("--group_size",
                        type=int,
                        default=22,
                        help="Number of images per group.")
    
    parser.add_argument("--keep_count",
                        type=int,
                        default=2,
                        help="Number of images to keep from each group.")
    
    args = parser.parse_args()
    return args


def run(input_path: str, 
        output_path: str, 
        group_size: int = 22,
        keep_count: int = 2):
    
    if not os.path.isdir(input_path):
        print(f"❌ Error: Input directory '{input_path}' not found.")
        return

    os.makedirs(output_path, exist_ok=True)

    # List and sort image files
    files = sorted([f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not files:
        print(f"❌ No image files found in '{input_path}'.")
        return

    total_copied = 0
    for i in range(0, len(files), group_size):
        group = files[i:i + group_size]
        selected = group[:keep_count]  # pick first N images; selection logic can be modified

        for fname in selected:
            src = os.path.join(input_path, fname)
            dst = os.path.join(output_path, fname)
            try:
                shutil.copy(src, dst)
                total_copied += 1
            except Exception as e:
                print(f"⚠️ Failed to copy '{src}' -> '{dst}': {e}")

    print(f"✅ Saved {total_copied} images to '{output_path}'")


if __name__ == "__main__":
    args = init()
    run(args.input_path, 
        args.output_path, 
        args.group_size, 
        args.keep_count)
