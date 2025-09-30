import os
import cv2
import argparse


def init():
    parser = argparse.ArgumentParser(description="Resize a video to 640x480 resolution.")
    
    parser.add_argument("--input_path",
                        type=str,
                        required=True,
                        help="Path to the input video file.")
    
    parser.add_argument("--output_path",
                        type=str,
                        default=None,
                        help="Path to save the resized video. If not provided, adds '_resized' to input filename.")
    
    parser.add_argument("--width",
                        type=int,
                        default=640,
                        help="Width of the output video.")
    
    parser.add_argument("--height",
                        type=int,
                        default=480,
                        help="Height of the output video.")
    
    args = parser.parse_args()
    return args


def run(input_path: str, 
        output_path: str = None,
        width: int = 640, 
        height: int = 480):
    
    if not os.path.isfile(input_path):
        print(f"❌ Error: Input video '{input_path}' not found.")
        return

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_resized{ext}"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video '{input_path}'.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📹 Resizing video: {input_path} ({frame_count} frames) -> {width}x{height}")

    import tqdm
    for _ in tqdm.tqdm(range(frame_count), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (width, height))
        out.write(frame_resized)

    cap.release()
    out.release()
    print(f"✅ Video resized successfully and saved to '{output_path}'")


if __name__ == "__main__":
    args = init()
    run(args.input_path, 
        args.output_path, 
        args.width, 
        args.height)
