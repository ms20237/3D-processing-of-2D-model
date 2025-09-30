import cv2
import time
import os
import argparse

from pathlib import Path


def init():
    parser = argparse.ArgumentParser(description="Extract a segment from a video using HH:MM:SS format")
    parser.add_argument("--input_path",
                        type=str, 
                        default="./video_path/", 
                        help="Path to the input video file or folder containing videos")
    
    parser.add_argument("--output_path", 
                        type=str, 
                        default="./extracted_frames_activities", 
                        help="Folder to save the output frames")
    
    parser.add_argument("--stride",
                        type=int,
                        default=3,
                        help="Extract every nth frame from the video(s). For example, stride=3 saves one frame out of every three.")
    
    args = parser.parse_args()
    return args


def run(input_path: str, 
        output_path: str, 
        stride: int = 3):
    """
    Extract frames from a single video or all videos in a folder.

    Args:
        input_path (str): Path to a video file or a folder containing video files.
        output_path (str): Directory where extracted frames will be saved.
        stride (int): Extract every nth frame.
    """
    input_path = Path(input_path)

    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}

    # If input is a folder -> process all videos
    video_files = [input_path] if input_path.is_file() else [
        f for f in input_path.iterdir() if f.suffix.lower() in video_extensions
    ]

    os.makedirs(output_path, exist_ok=True)
    for video_file in video_files:
        cap = cv2.VideoCapture(str(video_file))
        video_name = video_file.stem
        count, saved_count = 0, 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if count % stride == 0:
                frame_name = f"{video_name}_{saved_count:05d}.jpg"
                cv2.imwrite(str(Path(output_path) / frame_name), frame)
                saved_count += 1

            count += 1

        cap.release()
        print(f"[INFO] Extracted {saved_count} frames from {video_name} (stride={stride})")

    
if __name__ == "__main__":
    args = init()
    run(args.input_path, 
        args.output_path, 
        args.stride)

