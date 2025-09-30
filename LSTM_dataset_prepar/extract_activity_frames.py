import cv2
import os
import argparse
from datetime import datetime

from utils import time_to_seconds


def init():
    parser = argparse.ArgumentParser(description="Extract a segment from a video using HH:MM:SS format")
    parser.add_argument("--video_path",
                        type=str, 
                        default="./video_path/LFP_Video_Auto4.avi", 
                        help="Path to the input video file")
    
    parser.add_argument("--start_time", 
                        type=str, 
                        required=True,
                        help="Start time in HH:MM:SS")
    
    parser.add_argument("--stop_time", 
                        type=str, 
                        required=True, 
                        help="Stop time in HH:MM:SS")
    
    parser.add_argument("--output_path", 
                        type=str, 
                        default="./activity_parts_video", 
                        help="Folder to save the output clip")
    
    args = parser.parse_args()
    return args
  

def run(video_path: str, 
        start_str: str, 
        stop_str: str, 
        output_path: str):
    # start/stop time
    start_time = time_to_seconds(start_str)
    stop_time = time_to_seconds(stop_str)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_time * fps)
    stop_frame = int(stop_time * fps)

    start_frame = max(0, min(start_frame, total_frames))
    stop_frame = max(0, min(stop_frame, total_frames))

    if start_frame >= stop_frame:
        raise ValueError("Start time must be less than stop time and within video duration.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    filename = os.path.splitext(os.path.basename(video_path))[0]
    
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, f"{filename}_clip_{start_str.replace(':','-')}_to_{stop_str.replace(':','-')}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    current_frame = start_frame
    while current_frame < stop_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        current_frame += 1

    cap.release()
    out.release()
    print(f"[✅] Saved video clip: {output_path}")


if __name__ == "__main__":
    args = init()
    run(args.video_path, 
        args.start_time, 
        args.stop_time, 
        args.output_path)

