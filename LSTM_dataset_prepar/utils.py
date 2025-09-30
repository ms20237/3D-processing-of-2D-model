import cv2
import os
import argparse
from datetime import datetime


def time_to_seconds(time_str):
    """Convert HH:MM:SS to total seconds."""
    try:
        t = datetime.strptime(time_str, "%H:%M:%S")
        return t.hour * 3600 + t.minute * 60 + t.second
    except ValueError:
        raise ValueError(f"Invalid time format: {time_str}. Use HH:MM:SS")



