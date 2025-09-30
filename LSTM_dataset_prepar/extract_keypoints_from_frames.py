import os
import cv2
import argparse
import pandas as pd

from tqdm import tqdm
from ultralytics import YOLO


def init():
    parser = argparse.ArgumentParser(description="Extract YOLO keypoints from activity-labeled images and save to CSV.")
    
    parser.add_argument("--dataset_dir",
                        type=str,
                        default="./extracted_frames_activities_myself_inshot",
                        help="Directory containing subfolders of activity images.")
    
    parser.add_argument("--model_path",
                        type=str,
                        default="./model_path/best.pt",
                        help="Path to trained YOLO-Pose model.")
    
    parser.add_argument("--output_csv",
                        type=str,
                        default="./output_csv/keypoints_activity_v11s_new.csv",
                        help="Path to save output CSV with extracted keypoints and labels.")
    
    parser.add_argument("--conf",
                        type=float,
                        default=0.3,
                        help="Confidence threshold for YOLO predictions.")
    
    args = parser.parse_args()
    return args


def run(dataset_dir: str, 
        model_path: str, 
        output_csv: str, 
        conf: float = 0.3):
    # Validate dataset path
    if not os.path.isdir(dataset_dir):
        print(f"❌ Error: Dataset directory '{dataset_dir}' not found.")
        return

    # Load model
    try:
        model = YOLO(model_path)
        print(f"✅ Successfully loaded model from '{model_path}'")
    except Exception as e:
        print(f"❌ Failed to load model from '{model_path}': {e}")
        return

    rows = []
    activities = [a for a in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, a))]

    if not activities:
        print(f"❌ No activity subfolders found in '{dataset_dir}'.")
        return

    # Iterate over activity folders
    for activity in tqdm(activities, desc="Processing Activities"):
        activity_path = os.path.join(dataset_dir, activity)
        image_files = [f for f in os.listdir(activity_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

        if not image_files:
            tqdm.write(f"⚠️ Skipping '{activity}' (no images found)")
            continue

        for img_file in tqdm(image_files, desc=f"  {activity}", leave=False):
            img_path = os.path.join(activity_path, img_file)

            img = cv2.imread(img_path)
            if img is None:
                tqdm.write(f"⚠️ Could not read image: {img_path}")
                continue

            try:
                results = model.predict(source=img, conf=conf, save=False, verbose=False)
            except Exception as e:
                tqdm.write(f"❌ Prediction failed for {img_path}: {e}")
                continue

            if results and results[0].keypoints is not None and len(results[0].keypoints) > 0:
                kpts = results[0].keypoints[0].xy.cpu().numpy().squeeze()
                if kpts.shape[0] == 23:  # Ensure expected number of keypoints
                    flattened_kpts = kpts.flatten()
                    row = list(flattened_kpts) + [activity]
                    rows.append(row)
                else:
                    tqdm.write(f"⚠️ Skipped {img_path} (unexpected keypoints shape: {kpts.shape})")

    if not rows:
        print("❌ No valid keypoints extracted. CSV will not be saved.")
        return

    # Save CSV
    columns = [f'kp{i}_{axis}' for i in range(23) for axis in ['x', 'y']] + ['label']
    df = pd.DataFrame(rows, columns=columns)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    try:
        df.to_csv(output_csv, index=False)
        print(f"✅ Saved keypoint dataset to {output_csv}")
    except Exception as e:
        print(f"❌ Failed to save CSV: {e}")


if __name__ == "__main__":
    args = init()
    run(args.dataset_dir, 
        args.model_path, 
        args.output_csv, 
        args.conf)


