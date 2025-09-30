import pandas as pd
import numpy as np
import argparse


def init():
    parser = argparse.ArgumentParser(description="add features to csv file.")
    parser.add_argument("--input_path", 
                        type=str,
                        default='./output_csv/keypoints_activity_v11s_new.csv',
                        help="input csv dataset path")

    parser.add_argument("--output_path", 
                        type=str,
                        default='./output_csv/features_added_keypoints_activity.csv',
                        help="output csv dataset path")
    
    args = parser.parse_args()
    return args

    
def run(input_path: str, 
        output_path: str):
    # Load dataset
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: '{input_path}' not found. Please check the path.")
        exit()

    # --- Keypoint Mapping Assumption ---
    # If your keypoint layout is different, you MUST adjust this dictionary.
    keypoints_map = {
        'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
        'neck': 5, 'left_shoulder': 6, 'right_shoulder': 7, 'left_elbow': 8,
        'right_elbow': 9, 'left_wrist': 10, 'right_wrist': 11, 'mid_hip': 12,
        'left_hip': 13, 'right_hip': 14, 'left_knee': 15, 'right_knee': 16,
        'left_ankle': 17, 'right_ankle': 18, 'tail_base': 19, 'tail_mid1': 20,
        'tail_mid2': 21, 'tail_tip': 22
    }

    # Helper Functions 
    def calculate_angle(p1, p2, p3):
        """Calculates the angle between three 2D points in degrees."""
        v1 = p1 - p2
        v2 = p3 - p2
        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_product == 0:
            return np.nan
        cosine_angle = dot_product / norm_product
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def calculate_distance(p1, p2):
        """Calculates the Euclidean distance between two 2D points."""
        return np.linalg.norm(p1 - p2)

    def get_kp(row, kp_name):
        """Extracts keypoint coordinates from a dataframe row."""
        kp_idx = keypoints_map[kp_name]
        return np.array([row[f'kp{kp_idx}_x'], row[f'kp{kp_idx}_y']])

    # --- Feature Calculation ---
    features = pd.DataFrame(index=df.index)

    # Angles
    features['left_elbow_angle'] = df.apply(lambda row: calculate_angle(get_kp(row, 'left_shoulder'), get_kp(row, 'left_elbow'), get_kp(row, 'left_wrist')), axis=1)
    features['right_elbow_angle'] = df.apply(lambda row: calculate_angle(get_kp(row, 'right_shoulder'), get_kp(row, 'right_elbow'), get_kp(row, 'right_wrist')), axis=1)
    features['left_knee_angle'] = df.apply(lambda row: calculate_angle(get_kp(row, 'left_hip'), get_kp(row, 'left_knee'), get_kp(row, 'left_ankle')), axis=1)
    features['right_knee_angle'] = df.apply(lambda row: calculate_angle(get_kp(row, 'right_hip'), get_kp(row, 'right_knee'), get_kp(row, 'right_ankle')), axis=1)
    features['spine_curvature_angle'] = df.apply(lambda row: calculate_angle(get_kp(row, 'neck'), get_kp(row, 'mid_hip'), get_kp(row, 'tail_base')), axis=1)
    features['neck_angle'] = df.apply(lambda row: calculate_angle(get_kp(row, 'nose'), get_kp(row, 'neck'), get_kp(row, 'mid_hip')), axis=1)
    features['tail_angle'] = df.apply(lambda row: calculate_angle(get_kp(row, 'mid_hip'), get_kp(row, 'tail_base'), get_kp(row, 'tail_tip')), axis=1)

    # Distances
    features['body_length'] = df.apply(lambda row: calculate_distance(get_kp(row, 'neck'), get_kp(row, 'tail_base')), axis=1)
    features['tail_length'] = df.apply(lambda row: calculate_distance(get_kp(row, 'tail_base'), get_kp(row, 'tail_tip')), axis=1)
    features['nose_to_left_wrist_dist'] = df.apply(lambda row: calculate_distance(get_kp(row, 'nose'), get_kp(row, 'left_wrist')), axis=1)
    features['nose_to_right_wrist_dist'] = df.apply(lambda row: calculate_distance(get_kp(row, 'nose'), get_kp(row, 'right_wrist')), axis=1)
    features['wrist_distance'] = df.apply(lambda row: calculate_distance(get_kp(row, 'left_wrist'), get_kp(row, 'right_wrist')), axis=1)
    features['nose_to_left_ankle_dist'] = df.apply(lambda row: calculate_distance(get_kp(row, 'nose'), get_kp(row, 'left_ankle')), axis=1)
    features['nose_to_right_ankle_dist'] = df.apply(lambda row: calculate_distance(get_kp(row, 'nose'), get_kp(row, 'right_ankle')), axis=1)
    features['hind_paw_distance'] = df.apply(lambda row: calculate_distance(get_kp(row, 'left_ankle'), get_kp(row, 'right_ankle')), axis=1)

    # Ratios
    features['nose_to_left_wrist_ratio'] = (features['nose_to_left_wrist_dist'] / features['body_length']).replace([np.inf, -np.inf], np.nan)
    features['nose_to_right_wrist_ratio'] = (features['nose_to_right_wrist_dist'] / features['body_length']).replace([np.inf, -np.inf], np.nan)
    features['hind_paw_distance_ratio'] = (features['hind_paw_distance'] / features['body_length']).replace([np.inf, -np.inf], np.nan)

    # --- Rearrange and Save ---
    kp_cols = [col for col in df.columns if col.startswith('kp')]
    label_col = ['label']
    new_df = pd.concat([df[kp_cols], features, df[label_col]], axis=1)

    new_df.to_csv(output_path, index=False)

    print(f"New CSV file '{output_path}' has been created with the 18 new features.")
    print("Process complete. Here are the first 5 rows of your new data:")
    print(new_df.head())

    
if __name__ == '__main__':
    args = init()
    run(args.input_path,
        args.output_path)    
    
    