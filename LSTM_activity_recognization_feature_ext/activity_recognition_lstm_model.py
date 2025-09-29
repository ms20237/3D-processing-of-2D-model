import cv2
import torch
import numpy as np
from collections import deque, Counter
import argparse
import os
from ultralytics import YOLO
import time

from train_extra_feature_only_model import load_feature_trained_model
from utils import calculate_angle, calculate_distance

# Keypoint mapping
KP_NAMES = {
    0: 'Nose', 1: 'Left_Ear', 2: 'Right_Ear', 3: 'Neck',
    4: 'Left_Shoulder', 5: 'Right_Shoulder', 6: 'Left_Elbow', 7: 'Right_Elbow',
    8: 'Left_Wrist', 9: 'Right_Wrist', 10: 'Spine_Shoulder', 11: 'Mid_Spine',
    12: 'Left_Hip', 13: 'Right_Hip', 14: 'Left_Knee', 15: 'Right_Knee',
    16: 'Left_Ankle', 17: 'Right_Ankle', 18: 'Spine_Lower',
    19: 'Tail1', 20: 'Tail2', 21: 'Tail3', 22: 'Tail4'
}
kp_map = {name: idx for idx, name in KP_NAMES.items()}

# Engineered features
ENGINEERED_FEATURES = [
    'left_elbow_angle', 'right_elbow_angle', 'left_knee_angle', 'right_knee_angle',
    'spine_curvature_angle', 'neck_angle', 'body_length', 'tail_length',
    'nose_to_left_wrist_dist', 'nose_to_right_wrist_dist', 'wrist_distance',
    'nose_to_left_ankle_dist', 'nose_to_right_ankle_dist', 'hind_paw_distance',
    'nose_to_left_wrist_ratio', 'nose_to_right_wrist_ratio',
    'hind_paw_distance_ratio', 'tail_angle'
]

# best: 10
def init():
    parser = argparse.ArgumentParser(description="Run action recognition on a video using YOLO and a custom LSTM model.")
    parser.add_argument('--video', type=str, default="./videos/LFP_Video_Auto2.avi")
    parser.add_argument('--yolo_model', type=str, default='./runs/pose/mouse_pose640_activity_lpf2/weights/best.pt')
    parser.add_argument('--lstm_model', type=str, default='./model/ext_feature_only_lstm_model10.pt')
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--time_threshold', type=float, default=0.3)
    parser.add_argument('--output_path', type=str, default='./output_path/activity_labeled_video2.mp4')
    
    return parser.parse_args()


def run(video_path: str,
        lstm_model_path: str,
        yolo_model_path: str,
        seq_len: int,
        device: str,
        output_path: str = None,
        time_threshold: float = 0.6,
        confidence_threshold = 0.3,
        smoothing_window = 80,
        speed_threshold = 2.5):
    """
        Runs action recognition on a video file using YOLO for pose estimation
        and an LSTM for action classification. Applies moving detection (speed-based)
        and activity stabilization (time threshold).
    """
    print(f"Using device: {device}")

    # Load YOLO model
    try:
        yolo_model = YOLO(yolo_model_path)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    # Load LSTM model
    try:
        lstm_model, metadata = load_feature_trained_model(lstm_model_path, device)
        lstm_model.eval()
    except Exception as e:
        print(f"Error loading LSTM model: {e}")
        return

    scaler = metadata['scaler']
    label_names = metadata['label_names']
    expected_input_size = lstm_model.input_size

    print(f"LSTM model loaded. Expects input size: {expected_input_size}. Actions: {label_names}")

    # Open video
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    keypoints_sequence = deque(maxlen=seq_len)
    prediction_history = deque(maxlen=smoothing_window)

    # Movement tracking
    prev_center = None
    speed_history = deque(maxlen=15)

    # Timer
    activity_start_time = None
    last_prediction = "None"
    current_action = "Waiting..."

    # Setup writer
    writer = None
    
    os.makedirs("./output_path", exist_ok=True)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*('mp4v' if output_path.endswith('.mp4') else 'XVID'))
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    print(f"Output will be saved to: {output_path}")

    # Frame loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        yolo_results = yolo_model(frame, verbose=False)

        if yolo_results and yolo_results[0].keypoints and len(yolo_results[0].keypoints.xy[0]) > 0:
            keypoints = yolo_results[0].keypoints.xy[0].cpu().numpy()
            if keypoints.shape[0] != 23:
                full_kps = np.zeros((23, 2))
                full_kps[:min(keypoints.shape[0], 23)] = keypoints[:min(keypoints.shape[0], 23)]
                keypoints_dict = {name: full_kps[idx] for name, idx in kp_map.items()}
            else:
                keypoints_dict = {name: keypoints[idx] for name, idx in kp_map.items()}

            # Movement detection
            center_point = keypoints_dict['Mid_Spine'] if 'Mid_Spine' in keypoints_dict else np.mean(
                [keypoints_dict['Left_Hip'], keypoints_dict['Right_Hip']], axis=0
            )
            if prev_center is not None:
                speed = np.linalg.norm(center_point - prev_center)
                speed_history.append(speed)
            prev_center = center_point
            avg_speed = np.mean(speed_history) if speed_history else 0

            # Feature extraction
            body_len = calculate_distance(keypoints_dict['Nose'], keypoints_dict['Spine_Lower'])
            hind_paw_dist = calculate_distance(keypoints_dict['Left_Ankle'], keypoints_dict['Right_Ankle'])
            features_dict = {
                'left_elbow_angle': calculate_angle(keypoints_dict['Left_Shoulder'], keypoints_dict['Left_Elbow'], keypoints_dict['Left_Wrist']),
                'right_elbow_angle': calculate_angle(keypoints_dict['Right_Shoulder'], keypoints_dict['Right_Elbow'], keypoints_dict['Right_Wrist']),
                'left_knee_angle': calculate_angle(keypoints_dict['Left_Hip'], keypoints_dict['Left_Knee'], keypoints_dict['Left_Ankle']),
                'right_knee_angle': calculate_angle(keypoints_dict['Right_Hip'], keypoints_dict['Right_Knee'], keypoints_dict['Right_Ankle']),
                'spine_curvature_angle': calculate_angle(keypoints_dict['Spine_Shoulder'], keypoints_dict['Mid_Spine'], keypoints_dict['Spine_Lower']),
                'neck_angle': calculate_angle(keypoints_dict['Nose'], keypoints_dict['Neck'], keypoints_dict['Spine_Shoulder']),
                'body_length': body_len,
                'tail_length': calculate_distance(keypoints_dict['Tail1'], keypoints_dict['Tail4']),
                'nose_to_left_wrist_dist': calculate_distance(keypoints_dict['Nose'], keypoints_dict['Left_Wrist']),
                'nose_to_right_wrist_dist': calculate_distance(keypoints_dict['Nose'], keypoints_dict['Right_Wrist']),
                'wrist_distance': calculate_distance(keypoints_dict['Left_Wrist'], keypoints_dict['Right_Wrist']),
                'nose_to_left_ankle_dist': calculate_distance(keypoints_dict['Nose'], keypoints_dict['Left_Ankle']),
                'nose_to_right_ankle_dist': calculate_distance(keypoints_dict['Nose'], keypoints_dict['Right_Ankle']),
                'hind_paw_distance': hind_paw_dist,
                'nose_to_left_wrist_ratio': calculate_distance(keypoints_dict['Nose'], keypoints_dict['Left_Wrist']) / body_len if body_len else 0,
                'nose_to_right_wrist_ratio': calculate_distance(keypoints_dict['Nose'], keypoints_dict['Right_Wrist']) / body_len if body_len else 0,
                'hind_paw_distance_ratio': hind_paw_dist / body_len if body_len else 0,
                'tail_angle': calculate_angle(keypoints_dict['Spine_Lower'], keypoints_dict['Tail1'], keypoints_dict['Tail4'])
            }
            engineered_features = np.array([features_dict[f] for f in ENGINEERED_FEATURES])
            if len(engineered_features) != expected_input_size:
                continue

            scaled_features = scaler.transform(engineered_features.reshape(1, -1))
            keypoints_sequence.append(scaled_features.flatten())

            if len(keypoints_sequence) == seq_len:
                seq_arr = np.array(keypoints_sequence)
                input_tensor = torch.FloatTensor(seq_arr).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs, _ = lstm_model(input_tensor)
                    final_output = outputs[:, -1, :]
                    probs = torch.softmax(final_output, dim=1)
                    conf, pred_idx = torch.max(probs, 1)
                    if conf.item() >= confidence_threshold:
                        prediction_history.append(label_names[pred_idx.item()])
                    else:
                        prediction_history.append("None")

            # Temporal smoothing
            if prediction_history:
                most_common_pred = Counter(prediction_history).most_common(1)[0][0]

                # Priority: moving
                if avg_speed > speed_threshold:
                    current_action = "moving"
                    activity_start_time = None
                else:
                    if most_common_pred != "None":
                        if last_prediction != most_common_pred:
                            activity_start_time = time.time()
                        if activity_start_time and (time.time() - activity_start_time) > time_threshold:
                            current_action = most_common_pred
                        else:
                            current_action = "None"
                    else:
                        current_action = "None"
                        activity_start_time = None
                    last_prediction = most_common_pred

            annotated_frame = yolo_results[0].plot()
        else:
            annotated_frame = frame
            current_action = "No rat detected"
            activity_start_time = None
            last_prediction = "None"

        cv2.putText(annotated_frame, f'Action: {current_action}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        if writer:
            writer.write(annotated_frame)
        cv2.imshow('YOLO + LSTM Action Recognition', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Inference complete.")


if __name__ == '__main__':
    args = init()
    selected_device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
    run(args.video, args.lstm_model, args.yolo_model, args.seq_len,
        selected_device, args.output_path, args.time_threshold)
