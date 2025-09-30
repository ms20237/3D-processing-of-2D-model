import os
import argparse
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


def init():
    parser = argparse.ArgumentParser(description="Normalize features in a CSV file using MinMaxScaler.")
    
    parser.add_argument("--input_path",
                        type=str,
                        default="./output_path/features_added_keypoints_activity_v11s_new.csv",
                        help="Path to the input CSV file with features and labels.")
    
    parser.add_argument("--output_path",
                        type=str,
                        default="./output_path/normalized_features_added_keypoints_activity_v11s_new.csv",
                        help="Path to save the normalized CSV file.")
    
    args = parser.parse_args()
    return args


def run(input_path: str, 
        output_path: str):
    # Load CSV
    try:
        df = pd.read_csv(input_path)
        print(f"✅ Loaded '{input_path}' with shape {df.shape}")
    except FileNotFoundError:
        print(f"❌ Error: File '{input_path}' not found.")
        return
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return

    if "label" not in df.columns:
        print("❌ Error: No 'label' column found in the dataset.")
        print(f"Available columns: {list(df.columns)}")
        return

    # Separate features and labels
    features = df.drop("label", axis=1)
    labels = df["label"]

    # Normalize features
    scaler = MinMaxScaler()
    try:
        normalized_features = scaler.fit_transform(features)
    except Exception as e:
        print(f"❌ Error during normalization: {e}")
        return

    normalized_df = pd.DataFrame(normalized_features, columns=features.columns)
    normalized_df["label"] = labels

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    try:
        normalized_df.to_csv(output_path, index=False)
        print(f"✅ Saved normalized data to '{output_path}' with shape {normalized_df.shape}")
    except Exception as e:
        print(f"❌ Error saving normalized CSV: {e}")


if __name__ == "__main__":
    args = init()
    run(args.input_path, 
        args.output_path)
