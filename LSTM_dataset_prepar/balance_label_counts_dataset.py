import pandas as pd
import argparse
from tqdm import tqdm


def init():
    parser = argparse.ArgumentParser(description="balance labels count in csv file.")
    parser.add_argument("--input_path", 
                        type=str,
                        default='./output_csv/normalized_features_added_keypoints_activity_v11s_new.csv',
                        help="input csv dataset path")

    parser.add_argument("--output_path", 
                        type=str,
                        default='./output_csv/balanced_normalized_features_added_keypoints_activity_v11s_new.csv',
                        help="output csv dataset path")

    parser.add_argument("--activity_column", 
                        type=str,
                        default='label',
                        help="column name of activities.")
        
    parser.add_argument("--random_state_seed", 
                        type=int,
                        default=42,
                        help="random state seed number.")    

    args = parser.parse_args()
    return args


def run(input_path: str, 
        output_path: str, 
        activity_col: str, 
        random_seed: int = 42):
    # Load dataset
    try:
        df = pd.read_csv(input_path)
        print(f"Successfully loaded '{input_path}'.")
        print("-" * 30)
    except FileNotFoundError:
        print(f"Error: The file '{input_path}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    if activity_col not in df.columns:
        print(f"Error: The column '{activity_col}' was not found in the file.")
        print(f"Available columns are: {list(df.columns)}")
        return

    print("Initial distribution of activities:")
    initial_counts = df[activity_col].value_counts()
    print(initial_counts)
    print("-" * 30)

    min_count = initial_counts.min()
    print(f"The minimum sample count is {min_count}. This will be the target for each activity.")
    print("-" * 30)

    print("Balancing the dataset...")

    # Use tqdm for progress bar over the groups
    balanced_samples = []
    for label, group_df in tqdm(df.groupby(activity_col), desc="Sampling each activity"):
        sampled = group_df.sample(n=min_count, random_state=random_seed)
        balanced_samples.append(sampled)

    balanced_df = pd.concat(balanced_samples).reset_index(drop=True)

    print("Balancing complete.")
    print("-" * 30)

    print("New balanced distribution of activities:")
    print(balanced_df[activity_col].value_counts())
    print("-" * 30)

    try:
        balanced_df.to_csv(output_path, index=False)
        print(f"Successfully created and saved the balanced dataset to '{output_path}'")
    except Exception as e:
        print(f"An error occurred while saving the CSV file: {e}")


if __name__ == '__main__':
    args = init()
    run(args.input_path,
        args.output_path,
        args.activity_column,
        args.random_state_seed
    )

