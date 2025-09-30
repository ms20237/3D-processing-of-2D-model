import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def init():
    parser = argparse.ArgumentParser(description="Plot labels distribution of a CSV file containing keypoints.")
    parser.add_argument("--input_path", 
                        type=str,
                        default='./output_csv/keypoints_activity_v11s_new.csv',
                        help="Path to the input CSV dataset")
    
    args = parser.parse_args()
    return args


def run(input_path: str):
    # Load dataset
    df = pd.read_csv(input_path)

    # Check data balance
    print("--- Class Distribution ---")
    print(df['label'].value_counts())
    print("\n" + "="*50 + "\n")

    # Check statistical properties of the features
    print("--- Feature Statistics ---")
    print(df.drop('label', axis=1).describe())
    print("\n" + "="*50 + "\n")

    # Reshape into long format
    df_long = df.melt(id_vars=['label'], var_name='keypoint', value_name='value')

    # Separate coordinate type and keypoint id
    df_long['coord_type'] = df_long['keypoint'].apply(lambda x: 'x' if '_x' in x else 'y')
    df_long['keypoint_id'] = df_long['keypoint'].apply(lambda x: x.split('_')[0])

    # Select subset of keypoints to avoid clutter
    selected_keypoints = [
        'kp0','kp1','kp2','kp3','kp4','kp5','kp6','kp7','kp8','kp9',
        'kp10','kp11','kp12','kp13','kp14','kp15','kp16','kp17','kp18','kp19',
        'kp20','kp21','kp22'
    ]
    df_filtered = df_long[df_long['keypoint_id'].isin(selected_keypoints)]

    # Create boxplot
    plt.figure(figsize=(18, 10))
    sns.boxplot(data=df_filtered, x='keypoint', y='value', hue='label')
    plt.title('Distribution of Selected Keypoint Coordinates by Activity')
    plt.xlabel('Keypoint')
    plt.ylabel('Coordinate Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('keypoint_boxplot2.png')
    plt.show()

    print("Saved boxplot as keypoint_boxplot2.png")
    

if __name__ == '__main__':
    args = init()
    run(args.input_path)
