import os
import argparse
import pandas as pd


def init():
    parser = argparse.ArgumentParser(description="Merge multiple CSV files into one.")
    
    parser.add_argument("--input_csvs",
                        nargs="+",
                        required=True,
                        help="List of input CSV files to merge (space-separated). Example: file1.csv file2.csv file3.csv")
    
    parser.add_argument("--output_csv",
                        type=str,
                        default="merged_keypoints_activity.csv",
                        help="Path to save the merged CSV file.")
    
    args = parser.parse_args()
    return args


def run(input_csvs: list, 
        output_csv: str):
    dataframes = []
    
    for csv_file in input_csvs:
        if not os.path.isfile(csv_file):
            print(f"⚠️ Skipping: '{csv_file}' not found.")
            continue
        
        try:
            df = pd.read_csv(csv_file)
            print(f"✅ Loaded '{csv_file}' with {len(df)} rows and {len(df.columns)} columns.")
            dataframes.append(df)
        except Exception as e:
            print(f"❌ Failed to read '{csv_file}': {e}")
    
    if not dataframes:
        print("❌ No valid CSV files were loaded. Exiting.")
        return
    
    # Concatenate DataFrames
    merged_df = pd.concat(dataframes, ignore_index=True)
    print(f"📊 Merged DataFrame shape: {merged_df.shape}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    
    try:
        merged_df.to_csv(output_csv, index=False)
        print(f"✅ Successfully saved merged CSV to '{output_csv}'")
    except Exception as e:
        print(f"❌ Failed to save merged CSV: {e}")


if __name__ == "__main__":
    args = init()
    run(args.input_csvs, 
        args.output_csv)



