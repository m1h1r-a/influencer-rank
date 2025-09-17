import glob
import os

import pandas as pd

CSV_FOLDER_PATH = "object_csvs"
OUTPUT_FILENAME = "image_objects.csv"


def combine_object_csvs(folder_path, output_file):
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    if not csv_files:
        print(f"Error: No CSV files found in '{folder_path}'.")
        return

    print(f"Found {len(csv_files)} files to combine: {csv_files}")

    df_list = [pd.read_csv(file) for file in csv_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.drop_duplicates(subset="post_id", inplace=True)
    combined_df.to_csv(output_file, index=False)

    print(f"\nSuccessfully combined files into '{output_file}'")
    print(f"Total unique posts with object data: {len(combined_df)}")


if __name__ == "__main__":
    if not os.path.isdir(CSV_FOLDER_PATH):
        print(
            f"Error: Please create a folder named '{CSV_FOLDER_PATH}' and place your 4 CSV files inside it."
        )
    else:
        combine_object_csvs(CSV_FOLDER_PATH, OUTPUT_FILENAME)
