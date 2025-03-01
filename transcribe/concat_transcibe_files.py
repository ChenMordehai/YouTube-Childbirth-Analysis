import os
import pandas as pd
import argparse


def concatenate_csv_files(input_dir, output_file):
    """
    Concatenates all CSV files in the given directory into one CSV file.

    Parameters:
        input_dir (str): Path to the directory containing CSV files.
        output_file (str): Path to save the concatenated CSV file.
    """

    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    if not csv_files:
        print("No CSV files found in the directory.")
        return

    dataframes = []

    for file in csv_files:
        file_path = os.path.join(input_dir, file)
        df = pd.read_csv(file_path)
        dataframes.append(df)

    # Concatenate all DataFrames
    merged_df = pd.concat(dataframes, ignore_index=True)
    print(len(merged_df))

    # Save to output file
    merged_df.to_csv(output_file, index=False)
    print(f"Merged CSV saved to {output_file}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Concatenate CSV files in a directory.")
    # parser.add_argument("input_dir", type=str, help="Path to the directory containing CSV files",
    #                     default="/sise/home/mordeche/bigdata_youtube/transcribe_data")
    # parser.add_argument("output_file", type=str, help="Path to save the concatenated CSV file",
    #                     default="/sise/home/mordeche/bigdata_youtube/data/concat_transcribe_dfs.csv")

    # args = parser.parse_args()

    # concatenate_csv_files(args.input_dir, args.output_file)
    concatenate_csv_files("/sise/home/mordeche/bigdata_youtube/transcribe_data",
                          "/sise/home/mordeche/bigdata_youtube/data/concat_transcribe_dfs.csv")
