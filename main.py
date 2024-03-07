import numpy as np
import pandas as pd


def get_data_info_train():
    urls = ["TRAIN/train_data_abusive_taskB.csv", "TRAIN/train_data_offensive_abusive_taskC.csv", "TRAIN/train_data_offensive_taskA.csv"]
    for url in urls:
        # Read the CSV file
        df = pd.read_csv(url)

        # Get the basic structure of the data
        print("Basic info about the data:")
        print("--------------------------")
        print("Data Location:", url)
        print("Number of rows:", len(df))
        print("Number of columns:", len(df.columns))
        print("Column names:", df.columns.tolist())
        print("Data types:")
        print(df.dtypes)
        
        # Count occurrences of specified columns
        specified_columns = ['abusive', 'abusive_offensive_not', 'offensive_aggregated', 'offense_a1', 'offense_a2', 'offense_a3', 'offense_a4']
        column_counts = {}
        for col in specified_columns:
            if col in df.columns:
                column_counts[col] = df[col].value_counts()

        # Print the counts
        print("\nCounts of specified columns:")
        for col, counts in column_counts.items():
            print(f"{col}:")
            print(counts)
            print()

def get_data_info_dev():
    urls = ["DEV\data_submission_format_scorer.csv", "DEV\dev_data_abusive_gold.csv", 
            "DEV\dev_data_abusive_offensive_gold.csv", "DEV\dev_data_offensive_gold.csv",
            "DEV\dev_data_text.csv"]
    
    for url in urls:
        # Read the CSV file
        df = pd.read_csv(url)

        # Get the basic structure of the data
        print("Basic info about the data:")
        print("--------------------------")
        print("Number of rows:", len(df))
        print("Number of columns:", len(df.columns))
        print("Column names:", df.columns.tolist())
        print("Data types:")
        print(df.dtypes)

        # Count occurrences of specified columns if present
        specified_columns = ['abusive', 'abusive_offensive_not', 'offensive_aggregated', 'offense_a1', 'offense_a2', 'offense_a3', 'offense_a4']
        column_counts = {}
        for col in specified_columns:
            if col in df.columns:
                column_counts[col] = df[col].value_counts()

        # Print the counts
        print("\nCounts of specified columns:")
        for col, counts in column_counts.items():
            print(f"{col}:")
            print(counts)
            print()

def main():
    
    # get_data_info_train()

    get_data_info_dev()



if __name__ == "__main__":
    main()