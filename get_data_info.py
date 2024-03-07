import numpy as np
import pandas as pd

class Data_info():

    def __init__(self, url):
        self.url = url
        self.df = pd.read_csv(url)
        self.rows = len(self.df)
        self.cols = len(self.df.columns)
        self.cols_labels = self.df.columns.tolist()
        self.column_counts = self.get_data()

    def get_data(self):
        # Count occurrences of specified columns
        specified_columns = ['abusive', 'abusive_offensive_not', 'offensive_aggregated', 'offense_a1', 'offense_a2', 'offense_a3', 'offense_a4']
        column_counts = {}
        for col in specified_columns:
            if col in self.df.columns:
                column_counts[col] = self.df[col].value_counts()
        return column_counts
    
    def print_data(self):
        # Get the basic structure of the data
        print("Basic info about the data:")
        print("--------------------------")
        print("Data Location:", self.url)
        print("Number of rows:", self.rows)
        print("Number of columns:", self.cols)
        print("Column names:", self.cols_labels)
        # Print the counts
        print("\nCounts of specified columns:")
        for col, counts in self.column_counts.items():
            print(f"{col}:")
            print(counts)
            print()

   