import numpy as np
import pandas as pd
from get_data_info import Data_info



def get_all_data_info():
    # Open the text file in read mode
    with open('data_urls.txt', 'r') as file:
        # Read each line in the file
        for line in file:
            # Process each line as needed
            data = Data_info(line.strip())
            data.print_data()

def main():
    get_all_data_info()
    

if __name__ == "__main__":
    main()