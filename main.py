import numpy as np
import pandas as pd
import sys
from get_data_info import Data_info
from dutch_model import BERTonSVMClassifier


def get_all_data_info():
    # Open the text file in read mode
    with open('data_urls.txt', 'r') as file:
        # Read each line in the file
        for line in file:
            # Process each line as needed
            data = Data_info(line.strip())
            data.print_data()


def main(argv):
    #get_all_data_info()
    bertje = BERTonSVMClassifier()
    train_d, train_l, test_d, test_l = bertje.data_split(argv[1])

    train_e = bertje.tokenize(train_d)
    bertje.train(train_e, train_l)

    test_e = bertje.tokenize(test_d)
    predicted = bertje.predict(test_e)
    bertje.metrics(test_l, predicted)
    

if __name__ == "__main__":
    main(sys.argv)
