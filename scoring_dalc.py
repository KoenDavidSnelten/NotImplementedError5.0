import pandas as pd
import sys
from sklearn.metrics import classification_report

def compute_macro_f1(pred, gold):

    df_pred = pd.read_csv(pred, delimiter=",", header=0)
    df_gold = pd.read_csv(gold, delimiter=",", header=0)

    target_names = ['ABUSIVE', 'NOT', 'OFFENSIVE']
    results = classification_report(df_pred.iloc[:, 1], df_gold['abusive_offensive_not'], target_names=target_names)

    print(results)

def main():

    predictions = sys.argv[1]
    reference = sys.argv[2]

    compute_macro_f1(predictions, reference)




if __name__ == '__main__':
    main()
