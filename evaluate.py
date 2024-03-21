import pandas as pd
from transformers import AutoTokenizer, AutoModel
import sys

def evaluate(infile, bert, svm):
    # load models
    tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base')
    model = AutoModel.from_pretrained(bert)  # PyTorch
    # load dev_data_text
    df_test = pd.read_csv(infile)
    # tokenize
    tokenized = tokenizer(df_test["text"], padding="max_length",
                          truncation=True)
    label_map_BERTweet = {0: "ABUSIVE", 1: "NOT", 2: "OFFENSIVE"}

    # predict labels
    predicted = model.predict(tokenized)

    # make output file
    df_test["abusive_offensive_not"] = predicted
    df_test.to_csv('test_set_out.csv', index=False)


if __name__ == "__main__":
    evaluate(sys.argv[1], sys.argv[2], sys.argv[3])
