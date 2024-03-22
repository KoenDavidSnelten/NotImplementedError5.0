import os
from argparse import ArgumentParser
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import sys

def evaluate(infile, bert):
    # load models
    tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base')
    model = AutoModelForSequenceClassification.from_pretrained(bert, use_safetensors=True)
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

    parser = ArgumentParser(
        description="Evaluate the model on the given test data."
    )

    # Add bertweet
    import sys
    sys.path.append('BERTweet')

    parser.add_argument("infile", help="Path to the (dev) data input file")
    parser.add_argument("model", help="Path to the model")
    parser.add_argument("output", help="Path to the output file")

    args = parser.parse_args()

    model_path = os.path.join(os.getcwd(), args.model)
    print(model_path)

    evaluate(args.infile, model_path)

