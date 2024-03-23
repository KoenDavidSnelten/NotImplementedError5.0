import os
from argparse import ArgumentParser
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from TweetNormalizer import normalizeTweet
import sys


def _normalize_tweet_bertweet(tweet):
  # In our data usernames are already normalized to @USER so this will not change anything
  # however URLS are already replaced to URL, but BERTweet uses HTTPURL instead of URL
  # so we have to replace these too, the rest is done by normalizeTweet from BERTweet
  tweet['normalized_text'] = normalizeTweet(tweet['text']).replace('URL', 'HTTPURL')
  return tweet

def evaluate(infile, bert):
    # Load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base')
    model = AutoModelForSequenceClassification.from_pretrained(bert, use_safetensors=True)
    print("Models loaded")

    # Load the data to be predicted
    data = pd.read_csv(infile)
    print("Data loaded")

    # Normalize the data
    data = data.apply(_normalize_tweet_bertweet, axis=1)

    test_X = data['normalized_text'].tolist()
    print("Text normalized")

    # Tokenize the data
    tokenized = tokenizer(test_X, padding="max_length",
                          truncation=True, return_tensors="pt")
    print("Text tokenized")

    label_map_BERTweet = {0: "ABUSIVE", 1: "NOT", 2: "OFFENSIVE"}

    # predict labels
    # predicted = model.predict(tokenized)
    predications = model(**tokenized)
    print("Predictions generated")
    print(predications)

    # make output file
    # df_test["abusive_offensive_not"] = predicted
    # df_test.to_csv('test_set_out.csv', index=False)


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Evaluate the model on the given test data."
    )

    # Add bertweet
    sys.path.append('BERTweet')

    parser.add_argument("infile", help="Path to the (dev) data input file")
    parser.add_argument("model", help="Path to the model")
    parser.add_argument("output", help="Path to the output file")

    args = parser.parse_args()

    model_path = os.path.join(os.getcwd(), args.model)
    print(model_path)

    evaluate(args.infile, model_path)

