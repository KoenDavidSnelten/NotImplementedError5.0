# Comparing BERTweet and BERTje in abusive language detection

We trained a total of 4 models on the 'DALC' dataset to see if models that are language specific (BERTje) or models that are task/data specific (BERTweet) perform better at abusive/offensive language detection.

## The models

For both BERTje and BERTweet we trained a SVM on the word embeddings from the specific models. Next to this we fine tuned both models. 

### BERTje

For more information on the BERTje model itself view their [github page](https://github.com/wietsedv/bertje)

### BERTweet 

For more information on the BERTweet model itself view their [huggingface documentation](https://huggingface.co/docs/transformers/en/model_doc/bertweet)

