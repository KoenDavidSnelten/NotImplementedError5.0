# Comparing BERTweet and BERTje in abusive language detection

We trained a total of 4 models on the 'DALC' dataset to see if models that are language specific (BERTje) or models that are task/data specific (BERTweet) perform better at abusive/offensive language detection.

## The models

For both BERTje and BERTweet we trained a SVM on the word embeddings from the specific models. Next to this we fine tuned both models. 

### BERTje

For more information on the BERTje model itself view their [github page](https://github.com/wietsedv/bertje)

The training of the SVM and fine-tuned model is done using notebooks. Which are stored in the `bertje` folder.

- [`bertje/bertje_finetuned.ipynb`](bertje/bertje_finetuned.ipynb) trains the fine-tuned BERTje model
- [`bertje/bertje_svm.ipynb`](bertje/bertje_svm.ipynb) trains the fine-tuned BERTje model

Instructions on how the use the notebooks is in the notebook itself. However it is important to have the DALC data in the same directory with the filename: `train_data_offensive_abusive_taskC.csv`

### BERTweet 

For more information on the BERTweet model itself view their [huggingface documentation](https://huggingface.co/docs/transformers/en/model_doc/bertweet)

The training of the SVM and fine-tuned model is done using notebooks. Which are stored in the `bertweet` folder.

- [`bertweet/bertweet_finetuned.ipynb`](bertweet/bertweet_finetuned.ipynb) trains the fine-tuned bertweet model
- [`bertweet/bertweet_svm.ipynb`](bertweet/bertweet_svm.ipynb) trains the fine-tuned bertweet model

Instructions on how the use the notebooks is in the notebook itself. However it is important to have the DALC data in the same directory with the filename: `train_data_offensive_abusive_taskC.csv`
