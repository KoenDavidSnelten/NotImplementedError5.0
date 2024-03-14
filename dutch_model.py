import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, TFAutoModel
from sklearn.svm import SVC
from sklearn.metrics import f1_score


class BERTonSVMClassifier:
    def __init__(self, model_url="GroNLP/bert-base-dutch-cased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_url)
        self.model = AutoModel.from_pretrained(model_url)  # PyTorch
    
    def data_split(self, df_url, train_len=0.008, test_len=0.002):
        """Split the data according to the specified proportions"""
        data_frame = pd.read_csv(df_url)
        df_len = len(data_frame)

        # Shuffle the DataFrame to ensure randomness
        df_shuffled = data_frame.sample(frac=1, random_state=42)

        df_train = df_shuffled.tail(int(df_len*train_len))
        train_data = df_train['text'].tolist()
        train_labels = df_train['abusive_offensive_not'].tolist()

        df_test = df_shuffled.head(int(df_len*test_len))
        test_data = df_test['text'].tolist()
        test_labels = df_test['abusive_offensive_not'].tolist()

        return train_data, train_labels, test_data, test_labels
    
    def tokenize(self, indata):
        """Tokenize and obtain BERTje embeddings for training data"""
        input_ids = self.tokenizer(indata, padding=True,
                                         truncation=True, return_tensors="pt")
        with torch.no_grad():
            train_outputs = self.model(**input_ids)
        return train_outputs.last_hidden_state.mean(dim=1).numpy()

    def train(self, embeds, labels):
        """Train SVM classifier"""
        self.svm_classifier = SVC(kernel='linear')
        self.svm_classifier.fit(embeds, labels)

    def predict(self, embeds):
        """Predict test data"""
        return self.svm_classifier.predict(embeds)

    def metrics(self, true, predicted):
        """Calculate F1 score"""
        f1 = f1_score(true, predicted, average='macro')
        print("F1 Score:", f1)
