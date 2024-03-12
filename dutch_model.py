import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, TFAutoModel
from sklearn.svm import SVC
from sklearn.metrics import f1_score

# Load BERTje tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("GroNLP/bert-base-dutch-cased")
model = AutoModel.from_pretrained("GroNLP/bert-base-dutch-cased")  # PyTorch

# Load data
data_frame = pd.read_csv("TRAIN/train_data_offensive_abusive_taskC.csv")

# split data 80 / 20
length_data_frame = len(data_frame)

# Shuffle the DataFrame to ensure randomness
data_frame_shuffled = data_frame.sample(frac=1, random_state=42)


train_data_frame = data_frame_shuffled.tail(int(length_data_frame*0.08))
train_data = train_data_frame['text'].tolist()
train_labels = train_data_frame['abusive_offensive_not'].tolist()

test_data_frame = data_frame_shuffled.head(int(length_data_frame*0.02))
test_data = test_data_frame['text'].tolist()
test_labels = test_data_frame['abusive_offensive_not'].tolist()


# Tokenize and obtain BERTje embeddings for training data
train_input_ids = tokenizer(train_data, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    train_outputs = model(**train_input_ids)
train_embeddings = train_outputs.last_hidden_state.mean(dim=1).numpy()


# Train SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(train_embeddings, train_labels)

# Tokenize and obtain BERTje embeddings for test data
test_input_ids = tokenizer(test_data, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    test_outputs = model(**test_input_ids)
test_embeddings = test_outputs.last_hidden_state.mean(dim=1).numpy()

# Predict on test data
predictions = svm_classifier.predict(test_embeddings)

# Calculate F1 score
f1 = f1_score(test_labels, predictions, average='macro')
print("F1 Score:", f1)