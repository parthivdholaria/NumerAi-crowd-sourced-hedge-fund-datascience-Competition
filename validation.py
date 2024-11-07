import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
from numerapi import NumerAPI
import json
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Define paths
model_path = "saved_models/model.pth"  # Replace with your model file name
features_file_path = "saved_models/features.txt"  # File containing the selected features
validation_data_path = "data/validation.parquet"
features_json_path = "data/features.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ClassificationLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, lr):
        super(ClassificationLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.class_to_bucket = {0: 0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1}

        self.history = {'loss': [], 'f1_score': []}

    def forward(self, x):
        h, _ = self.lstm(x)
        x = self.linear(h[:, -1, :])  # Take last output for classification
        return x

    def train_model(self, train_loader, num_epochs, device):
        self.to(device)
        for epoch in range(num_epochs):
            epoch_loss = 0
            all_preds, all_labels = [], []
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)
                self.optimizer.zero_grad()
                outputs = self(inputs)
                label_indices = torch.tensor([self.bucket_to_class(val) for val in labels.cpu().numpy()], dtype=torch.long).to(device)
                loss = self.criterion(outputs, label_indices)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(label_indices.cpu().numpy())

            avg_loss = epoch_loss / len(train_loader)
            f1 = f1_score(all_labels, all_preds, average='macro')
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, F1 Score: {f1:.4f}")
            self.history['loss'].append(avg_loss)
            self.history['f1_score'].append(f1)

    def bucket_to_class(self, val):
        bucket_to_class = {v: k for k, v in self.class_to_bucket.items()}
        return bucket_to_class[val]
model = torch.load(model_path, map_location=device)
model.eval()  # Set the model to evaluation mode

with open(features_file_path, "r") as f:
    selected_features = [line.strip() for line in f.readlines()]

print(f"Selected features for validation: {selected_features}")

# Load features metadata for reference
with open(features_json_path, "r") as f:
    feature_metadata = json.load(f)
target_cols = feature_metadata["targets"]

# Load and filter the validation data to include only selected features
validation_data = pd.read_parquet(
    validation_data_path,
    columns=["era", "data_type"] + selected_features + target_cols
)

# Filter for validation rows only
validation_data = validation_data[validation_data["data_type"] == "validation"]
validation_data = validation_data.drop(columns=["data_type"])
validation = validation_data[['era','target'] + selected_features]
validation = validation[validation_data["era"].isin(validation["era"].unique()[::5])]

# Remove embargo eras
last_train_era = int(validation["era"].unique()[-1])
eras_to_embargo = [str(era).zfill(4) for era in [last_train_era + i for i in range(4)]]
validation = validation[~validation["era"].isin(eras_to_embargo)]
validation.sort_values(by=["era"], inplace=True)
final_feature = selected_features
X_val = validation[final_feature]
y_val = validation["target"]

# Define class mapping
bucket_to_class = {0: 0, 0.25: 1, 0.5: 2, 0.75: 3, 1: 4}
y_val_class_indices = y_val.map(bucket_to_class)
window_len = 5

Xtest = validation.iloc[:,2:]
ytest = validation.iloc[:,1]

def preprocess_validation_data(validation_df, window_len, features):
    Xraw = validation_df[features]
    Xraw_filled = Xraw.fillna(-1)

    new_data = []

    # Pad the start of the data with the same number of missing values as window_len - 1
    padding = torch.full((window_len - 1, len(features)), -1)  # -1 for padding
    Xraw_padded = torch.cat((padding, torch.tensor(Xraw_filled.values, dtype=torch.float32)))

    for start in range(0, len(Xraw_filled)):
        new_row_data = Xraw_padded[start : start + window_len].reshape(window_len, len(features))
        new_data.append(new_row_data)

    return torch.stack(new_data)


# X_val = preprocess_validation_data(validation_df,window_len,features)
X_val = preprocess_validation_data(validation,window_len,final_feature)
with torch.no_grad():
    outputs = model(X_val)
    predictions = torch.argmax(outputs, dim=1)  
    predictions = predictions.numpy()
# Convert predictions back to original bucket values if needed
class_to_bucket = {v: k for k, v in bucket_to_class.items()}
predicted_buckets = [class_to_bucket[pred] for pred in predictions]

# Calculate Overall F1 Score and Accuracy
overall_f1 = f1_score(y_val_class_indices, predictions, average="macro")
overall_accuracy = accuracy_score(y_val_class_indices, predictions)
print(f"Overall F1 Score: {overall_f1:.4f}")
print(f"Overall Accuracy: {overall_accuracy:.4f}")

# Classwise Metrics
classwise_report = classification_report(y_val_class_indices, predictions, output_dict=True)
print("\nClasswise Metrics:")
for class_label, metrics in classwise_report.items():
    if class_label not in ["accuracy", "macro avg", "weighted avg"]:
        print(f"Class {class_label}:")
        print(f"  F1 Score: {metrics['f1-score']:.4f}")
        print(f"  Accuracy: {metrics['precision'] * metrics['recall']:.4f}")
