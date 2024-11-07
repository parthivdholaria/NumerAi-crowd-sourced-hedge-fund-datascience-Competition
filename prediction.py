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
from datetime import datetime
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
model_path = "saved_models/model.pth"  # Path to the saved model file
features_file_path = "saved_models/features.txt"  # File containing the selected features
live_data_path = "data/live.parquet"
predictions_dir = "predictions"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path, map_location=device)
model.eval()  # Set the model to evaluation mode

with open(features_file_path, "r") as f:
    selected_features = [line.strip() for line in f.readlines()]

print(f"Selected features for prediction: {selected_features}")

live_data = pd.read_parquet(live_data_path, columns=selected_features)
def preprocess_data(data_df, window_len, features):
    Xraw = data_df[features]
    Xraw_filled = Xraw.fillna(-1)  # Replace NaN values with -1

    new_data = []
    padding = torch.full((window_len - 1, len(features)), -1)  # -1 padding for sequence

    # Convert Xraw_filled to a tensor
    Xraw_tensor = torch.tensor(Xraw_filled.values, dtype=torch.float32)

    # Concatenate padding and data
    Xraw_padded = torch.cat((padding, Xraw_tensor), dim=0)

    # Sliding window approach
    for start in range(len(Xraw_tensor)):
        new_row_data = Xraw_padded[start : start + window_len].reshape(window_len, len(features))
        new_data.append(new_row_data)

    # Return the final tensor
    return torch.stack(new_data)

# Apply preprocessing to live data
window_len = 5  # Window length as defined in reference code
X_live = preprocess_data(live_data, window_len, selected_features).to(device)

with torch.no_grad():
    outputs = model(X_live)
    
    if torch.isnan(outputs).any():
        outputs = torch.where(torch.isnan(outputs), torch.tensor(0.5, device=outputs.device), outputs)
    
    predictions = torch.argmax(outputs, dim=1).cpu().numpy() 

live_mapping = {0: 0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1}
mapped_predictions = [live_mapping[pred] for pred in predictions]


predictions_df = pd.DataFrame({
    "id": live_data.index, 
    "prediction": mapped_predictions
})

timestamp = datetime.now().strftime("%d-%m-%Y")
predictions_filename = f"{predictions_dir}/{timestamp}_predictions.csv"

predictions_df.to_csv(predictions_filename, index=False)
print(f"Predictions saved to {predictions_filename}")