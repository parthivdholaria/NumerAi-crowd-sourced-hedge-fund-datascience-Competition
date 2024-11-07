import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import json
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

data_dir = "data"
model_dir = "saved_models"
feature_file_path = os.path.join(model_dir, "features.txt")
data_version = "v5.0"
train_data_path = os.path.join(data_dir, "train.parquet")
features_json_path = os.path.join(data_dir, "features.json")

# Load feature metadata
with open(features_json_path, "r") as f:
    feature_metadata = json.load(f)
feature_cols = feature_metadata["feature_sets"]["medium"]
target_col = "target"

# Load training data
training_data = pd.read_parquet(train_data_path, columns=["era"] + feature_cols + [target_col])
training_data["era"] = training_data["era"].astype(str)

# Filter recent training data if needed
training_data = training_data[-200000:]
corr_list = {feature: training_data[feature].corr(training_data[target_col]) for feature in feature_cols}
sorted_features = sorted(corr_list, key=corr_list.get, reverse=True)
final_feature = sorted_features[:40]  # Selecting top 40 features

# Save the selected features to a text file
with open(feature_file_path, "w") as f:
    for feature in final_feature:
        f.write(feature + "\n")
print(f"Selected features saved to {feature_file_path}")

targets_df = training_data[["era", target_col] + final_feature]
def create_dataset_with_window(Xraw, yraw, window_len: int):
    Xraw_filled = Xraw.fillna(-1)
    yraw_filled = yraw.fillna(-1)
    
    new_data = []
    new_cols = []
    new_labels = []
    
    for col_idx in range(window_len):
        local_new_cols = [f"{col}_ts{col_idx}" for col in Xraw_filled.columns]
        new_cols.extend(local_new_cols)

    for start in tqdm(range(0, len(Xraw_filled) - window_len + 1), desc="Creating Dataset"):
        new_row_data = Xraw_filled.iloc[start : start + window_len].values.reshape(-1)
        new_label_data = yraw_filled.iloc[start + window_len - 1]
        new_data.append(new_row_data)
        new_labels.append(new_label_data)

    return pd.DataFrame(new_data, columns=new_cols).astype(float), pd.Series(new_labels).astype(float)


Xraw = targets_df[final_feature]
yraw = targets_df[target_col]
window_len = 5

X, y = create_dataset_with_window(Xraw, yraw, window_len)
# Convert data to PyTorch tensors
X_numpy = X.values
y_numpy = y.values
feature_size = len(final_feature)
X_reshaped = X_numpy.reshape(-1, window_len, feature_size)

X_tensor = torch.tensor(X_reshaped, dtype=torch.float32)
y_tensor = torch.tensor(y_numpy, dtype=torch.float32)

# Define DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
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
        
input_size = len(final_feature)
hidden_size = 128
output_size = 5
lr = 0.001
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ClassificationLSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size, lr=lr)
model.train_model(train_loader, num_epochs, device)
model_save_path = os.path.join(model_dir, "model.pth")
torch.save(model, model_save_path)
print(f"Trained model saved to {model_save_path}")