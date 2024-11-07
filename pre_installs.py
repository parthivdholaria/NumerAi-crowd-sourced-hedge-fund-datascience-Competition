import os
from numerapi import NumerAPI
import pandas as pd
import json
import gdown

# Create necessary directories
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)
os.makedirs("predictions", exist_ok=True)

# Initialize NumerAPI
napi = NumerAPI()

# Set the data version and number of eras to include
DATA_VERSION = "v5.0"
NUM_ERAS = 106

# Download training data and features metadata
napi.download_dataset(f"{DATA_VERSION}/train.parquet", dest_path="data/train.parquet")
napi.download_dataset(f"{DATA_VERSION}/features.json", dest_path="data/features.json")

# Load features metadata
with open("data/features.json", "r") as f:
    feature_metadata = json.load(f)
feature_cols = feature_metadata["feature_sets"]["medium"]
target_cols = feature_metadata["targets"]

# Load and filter training data
training_data = pd.read_parquet(
    "data/train.parquet",
    columns=["era"] + feature_cols + ["target"]
)
training_data["era"] = training_data["era"].astype(str)

# Filter for the most recent eras
unique_eras = sorted(training_data["era"].unique(), reverse=True)
recent_eras = unique_eras[:NUM_ERAS]
training_data = training_data[training_data["era"].isin(recent_eras)]

# Save processed training data
training_data.to_parquet("data/training_data_filtered.parquet")

# Download live data
napi.download_dataset(f"{DATA_VERSION}/live.parquet", dest_path="data/live.parquet")

# Load and save live data with selected features
live_features = pd.read_parquet("data/live.parquet", columns=feature_cols)
live_features.to_parquet("data/live_features.parquet")

# Download and process validation data
napi.download_dataset(f"{DATA_VERSION}/validation.parquet", dest_path="data/validation.parquet")

validation_data = pd.read_parquet(
    "data/validation.parquet",
    columns=["era", "data_type"] + feature_cols + target_cols
)
validation_data = validation_data[validation_data["data_type"] == "validation"]
validation_data.drop(columns=["data_type"], inplace=True)

# Save filtered validation data
validation_data.to_parquet("data/validation_data_filtered.parquet")

print("Data download and setup completed successfully.")



model_id = "1SVO5qFosW55cpqmqnYaYp83AZ_4J3PVp"
url = f"https://drive.google.com/uc?id={model_id}"
output_path = f"saved_models/model.pth"
gdown.download(url, output_path, quiet=False)

features_id = "1RThPx3CgIYhlEBhy0-yNSU9OWtVK9Ff4"
url = f"https://drive.google.com/uc?id={features_id}"
output_path = f"saved_models/features.txt"
gdown.download(url, output_path, quiet=False)