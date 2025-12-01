from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os
import pandas as pd

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# First, create the dataset repository
repo_id = "Retheesh/tourism-customer-prediction"
repo_type = "dataset"

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repository '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"Dataset repository '{repo_id}' not found. Creating new repository...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset repository '{repo_id}' created.")

# Upload the tourism.csv file
try:
    # Read the local file to verify it exists
    df = pd.read_csv("tourism_project/data/tourism.csv")
    print(f"Local dataset loaded successfully. Shape: {df.shape}")

    # Upload the dataset file
    api.upload_file(
        path_or_fileobj="tourism_project/data/tourism.csv",
        path_in_repo="tourism.csv",
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print("Dataset uploaded successfully to Hugging Face Hub")

except FileNotFoundError:
    print("Error: tourism.csv file not found in tourism_project/data/")
    print("Please make sure the file exists in the correct location")
except Exception as e:
    print(f"Error uploading dataset: {e}")
