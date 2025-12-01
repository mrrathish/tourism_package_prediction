import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/Retheesh/tourism-customer-prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Data preprocessing
# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['MonthlyIncome'].fillna(df['MonthlyIncome'].median(), inplace=True)

# Encode categorical variables
label_encoders = {}
categorical_columns = ['TypeofContact', 'Occupation', 'Gender', 'MaritalStatus', 'Designation', 'ProductPitched']

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Define target variable
target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col, 'CustomerID'])
y = df[target_col]

print(f"Features shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data preparation completed successfully.")
print(f"Training set: {Xtrain.shape}, Test set: {Xtest.shape}")

files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id="Retheesh/tourism-customer-prediction",
        repo_type="dataset",
    )
    print(f"Uploaded {file_path} to Hugging Face")
