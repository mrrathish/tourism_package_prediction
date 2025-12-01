import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/Retheesh/tourism_package_prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

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

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id="Retheesh/tourism_package_prediction",
        repo_type="dataset",
    )
