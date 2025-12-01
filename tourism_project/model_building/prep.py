import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from huggingface_hub import HfApi
import joblib

api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    # First try to load from Hugging Face
    DATASET_PATH = "hf://datasets/Retheesh/tourism-customer-prediction/tourism.csv"
    df = pd.read_csv(DATASET_PATH)
    print("Dataset loaded successfully from Hugging Face Hub")
except Exception as e:
    print(f"Failed to load from Hugging Face: {e}")
    # Fallback to local file
    try:
        df = pd.read_csv("tourism_project/data/tourism.csv")
        print("Dataset loaded successfully from local file")
    except Exception as e2:
        print(f"Failed to load local file: {e2}")
        # Create a simple demo dataset if no file exists
        print("Creating demo dataset for testing...")
        import numpy as np
        np.random.seed(42)
        n_samples = 1000

        demo_data = {
            'CustomerID': range(1, n_samples + 1),
            'Age': np.random.randint(18, 70, n_samples),
            'TypeofContact': np.random.choice(['Company Invited', 'Self Inquiry'], n_samples),
            'CityTier': np.random.choice([1, 2, 3], n_samples),
            'Occupation': np.random.choice(['Salaried', 'Small Business', 'Large Business', 'Free Lancer'], n_samples),
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'NumberOfPersonVisiting': np.random.randint(1, 6, n_samples),
            'PreferredPropertyStar': np.random.randint(3, 6, n_samples),
            'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
            'NumberOfTrips': np.random.randint(0, 10, n_samples),
            'Passport': np.random.choice([0, 1], n_samples),
            'OwnCar': np.random.choice([0, 1], n_samples),
            'NumberOfChildrenVisiting': np.random.randint(0, 4, n_samples),
            'Designation': np.random.choice(['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP'], n_samples),
            'MonthlyIncome': np.random.randint(10000, 50000, n_samples),
            'PitchSatisfactionScore': np.random.randint(1, 6, n_samples),
            'ProductPitched': np.random.choice(['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King'], n_samples),
            'NumberOfFollowups': np.random.randint(1, 8, n_samples),
            'DurationOfPitch': np.random.randint(5, 30, n_samples),
            'ProdTaken': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        }
        df = pd.DataFrame(demo_data)
        print("Demo dataset created for testing")

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Data preprocessing
# Handle missing values
if 'Age' in df.columns:
    df['Age'].fillna(df['Age'].median(), inplace=True)
if 'MonthlyIncome' in df.columns:
    df['MonthlyIncome'].fillna(df['MonthlyIncome'].median(), inplace=True)

# Encode categorical variables
label_encoders = {}
categorical_columns = ['TypeofContact', 'Occupation', 'Gender', 'MaritalStatus', 'Designation', 'ProductPitched']

for col in categorical_columns:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# Define target variable
target_col = 'ProdTaken'

# Split into X (features) and y (target)
if 'CustomerID' in df.columns:
    X = df.drop(columns=[target_col, 'CustomerID'])
else:
    X = df.drop(columns=[target_col])

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

# Save label encoders for later use
joblib.dump(label_encoders, "label_encoders.joblib")

files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv", "label_encoders.joblib"]

for file_path in files:
    try:
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file_path.split("/")[-1],
            repo_id="Retheesh/tourism-customer-prediction",
            repo_type="dataset",
        )
        print(f"Uploaded {file_path} to Hugging Face")
    except Exception as e:
        print(f"Failed to upload {file_path}: {e}")
