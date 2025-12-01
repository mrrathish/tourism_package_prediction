import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-customer-prediction-experiment")

api = HfApi()

Xtrain_path = "hf://datasets/Retheesh/tourism-customer-prediction/Xtrain.csv"
Xtest_path = "hf://datasets/Retheesh/tourism-customer-prediction/Xtest.csv"
ytrain_path = "hf://datasets/Retheesh/tourism-customer-prediction/ytrain.csv"
ytest_path = "hf://datasets/Retheesh/tourism-customer-prediction/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze()
ytest = pd.read_csv(ytest_path).squeeze()

# Define numeric and categorical features
numeric_features = [
    'Age', 'NumberOfPersonVisiting', 'PreferredPropertyStar', 
    'NumberOfTrips', 'NumberOfChildrenVisiting', 'MonthlyIncome',
    'NumberOfFollowups', 'DurationOfPitch'
]

categorical_features = [
    'TypeofContact', 'CityTier', 'Occupation', 'Gender', 
    'MaritalStatus', 'Passport', 'OwnCar', 'Designation', 
    'PitchSatisfactionScore', 'ProductPitched'
]

# Preprocessor
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)

# Simplified hyperparameter grid for faster execution
param_grid = {
    'randomforestclassifier__n_estimators': [100, 150],
    'randomforestclassifier__max_depth': [10, 15],
    'randomforestclassifier__min_samples_split': [2, 5]
}

# Pipeline
model_pipeline = make_pipeline(preprocessor, rf_model)

with mlflow.start_run():
    # Grid Search with reduced CV for faster execution
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1, scoring='roc_auc')
    grid_search.fit(Xtrain, ytrain)

    # Log parameter sets
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_roc_auc", mean_score)

    # Best model
    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Predictions
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)
    y_pred_proba_test = best_model.predict_proba(Xtest)[:, 1]

    # Metrics
    train_accuracy = accuracy_score(ytrain, y_pred_train)
    test_accuracy = accuracy_score(ytest, y_pred_test)

    train_precision = precision_score(ytrain, y_pred_train, zero_division=0)
    test_precision = precision_score(ytest, y_pred_test, zero_division=0)

    train_recall = recall_score(ytrain, y_pred_train, zero_division=0)
    test_recall = recall_score(ytest, y_pred_test, zero_division=0)

    train_f1 = f1_score(ytrain, y_pred_train, zero_division=0)
    test_f1 = f1_score(ytest, y_pred_test, zero_division=0)

    test_roc_auc = roc_auc_score(ytest, y_pred_proba_test)

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_precision": train_precision,
        "test_precision": test_precision,
        "train_recall": train_recall,
        "test_recall": test_recall,
        "train_f1": train_f1,
        "test_f1": test_f1,
        "test_roc_auc": test_roc_auc
    })

    # Save the model locally
    model_path = "tourism_customer_prediction_model.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "Retheesh/tourism-customer-prediction-model"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    api.upload_file(
        path_or_fileobj="tourism_customer_prediction_model.joblib",
        path_in_repo="tourism_customer_prediction_model.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
