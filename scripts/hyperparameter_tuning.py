import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as ImbPipeline
from scipy import sparse
import matplotlib.pyplot as plt
import joblib
from scipy.stats import randint, uniform
import logging
class FraudDetectionModel:
    def __init__(self, credit_file, fraud_file):
        """
        Initialize the FraudDetectionModel class with file paths for datasets.
        """
        self.credit_file = credit_file
        self.fraud_file = fraud_file
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def preprocess(self, dataset_type):
        """
        Preprocess the dataset based on the type (credit or fraud).
        Handles datetime columns, categorical columns, and memory optimization.
        """
        if dataset_type == 'credit':
            data = pd.read_csv(self.credit_file)
            target = 'Class'
        else:
            data = pd.read_csv(self.fraud_file)
            target = 'class'

        # Handle datetime columns
        date_cols = []
        for col in data.select_dtypes(include=['object']).columns:
            if col == target:
                continue
            try:
                data[col] = pd.to_datetime(data[col], errors='raise')
                date_cols.append(col)
            except:
                pass

        # Feature engineering for datetime columns
        for col in date_cols:
            data[f'{col}_year'] = data[col].dt.year
            data[f'{col}_month'] = data[col].dt.month
            data[f'{col}_day'] = data[col].dt.day
            data[f'{col}_hour'] = data[col].dt.hour
            data[f'{col}_minute'] = data[col].dt.minute
            data = data.drop(col, axis=1)

        # Handle categorical columns with memory optimization
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        high_cardinality_cols = []
        low_cardinality_cols = []

        # Separate columns by cardinality
        for col in categorical_cols:
            if data[col].nunique() > 50:
                high_cardinality_cols.append(col)
            else:
                low_cardinality_cols.append(col)

        # Process high cardinality columns
        if high_cardinality_cols:
            self.logger.info(f"Processing high cardinality columns: {high_cardinality_cols}")
            # Use label encoding for high cardinality
            for col in high_cardinality_cols:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))

        # Process low cardinality columns
        if low_cardinality_cols:
            self.logger.info(f"Processing low cardinality columns: {low_cardinality_cols}")
            data = pd.get_dummies(data, columns=low_cardinality_cols, drop_first=True, sparse=True)

        # Drop columns with all NaNs
        all_nan_cols = data.columns[data.isnull().all()].tolist()
        if all_nan_cols:
            self.logger.warning(f"Dropping columns with all NaNs: {all_nan_cols}")
            data = data.drop(all_nan_cols, axis=1)

        # Split into features and target
        X = data.drop(target, axis=1)
        y = data[target]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Create preprocessing pipeline
        numeric_features = X_train.select_dtypes(include=['number']).columns.tolist()
        numeric_transformer = ImbPipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])

        # Apply preprocessing
        X_train = numeric_transformer.fit_transform(X_train)
        X_test = numeric_transformer.transform(X_test)

        # Convert to sparse matrices if needed
        if isinstance(X_train, np.ndarray) and X_train.shape[1] > 1000:
            self.logger.warning("Converting to sparse matrices to save memory")
            X_train = sparse.csr_matrix(X_train)
            X_test = sparse.csr_matrix(X_test)

        return X_train, X_test, y_train, y_test

    def tune_random_forest(self, X_train, y_train):
        """
        Perform hyperparameter tuning for Random Forest using RandomizedSearchCV.
        """
        param_dist = {
            'n_estimators': randint(100, 300),
            'max_depth': randint(10, 30),
            'min_samples_split': randint(2, 10),
            'min_samples_leaf': randint(1, 4),
        }
        rf = RandomForestClassifier(random_state=42)
        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=10,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42,
        )
        random_search.fit(X_train, y_train)
        return random_search.best_estimator_

    def tune_gradient_boosting(self, X_train, y_train):
        """
        Perform hyperparameter tuning for Gradient Boosting using RandomizedSearchCV.
        """
        param_dist = {
            'n_estimators': randint(100, 300),
            'learning_rate': uniform(0.01, 0.2),
            'max_depth': randint(3, 7),
            'min_samples_split': randint(2, 10),
        }
        gb = GradientBoostingClassifier(random_state=42)
        random_search = RandomizedSearchCV(
            estimator=gb,
            param_distributions=param_dist,
            n_iter=10,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42,
        )
        random_search.fit(X_train, y_train)
        return random_search.best_estimator_

    def plot_roc_curve(self, y_true, y_proba, model_name, dataset_type):
        """
        Plot the ROC curve and save it as an image.
        """
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_score = roc_auc_score(y_true, y_proba)

        plt.figure()
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {dataset_type} Dataset')
        plt.legend(loc='lower right')
        plt.savefig(f"roc_curve_{dataset_type}_{model_name}.png")
        plt.close()

    def log_to_mlflow(self, model, X_test, y_test, model_name, dataset_type):
        """
        Log model parameters, metrics, and artifacts to MLflow.
        """
        with mlflow.start_run(run_name=f"{dataset_type}_{model_name}"):
            # Log parameters
            mlflow.log_params(model.get_params())

            # Log metrics
            y_proba = model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_proba)
            mlflow.log_metric("AUC", auc_score)

            # Log model
            mlflow.sklearn.log_model(model, f"{dataset_type}_{model_name}_model")

            # Log ROC curve plot
            self.plot_roc_curve(y_test, y_proba, model_name, dataset_type)
            mlflow.log_artifact(f"roc_curve_{dataset_type}_{model_name}.png")

    def run(self, dataset_type):
        """
        Run the pipeline for the specified dataset type.
        """
        if dataset_type not in ["credit", "fraud"]:
            raise ValueError("Invalid dataset type. Use 'credit' or 'fraud'.")

        # Set MLflow experiment
        mlflow.set_experiment(f"Hyperparameter_Tuning_{dataset_type}")

        # Preprocess data
        X_train, X_test, y_train, y_test = self.preprocess(dataset_type)

        if dataset_type == "credit":
            # Tune Random Forest for Credit Dataset
            print("Tuning Random Forest for Credit Dataset...")
            best_rf = self.tune_random_forest(X_train, y_train)
            print("Best Random Forest Parameters:", best_rf.get_params())

            # Save Random Forest model
            joblib.dump(best_rf, "best_random_forest_credit.pkl")

            # Log to MLflow
            self.log_to_mlflow(best_rf, X_test, y_test, "Random_Forest", "Credit")

        elif dataset_type == "fraud":
            # Tune Gradient Boosting for Fraud Dataset
            print("Tuning Gradient Boosting for Fraud Dataset...")
            best_gb = self.tune_gradient_boosting(X_train, y_train)
            print("Best Gradient Boosting Parameters:", best_gb.get_params())

            # Save Gradient Boosting model
            joblib.dump(best_gb, "best_gradient_boosting_fraud.pkl")

            # Log to MLflow
            self.log_to_mlflow(best_gb, X_test, y_test, "Gradient_Boosting", "Fraud")


if __name__ == "__main__":
    # File paths
    credit_file = "E:/DS+ML/AIM3/WEEK.12/Data/featured_credit_data.csv"
    fraud_file = "E:/DS+ML/AIM3/WEEK.12/Data/featured_fraud_data.csv"

    # Create an instance of the class
    fraud_detection = FraudDetectionModel(credit_file, fraud_file)

    # Run the pipeline for the credit dataset
    fraud_detection.run("credit")

    # Run the pipeline for the fraud dataset
    fraud_detection.run("fraud")