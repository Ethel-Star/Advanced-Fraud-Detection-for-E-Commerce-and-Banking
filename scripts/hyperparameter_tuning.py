import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as ImbPipeline
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns
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
        self.logger = self._setup_logging()

    def _setup_logging(self):
        """Configure logging for the class."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            os.makedirs("logs", exist_ok=True)
            # File handler for saving logs to a file
            file_handler = logging.FileHandler("logs/model_training.log")
            # Stream handler for console output
            stream_handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            stream_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO)
        return logger

    def preprocess(self, dataset_type):
        """
        Preprocess the dataset based on the type (credit or fraud).
        Handles datetime columns, categorical columns, and memory optimization.
        """
        try:
            self.logger.info(f"Preprocessing {dataset_type} dataset...")
            if dataset_type == "credit":
                data = pd.read_csv(self.credit_file)
                target = "Class"
            else:
                data = pd.read_csv(self.fraud_file)
                target = "class"

            # Handle datetime columns
            date_cols = []
            for col in data.select_dtypes(include=["object"]).columns:
                if col == target:
                    continue
                try:
                    data[col] = pd.to_datetime(data[col], errors="raise")
                    date_cols.append(col)
                except Exception as e:
                    self.logger.warning(f"Could not convert column {col} to datetime: {e}")

            # Feature engineering for datetime columns
            for col in date_cols:
                data[f"{col}_year"] = data[col].dt.year
                data[f"{col}_month"] = data[col].dt.month
                data[f"{col}_day"] = data[col].dt.day
                data[f"{col}_hour"] = data[col].dt.hour
                data[f"{col}_minute"] = data[col].dt.minute
                data = data.drop(col, axis=1)

            # Handle categorical columns with memory optimization
            categorical_cols = data.select_dtypes(include=["object"]).columns.tolist()
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
            numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
            numeric_transformer = ImbPipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", RobustScaler()),
                ]
            )

            # Apply preprocessing
            X_train = numeric_transformer.fit_transform(X_train)
            X_test = numeric_transformer.transform(X_test)

            # Convert to sparse matrices if needed
            if isinstance(X_train, np.ndarray) and X_train.shape[1] > 1000:
                self.logger.warning("Converting to sparse matrices to save memory")
                X_train = sparse.csr_matrix(X_train)
                X_test = sparse.csr_matrix(X_test)

            self.logger.info(f"Preprocessing completed for {dataset_type} dataset.")
            return X_train, X_test, y_train, y_test

        except Exception as e:
            self.logger.error(f"Error during preprocessing: {e}", exc_info=True)
            raise

    def tune_random_forest(self, X_train, y_train):
        """
        Perform hyperparameter tuning for Random Forest using RandomizedSearchCV.
        """
        try:
            self.logger.info("Starting hyperparameter tuning for Random Forest...")
            param_dist = {
                "n_estimators": randint(100, 300),
                "max_depth": randint(10, 30),
                "min_samples_split": randint(2, 10),
                "min_samples_leaf": randint(1, 4),
            }
            rf = RandomForestClassifier(random_state=42)
            random_search = RandomizedSearchCV(
                estimator=rf,
                param_distributions=param_dist,
                n_iter=10,
                cv=3,
                scoring="roc_auc",
                n_jobs=-1,
                random_state=42,
            )
            random_search.fit(X_train, y_train)
            self.logger.info("Random Forest hyperparameter tuning completed.")
            return random_search.best_estimator_

        except Exception as e:
            self.logger.error(f"Error during Random Forest tuning: {e}", exc_info=True)
            raise

    def tune_gradient_boosting(self, X_train, y_train):
        """
        Perform hyperparameter tuning for Gradient Boosting using RandomizedSearchCV.
        """
        try:
            self.logger.info("Starting hyperparameter tuning for Gradient Boosting...")
            param_dist = {
                "n_estimators": randint(100, 300),
                "learning_rate": uniform(0.01, 0.2),
                "max_depth": randint(3, 7),
                "min_samples_split": randint(2, 10),
            }
            gb = GradientBoostingClassifier(random_state=42)
            random_search = RandomizedSearchCV(
                estimator=gb,
                param_distributions=param_dist,
                n_iter=10,
                cv=3,
                scoring="roc_auc",
                n_jobs=-1,
                random_state=42,
            )
            random_search.fit(X_train, y_train)
            self.logger.info("Gradient Boosting hyperparameter tuning completed.")
            return random_search.best_estimator_

        except Exception as e:
            self.logger.error(f"Error during Gradient Boosting tuning: {e}", exc_info=True)
            raise

    def plot_roc_curve(self, y_true, y_proba, model_name, dataset_type):
        """
        Plot the ROC curve and save it as an image.
        """
        try:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc_score = roc_auc_score(y_true, y_proba)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.3f})")
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {dataset_type} Dataset")
            plt.legend(loc="lower right")
            plt.savefig(f"roc_curve_{dataset_type}_{model_name}.png")
            plt.close()
            self.logger.info(f"ROC curve saved for {dataset_type} dataset.")

        except Exception as e:
            self.logger.error(f"Error plotting ROC curve: {e}", exc_info=True)
            raise

    def plot_feature_importance(self, model, feature_names, model_name, dataset_type):
        """
        Plot feature importance for tree-based models.
        """
        try:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]

                plt.figure(figsize=(10, 6))
                plt.title(f"Feature Importance - {model_name} ({dataset_type})")
                plt.bar(range(len(importances)), importances[indices], align="center")
                plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
                plt.tight_layout()
                plt.savefig(f"feature_importance_{dataset_type}_{model_name}.png")
                plt.close()
                self.logger.info(f"Feature importance plot saved for {dataset_type} dataset.")
            else:
                self.logger.warning(f"Model {model_name} does not support feature importance.")

        except Exception as e:
            self.logger.error(f"Error plotting feature importance: {e}", exc_info=True)
            raise

    def plot_confusion_matrix(self, y_true, y_pred, model_name, dataset_type):
        """
        Plot confusion matrix.
        """
        try:
            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap="Blues")
            plt.title(f"Confusion Matrix - {model_name} ({dataset_type})")
            plt.savefig(f"confusion_matrix_{dataset_type}_{model_name}.png")
            plt.close()
            self.logger.info(f"Confusion matrix saved for {dataset_type} dataset.")

        except Exception as e:
            self.logger.error(f"Error plotting confusion matrix: {e}", exc_info=True)
            raise

    def log_to_mlflow(self, model, X_test, y_test, model_name, dataset_type, feature_names=None):
        """
        Log model parameters, metrics, and artifacts to MLflow.
        """
        try:
            with mlflow.start_run(run_name=f"{dataset_type}_{model_name}"):
                # Log parameters
                mlflow.log_params(model.get_params())

                # Log metrics
                y_proba = model.predict_proba(X_test)[:, 1]
                y_pred = model.predict(X_test)
                auc_score = roc_auc_score(y_test, y_proba)
                mlflow.log_metric("AUC", auc_score)

                # Classification report
                report = classification_report(y_test, y_pred, output_dict=True)
                mlflow.log_metrics(report["weighted avg"])

                # Log model
                mlflow.sklearn.log_model(model, f"{dataset_type}_{model_name}_model")

                # Log ROC curve plot
                self.plot_roc_curve(y_test, y_proba, model_name, dataset_type)
                mlflow.log_artifact(f"roc_curve_{dataset_type}_{model_name}.png")

                # Log confusion matrix
                self.plot_confusion_matrix(y_test, y_pred, model_name, dataset_type)
                mlflow.log_artifact(f"confusion_matrix_{dataset_type}_{model_name}.png")

                # Log feature importance
                if feature_names is not None:
                    self.plot_feature_importance(model, feature_names, model_name, dataset_type)
                    mlflow.log_artifact(f"feature_importance_{dataset_type}_{model_name}.png")

                self.logger.info(f"MLflow logging completed for {dataset_type} dataset.")

        except Exception as e:
            self.logger.error(f"Error during MLflow logging: {e}", exc_info=True)
            raise
    def run(self, dataset_type):
        """
        Run the pipeline for the specified dataset type.
        """
        try:
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
                self.log_to_mlflow(
                    best_rf,
                    X_test,
                    y_test,
                    "Random_Forest",
                    "Credit",
                    feature_names=X_train.columns if hasattr(X_train, "columns") else None,
                )

            elif dataset_type == "fraud":
                # Tune Gradient Boosting for Fraud Dataset
                print("Tuning Gradient Boosting for Fraud Dataset...")
                best_gb = self.tune_gradient_boosting(X_train, y_train)
                print("Best Gradient Boosting Parameters:", best_gb.get_params())

                # Save Gradient Boosting model
                joblib.dump(best_gb, "best_gradient_boosting_fraud.pkl")

                # Log to MLflow
                self.log_to_mlflow(
                    best_gb,
                    X_test,
                    y_test,
                    "Gradient_Boosting",
                    "Fraud",
                    feature_names=X_train.columns if hasattr(X_train, "columns") else None,
                )

        except Exception as e:
            self.logger.error(f"Pipeline failed for {dataset_type}: {e}", exc_info=True)
            raise