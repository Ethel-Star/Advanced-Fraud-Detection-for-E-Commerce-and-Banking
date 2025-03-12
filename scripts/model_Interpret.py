import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lime.lime_tabular import LimeTabularExplainer

class ModelExplanation:
    def __init__(self, credit_data_path, fraud_data_path, credit_model_path, fraud_model_path, target_column_credit, target_column_fraud):
        # Load data
        self.credit_data = pd.read_csv(credit_data_path)
        self.fraud_data = pd.read_csv(fraud_data_path)
        
        # Load models
        self.rf_model = joblib.load(credit_model_path)
        self.gb_model = joblib.load(fraud_model_path)
        
        # Prepare data for LIME
        self.target_column_credit = target_column_credit
        self.target_column_fraud = target_column_fraud
        
        # Credit Data Preparation
        self.X_credit = self.credit_data.drop(columns=[self.target_column_credit])
        self.y_credit = self.credit_data[self.target_column_credit]
        
        # Fraud Data Preparation
        self.X_fraud = self.fraud_data.drop(columns=[self.target_column_fraud])
        self.y_fraud = self.fraud_data[self.target_column_fraud]
        
        # Encode categorical columns (MATCH TRAINING PREPROCESSING)
        categorical_cols = self.X_fraud.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            self.X_fraud[col] = le.fit_transform(self.X_fraud[col].astype(str))
        
        # Convert to numeric (MATCH TRAINING PIPELINE)
        self.X_fraud = self.X_fraud.apply(pd.to_numeric, errors='coerce')
        
        # Split data
        self.X_credit_train, self.X_credit_test, self.y_credit_train, self.y_credit_test = train_test_split(
            self.X_credit, self.y_credit, test_size=0.3, random_state=42
        )
        self.X_fraud_train, self.X_fraud_test, self.y_fraud_train, self.y_fraud_test = train_test_split(
            self.X_fraud, self.y_fraud, test_size=0.3, random_state=42
        )
        # Align test columns with train
        self.X_fraud_test = self.X_fraud_test[self.X_fraud_train.columns]

    def lime_explanation(self, model, data, explainer, index, output_path, class_names):
        exp = explainer.explain_instance(
            data_row=data.iloc[index].values,
            predict_fn=model.predict_proba,
            num_features=5
        )
        fig = exp.as_pyplot_figure()
        plt.tight_layout()
        fig.savefig(output_path)
        plt.close()

    def explain_credit_data(self, instance_index, output_path):
        explainer_credit = LimeTabularExplainer(
            self.X_credit_train.values,
            training_labels=self.y_credit_train.values,
            mode='classification',
            class_names=['Not Fraud', 'Fraud'],
            feature_names=self.X_credit.columns.tolist(),
            discretize_continuous=True
        )
        self.lime_explanation(
            model=self.rf_model,
            data=self.X_credit_test,
            explainer=explainer_credit,
            index=instance_index,
            output_path=output_path,
            class_names=['Not Fraud', 'Fraud']
        )

    def explain_fraud_data(self, instance_index, output_path):
        explainer_fraud = LimeTabularExplainer(
            self.X_fraud_train.values,
            training_labels=self.y_fraud_train.values,
            mode='classification',
            class_names=['Not Fraud', 'Fraud'],
            feature_names=self.X_fraud.columns.tolist(),
            discretize_continuous=True
        )
        self.lime_explanation(
            model=self.gb_model,
            data=self.X_fraud_test,
            explainer=explainer_fraud,
            index=instance_index,
            output_path=output_path,
            class_names=['Not Fraud', 'Fraud']
        )

if __name__ == "__main__":
    credit_data_path = r"E:/DS+ML/AIM3/Final/Data/featured_credit_data.csv"
    fraud_data_path = r"E:/DS+ML/AIM3/Final/Data/featured_fraud_data.csv"
    credit_model_path = "best_random_forest_credit.pkl"
    fraud_model_path = "best_gradient_boosting_fraud.pkl"
    
    target_column_credit = 'Class'
    target_column_fraud = 'class'
    
    explanation = ModelExplanation(
        credit_data_path,
        fraud_data_path,
        credit_model_path,
        fraud_model_path,
        target_column_credit,
        target_column_fraud
    )
    
    explanation.explain_credit_data(0, "lime_credit_explanation.png")
    explanation.explain_fraud_data(0, "lime_fraud_explanation.png")
    print("LIME explanations saved successfully!")