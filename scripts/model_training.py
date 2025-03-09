import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, LSTM, SimpleRNN, Flatten
import matplotlib.pyplot as plt

class ModelTrainer:
    def __init__(self, fraud_file, credit_file):
        self.fraud_file = fraud_file
        self.credit_file = credit_file
        self.logger = self.setup_logger()
        self.results = {'credit': [], 'fraud': []}
        
        # Load and preprocess data
        self.credit_X_train, self.credit_X_test, self.credit_y_train, self.credit_y_test = self.preprocess('credit')
        self.fraud_X_train, self.fraud_X_test, self.fraud_y_train, self.fraud_y_test = self.preprocess('fraud')

    def setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger()

    def preprocess(self, dataset_type):
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

    def evaluate_model(self, model, X_test, y_test, model_name, dataset_type):
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba > 0.3).astype(int)  # Adjusted threshold for better recall
        else:
            y_pred = model.predict(X_test)
            
        auc = roc_auc_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        print(f"\n{'='*40}")
        print(f"{dataset_type.capitalize()} Dataset - {model_name} Results")
        print(f"AUC: {auc:.3f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("Confusion Matrix:")
        print(conf_matrix)
        print(f"{'='*40}\n")
        
        self.results[dataset_type].append({
            'Model': model_name,
            'AUC': auc,
            'Accuracy': report['accuracy'],
            'F1-Score': report['weighted avg']['f1-score']
        })
        
        self.logger.info(f"{dataset_type.capitalize()} - {model_name}: AUC={auc:.3f}, Accuracy={report['accuracy']:.3f}")

    def train_sklearn_models(self, dataset_type):
        models = {
            'Logistic Regression': ImbPipeline([
                ('smote', SMOTE(random_state=42)),
                ('classifier', LogisticRegression(
                    class_weight='balanced', 
                    max_iter=1000,
                    penalty='l2',
                    C=0.1
                ))
            ]),
            'Decision Tree': ImbPipeline([
                ('smote', SMOTE(random_state=42)),
                ('classifier', DecisionTreeClassifier(
                    class_weight='balanced',
                    max_depth=10,
                    min_samples_leaf=10
                ))
            ]),
            'Random Forest': ImbPipeline([
                ('smote', SMOTE(random_state=42)),
                ('classifier', RandomForestClassifier(
                    class_weight='balanced',
                    n_estimators=200,
                    max_depth=15,
                    min_samples_leaf=5
                ))
            ]),
            'Gradient Boosting': ImbPipeline([
                ('smote', SMOTE(random_state=42)),
                ('classifier', GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=5
                ))
            ]),
            'MLP': ImbPipeline([
                ('smote', SMOTE(random_state=42)),
                ('classifier', MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    max_iter=1000,
                    early_stopping=True,
                    alpha=0.0001,
                    random_state=42
                ))
            ])
        }
        
        X_train = getattr(self, f'{dataset_type}_X_train')
        X_test = getattr(self, f'{dataset_type}_X_test')
        y_train = getattr(self, f'{dataset_type}_y_train')
        y_test = getattr(self, f'{dataset_type}_y_test')
        
        print(f"\n{'='*50}")
        print(f"Training Traditional Models on {dataset_type.capitalize()} Dataset")
        print(f"{'='*50}")
        
        for name, pipeline in models.items():
            print(f"\nTraining {name}...")
            pipeline.fit(X_train, y_train)
            self.evaluate_model(pipeline.named_steps['classifier'], 
                               X_test, y_test, name, dataset_type)

    def build_dl_model(self, model_type, input_shape):
        model = Sequential()
        
        if model_type == 'CNN':
            model.add(Conv1D(64, 3, activation='relu', input_shape=input_shape))
            model.add(MaxPooling1D(2))
            model.add(Flatten())  # Now properly imported
        elif model_type == 'RNN':
            model.add(SimpleRNN(50, input_shape=input_shape))
        elif model_type == 'LSTM':
            model.add(LSTM(50, input_shape=input_shape))
            
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC', 'accuracy'])
        return model

    def train_dl_models(self, dataset_type):
        model_types = ['CNN', 'RNN', 'LSTM']
        X_train = getattr(self, f'{dataset_type}_X_train')
        X_test = getattr(self, f'{dataset_type}_X_test')
        y_train = getattr(self, f'{dataset_type}_y_train')
        y_test = getattr(self, f'{dataset_type}_y_test')
        
        # Apply SMOTE for class imbalance
        smote = SMOTE(random_state=42)
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
        
        print(f"\n{'='*50}")
        print(f"Training Deep Learning Models on {dataset_type.capitalize()} Dataset")
        print(f"{'='*50}")
        
        for model_type in model_types:
            print(f"\nTraining {model_type}...")
            
            # Reshape data
            if model_type == 'CNN':
                X_train_reshaped = X_train_sm.reshape(X_train_sm.shape[0], X_train_sm.shape[1], 1)
                X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                input_shape = (X_train_sm.shape[1], 1)
            else:
                X_train_reshaped = X_train_sm.reshape(X_train_sm.shape[0], 1, X_train_sm.shape[1])
                X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
                input_shape = (1, X_train_sm.shape[1])
            
            model = self.build_dl_model(model_type, input_shape)
            
            print(f"Training {model_type} with input shape {input_shape}...")
            history = model.fit(
                X_train_reshaped, y_train_sm,
                validation_split=0.2,
                epochs=50,
                batch_size=64,
                class_weight=self.get_class_weights(y_train_sm),
                verbose=0
            )
            
            # Evaluate
            print(f"Evaluating {model_type}...")
            y_pred = (model.predict(X_test_reshaped) > 0.5).astype(int)
            loss, auc, accuracy = model.evaluate(X_test_reshaped, y_test, verbose=0)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            self.results[dataset_type].append({
                'Model': model_type,
                'AUC': auc,
                'Accuracy': accuracy,
                'F1-Score': report['weighted avg']['f1-score']
            })
            
            print(f"\n{model_type} Results:")
            print(f"AUC: {auc:.3f}")
            print(f"Accuracy: {accuracy:.3f}")
            print(f"F1-Score: {report['weighted avg']['f1-score']:.3f}")
            print(confusion_matrix(y_test, y_pred))
            print("="*40)
            
            self.logger.info(f"{dataset_type.capitalize()} - {model_type}: AUC={auc:.3f}, Accuracy={accuracy:.3f}")

    def get_class_weights(self, y):
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        return {i: weight for i, weight in enumerate(class_weights)}

    def plot_comparisons(self):
        # Now plt is properly imported
        fig, axes = plt.subplots(3, 2, figsize=(18, 12))
        metrics = ['AUC', 'Accuracy', 'F1-Score']
        datasets = ['credit', 'fraud']
        
        for i, metric in enumerate(metrics):
            for j, ds in enumerate(datasets):
                ax = axes[i, j]
                data = pd.DataFrame(self.results[ds]).set_index('Model')
                data[metric].plot(kind='bar', ax=ax, legend=False, color='skyblue')
                ax.set_title(f'{metric} - {ds.capitalize()} Dataset', fontsize=14)
                ax.set_ylabel(metric, fontsize=12)
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.3f', padding=3)
                
                if i == 0 and j == 1:
                    handles = [plt.Rectangle((0,0),1,1, color='skyblue') for _ in data.index]
                    fig.legend(handles, data.index, loc='upper center', ncol=4, fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.suptitle('Model Performance Comparison', fontsize=16, y=0.98)
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("\nComparison plots saved to model_comparison.png")

    def print_summary_table(self):
        print("\n\n")
        print("="*80)
        print(f"{'Model Performance Summary':^80}")
        print("="*80)
        for ds in ['credit', 'fraud']:
            print(f"\n{'='*30} {ds.capitalize()} Dataset {'='*30}")
            print(f"{'Model':<20} | {'AUC':<10} | {'Accuracy':<10} | {'F1-Score':<10}")
            print("-"*70)
            for result in self.results[ds]:
                print(f"{result['Model']:<20} | {result['AUC']:<10.3f} | {result['Accuracy']:<10.3f} | {result['F1-Score']:<10.3f}")
        print("="*80)

    def run(self):
        self.logger.info("Starting model training and evaluation process")
        
        # Train traditional models
        for dataset in ['credit', 'fraud']:
            self.train_sklearn_models(dataset)
        
        # Train deep learning models
        for dataset in ['credit', 'fraud']:
            self.train_dl_models(dataset)
        
        # Generate comparison plots
        self.plot_comparisons()
        
        # Print summary table
        self.print_summary_table()
        
        self.logger.info("Training and evaluation process completed successfully")

if __name__ == "__main__":
    trainer = ModelTrainer(
        fraud_file="E:/DS+ML/AIM3/WEEK.12/Data/featured_fraud_data.csv",
        credit_file="E:/DS+ML/AIM3/WEEK.12/Data/featured_credit_data.csv"
    )
    trainer.run()