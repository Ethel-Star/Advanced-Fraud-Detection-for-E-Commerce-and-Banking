import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

# Unified logging configuration
def setup_logger(name, log_dir="logs", level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        # File handler
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        fh.setFormatter(formatter)
        # Stream handler
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(sh)
        logger.setLevel(level)
    return logger

class FraudFeatureEngineer:
    def __init__(self, fraud_data: pd.DataFrame):
        self.original_data = fraud_data.copy()
        self.engineered_data = None
        self.logger = setup_logger(self.__class__.__name__)

    def engineer_features(self) -> pd.DataFrame:
        """Full feature engineering pipeline"""
        try:
            self.logger.info("Starting feature engineering")
            self.engineered_data = self.original_data.copy()
            
            self._process_datetime()
            self._create_transaction_features()
            self._scale_features()
            self._encode_categoricals()
            self._finalize_features()
            
            self.logger.info("Feature engineering completed successfully")
            return self.engineered_data
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {str(e)}", exc_info=True)
            raise

    def _process_datetime(self):
        """Convert and extract datetime features"""
        # Convert to datetime with error handling
        self.engineered_data['purchase_time'] = pd.to_datetime(
            self.engineered_data['purchase_time'], errors='coerce'
        )
        self.engineered_data['signup_time'] = pd.to_datetime(
            self.engineered_data['signup_time'], errors='coerce'
        )

        # Extract time features
        self.engineered_data['purchase_hour'] = self.engineered_data['purchase_time'].dt.hour
        self.engineered_data['purchase_day'] = self.engineered_data['purchase_time'].dt.dayofweek
        self.engineered_data['signup_hour'] = self.engineered_data['signup_time'].dt.hour
        
        # Convert to timestamps
        self.engineered_data['purchase_timestamp'] = (
            self.engineered_data['purchase_time'].astype('int64') // 10**9
        )
        self.engineered_data['signup_timestamp'] = (
            self.engineered_data['signup_time'].astype('int64') // 10**9
        )

    def _create_transaction_features(self):
        """Create transaction behavior features"""
        # Transaction frequency
        self.engineered_data['transaction_frequency'] = (
            self.engineered_data.groupby('user_id')['user_id']
            .transform('count')
        )
        
        # Transaction velocity calculation
        first_transaction = (
            self.engineered_data.groupby('user_id')['purchase_time']
            .transform('min')
        )
        last_transaction = (
            self.engineered_data.groupby('user_id')['purchase_time']
            .transform('max')
        )
        
        time_diff = (last_transaction - first_transaction).dt.total_seconds()
        self.engineered_data['transaction_velocity'] = np.where(
            time_diff == 0,
            0,
            time_diff / self.engineered_data['transaction_frequency']
        )

    def _scale_features(self):
        """Apply feature scaling"""
        # Min-Max scaling
        minmax_cols = ['purchase_value', 'transaction_frequency', 'transaction_velocity']
        scaler = MinMaxScaler()
        self.engineered_data[minmax_cols] = scaler.fit_transform(
            self.engineered_data[minmax_cols]
        )
        
        # Standard scaling for age
        std_scaler = StandardScaler()
        self.engineered_data['age_scaled'] = std_scaler.fit_transform(
            self.engineered_data[['age']]
        )

    def _encode_categoricals(self):
        """Handle categorical encoding"""
        # One-hot encoding
        self.engineered_data = pd.get_dummies(
            self.engineered_data,
            columns=['source', 'browser', 'country'],
            drop_first=True,
            prefix=['src', 'br', 'ctry']
        )
        
        # Label encoding
        label_enc = LabelEncoder()
        self.engineered_data['sex_code'] = label_enc.fit_transform(
            self.engineered_data['sex']
        )

    def _finalize_features(self):
        """Final cleanup"""
        # Drop redundant columns
        self.engineered_data.drop(
            ['signup_time', 'purchase_time', 'sex'],
            axis=1,
            inplace=True
        )
        
        # Convert boolean columns
        bool_cols = self.engineered_data.select_dtypes(include='bool').columns
        self.engineered_data[bool_cols] = self.engineered_data[bool_cols].astype(int)

    def save_processed_data(self, output_path: str):
        """Save engineered features"""
        try:
            self.engineered_data.to_csv(output_path, index=False)
            self.logger.info(f"Data saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Save failed: {str(e)}")
            raise

class FeatureCredit:
    def __init__(self, data: pd.DataFrame, output_path: str):
        self.data = data.copy()
        self.output_path = output_path
        self.logger = setup_logger(self.__class__.__name__, log_dir=output_path)
        self.scaler = None

        required_columns = ["Time", "Amount"]
        missing = [col for col in required_columns if col not in self.data.columns]
        if missing:
            self.logger.error(f"Missing required columns: {missing}")
            raise ValueError(f"Missing required columns: {missing}")

    def scale_amount(self):
        """Scale the 'Amount' feature"""
        self.logger.info("Scaling 'Amount' feature...")
        try:
            self.scaler = StandardScaler()
            scaled_amount = self.scaler.fit_transform(
                self.data["Amount"].values.reshape(-1, 1)
            )
            self.data["scaled_amount"] = scaled_amount
            self.logger.info("'Amount' scaled successfully.")
        except Exception as e:
            self.logger.error(f"Error scaling 'Amount': {str(e)}")
            raise

    def create_time_features(self):
        """Create time-based features"""
        self.logger.info("Creating time-based features...")
        try:
            # Hour of day (0-23)
            self.data["time_hour"] = (self.data["Time"].astype(int) % 86400) // 3600
            # Day of week (0=Monday, 6=Sunday)
            self.data["day_of_week"] = (self.data["Time"].astype(int) // 86400) % 7
            # Weekend flag
            self.data["is_weekend"] = self.data["day_of_week"].apply(
                lambda x: 1 if x >= 5 else 0
            )
            self.logger.info("Time features created successfully.")
        except Exception as e:
            self.logger.error(f"Error creating time features: {str(e)}")
            raise

    def save_processed_data(self, filename: str = "featured_credit_data.csv"):
        """Save processed data"""
        try:
            output_file = os.path.join(self.output_path, filename)
            self.data.to_csv(output_file, index=False)
            self.logger.info(f"Data saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            raise

    def process(self):
        """Execute full pipeline"""
        try:
            self.logger.info("Starting feature engineering pipeline...")
            self.scale_amount()
            self.create_time_features()
            self.save_processed_data()
            self.logger.info("Pipeline completed successfully.")
            return self.data
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise