import pandas as pd
import logging
import os
from sklearn.preprocessing import StandardScaler
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class FraudFeatureEngineer:
    def __init__(self, fraud_data: pd.DataFrame):
        """
        Initialize the feature engineer with fraud data.
        :param fraud_data: DataFrame containing the fraud dataset.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        self.logger.info("Initializing FraudFeatureEngineer with data shape: %s", fraud_data.shape)
        
        # Validate required columns
        required_columns = ['user_id', 'purchase_time', 'purchase_value', 'age', 'source', 'browser', 'sex']
        missing_columns = [col for col in required_columns if col not in fraud_data.columns]
        if missing_columns:
            self.logger.error("Missing required columns: %s", missing_columns)
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        self.fraud_data = fraud_data.copy()
        self.scaler = None
        self.encoder = None
        
        if self.fraud_data.isnull().any().any():
            self.logger.warning("Input data contains missing values. Proceed with caution.")

    def add_transaction_frequency_velocity(self):
        """
        Add transaction frequency and velocity features.
        """
        self.logger.info("Adding transaction frequency and velocity features...")
        try:
            # Ensure 'purchase_time' is datetime
            self.fraud_data['purchase_time'] = pd.to_datetime(self.fraud_data['purchase_time'])
            
            # Sort by user and purchase time
            self.fraud_data.sort_values(by=['user_id', 'purchase_time'], inplace=True)
            
            # Transaction Frequency
            self.fraud_data['transaction_frequency'] = self.fraud_data.groupby('user_id')['user_id'].transform('count')
            
            # Transaction Velocity (seconds between transactions)
            self.fraud_data['transaction_velocity'] = self.fraud_data.groupby('user_id')['purchase_time'].diff().dt.total_seconds()
            self.fraud_data['transaction_velocity'] = self.fraud_data['transaction_velocity'].fillna(0)
            
            self.logger.info("Successfully added transaction frequency and velocity features.")
        except Exception as e:
            self.logger.error("Error adding transaction features: %s", str(e))
            raise

    def add_time_based_features(self):
        """
        Add time-based features: hour_of_day and day_of_week.
        """
        self.logger.info("Adding time-based features...")
        try:
            self.fraud_data['hour_of_day'] = self.fraud_data['purchase_time'].dt.hour
            self.fraud_data['day_of_week'] = self.fraud_data['purchase_time'].dt.dayofweek  # Monday=0, Sunday=6
            self.logger.info("Successfully added time-based features.")
        except Exception as e:
            self.logger.error("Error adding time-based features: %s", str(e))
            raise

    def normalize_features(self):
        """
        Normalize numerical features using StandardScaler.
        """
        numerical_features = ['purchase_value', 'age', 'transaction_frequency', 'transaction_velocity']
        self.logger.info("Normalizing numerical features: %s", numerical_features)
        try:
            if self.scaler is None:
                self.scaler = StandardScaler()
                self.fraud_data[numerical_features] = self.scaler.fit_transform(self.fraud_data[numerical_features])
                self.logger.info("Fitted StandardScaler and normalized features.")
            else:
                self.fraud_data[numerical_features] = self.scaler.transform(self.fraud_data[numerical_features])
                self.logger.info("Normalized features using existing StandardScaler.")
        except Exception as e:
            self.logger.error("Error normalizing features: %s", str(e))
            raise

    def encode_categorical_features(self):
        """
        Encode categorical features using OneHotEncoder.
        """
        categorical_features = ['source', 'browser', 'sex']
        self.logger.info("Encoding categorical features: %s", categorical_features)
        try:
            if self.encoder is None:
                self.encoder = OneHotEncoder(drop='first', sparse_output=False)
                encoded_features = self.encoder.fit_transform(self.fraud_data[categorical_features])
                self.logger.info("Fitted OneHotEncoder with categories: %s", self.encoder.categories_)
            else:
                encoded_features = self.encoder.transform(self.fraud_data[categorical_features])
                self.logger.info("Encoded features using existing OneHotEncoder.")
            
            # Create DataFrame and merge
            encoded_df = pd.DataFrame(
                encoded_features,
                columns=self.encoder.get_feature_names_out(categorical_features),
                index=self.fraud_data.index
            )
            self.fraud_data.drop(categorical_features, axis=1, inplace=True)
            self.fraud_data = pd.concat([self.fraud_data, encoded_df], axis=1)
            self.logger.info("Successfully encoded %d categorical features.", len(categorical_features))
        except Exception as e:
            self.logger.error("Error encoding categorical features: %s", str(e))
            raise

    def process(self):
        """
        Execute the full feature engineering pipeline.
        :return: Processed DataFrame.
        """
        self.logger.info("Starting full feature engineering pipeline...")
        try:
            self.add_transaction_frequency_velocity()
            self.add_time_based_features()
            self.normalize_features()
            self.encode_categorical_features()
            self.logger.info("Feature engineering completed. Final data shape: %s", self.fraud_data.shape)
            return self.fraud_data
        except Exception as e:
            self.logger.error("Feature engineering pipeline failed: %s", str(e))
            raise
class FeatureCredit:
    def __init__(self, data: pd.DataFrame, output_path: str):
        self.data = data.copy()
        self.output_path = output_path  # Initialize output_path first
        self.logger = self._setup_logger()
        self.scaler = None

        # Validate required columns
        required_columns = ["Time", "Amount"]
        missing = [col for col in required_columns if col not in self.data.columns]
        if missing:
            self.logger.error(f"Missing required columns: {missing}")
            raise ValueError(f"Missing required columns: {missing}")

    def _setup_logger(self):
        """Configure logging for the class."""
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            # Stream handler for console output
            stream_handler = logging.StreamHandler()
            # File handler for saving logs to a file
            file_handler = logging.FileHandler(os.path.join(self.output_path, "feature_engineering.log"))
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            stream_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)
            logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
        return logger

    def scale_amount(self):
        """Scale the 'Amount' feature using StandardScaler."""
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
        """Extract time-based features from the 'Time' column."""
        self.logger.info("Creating time-based features...")
        try:
            # Convert seconds since reference to hour of the day (0-23)
            self.data["time_hour"] = (self.data["Time"].astype(int) % 86400) // 3600

            # Extract day of the week (0=Monday, 6=Sunday)
            self.data["day_of_week"] = (self.data["Time"].astype(int) // 86400) % 7

            # Extract whether it's a weekend (1=Weekend, 0=Weekday)
            self.data["is_weekend"] = self.data["day_of_week"].apply(
                lambda x: 1 if x >= 5 else 0
            )

            self.logger.info("Time-based features created successfully.")
        except Exception as e:
            self.logger.error(f"Error creating time features: {str(e)}")
            raise

    def save_processed_data(self, filename: str ="featured_credit_data.csv"):
        """Save the processed DataFrame to CSV."""
        try:
            output_file = os.path.join(self.output_path, filename)
            self.data.to_csv(output_file, index=False)
            self.logger.info(f"Data saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            raise

    def process(self):
        """Execute the full pipeline."""
        self.logger.info("Starting feature engineering pipeline...")
        try:
            self.scale_amount()
            self.create_time_features()
            self.save_processed_data()
            self.logger.info("Pipeline completed successfully.")
            return self.data
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise