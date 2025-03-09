import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import logging
from io import StringIO
from sklearn.preprocessing import StandardScaler
class FraudDataProcessor:
    def __init__(self, fraud_data, creditcard_data, ip_to_country):
        # Initialize datasets as copies to preserve original data
        self.original_fraud = fraud_data.copy()
        self.original_creditcard = creditcard_data.copy()
        self.original_ip = ip_to_country.copy()
        
        # Processed data containers
        self.cleaned_fraud = None
        self.cleaned_creditcard = None
        self.cleaned_ip = None
        
        # Setup directories and logging
        os.makedirs("logs", exist_ok=True)
        os.makedirs("plots", exist_ok=True)
        logging.basicConfig(
            filename="logs/data_processing.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filemode="w"
        )

    def clean_data_types(self):
        """Step 1: Perform initial data type conversions and cleaning"""
        logging.info("Starting data type cleaning...")
        
        # Process fraud data
        self.cleaned_fraud = self.original_fraud.copy()
        self.cleaned_fraud['signup_time'] = pd.to_datetime(self.cleaned_fraud['signup_time'])
        self.cleaned_fraud['purchase_time'] = pd.to_datetime(self.cleaned_fraud['purchase_time'])
        self.cleaned_fraud['ip_address'] = (
            self.cleaned_fraud['ip_address']
            .astype(str)
            .str.replace(r'\.0$', '', regex=True)
        )
        self.cleaned_fraud[['source', 'browser', 'sex']] = (
            self.cleaned_fraud[['source', 'browser', 'sex']]
            .astype('category')
        )
        self.cleaned_fraud['time_to_purchase'] = (
            self.cleaned_fraud['purchase_time'] - 
            self.cleaned_fraud['signup_time']
        ).dt.total_seconds()

        # Process credit card data
        self.cleaned_creditcard = self.original_creditcard.copy()
        if (self.cleaned_creditcard['Time'] < 0).any():
            logging.warning("Negative values found in Time column")
        if (self.cleaned_creditcard['Amount'] < 0).any():
            logging.warning("Negative values found in Amount column")
        scaler = StandardScaler()
        self.cleaned_creditcard['scaled_amount'] = scaler.fit_transform(
            self.cleaned_creditcard['Amount'].values.reshape(-1, 1)
        )

        # Process IP data
        self.cleaned_ip = self.original_ip.copy()
        self.cleaned_ip['lower_bound_ip_address'] = (
            self.cleaned_ip['lower_bound_ip_address']
            .astype('int64')
        )
        self.cleaned_ip['upper_bound_ip_address'] = (
            self.cleaned_ip['upper_bound_ip_address']
            .astype('int64')
        )
        self.cleaned_ip['country'] = self.cleaned_ip['country'].astype('category')

    def check_missing_values(self):
        """Step 2: Check for missing values"""
        logging.info("Checking missing values...")
        fraud_missing = self.cleaned_fraud.isnull().sum()
        credit_missing = self.cleaned_creditcard.isnull().sum()
        ip_missing = self.cleaned_ip.isnull().sum()

        print("\nMissing Values in Fraud Data:\n", fraud_missing)
        print("\nMissing Values in Credit Card Data:\n", credit_missing)
        print("\nMissing Values in IP Data:\n", ip_missing)
        
        return {
            'fraud': fraud_missing,
            'creditcard': credit_missing,
            'ip': ip_missing
        }

    def remove_duplicates(self):
        """Step 3: Remove duplicate rows"""
        logging.info("Checking for duplicates...")
        fraud_dups = self.cleaned_fraud.duplicated().sum()
        credit_dups = self.cleaned_creditcard.duplicated().sum()
        ip_dups = self.cleaned_ip.duplicated().sum()

        if fraud_dups > 0:
            self.cleaned_fraud = self.cleaned_fraud.drop_duplicates()
            logging.warning(f"Removed {fraud_dups} duplicates from fraud data")
        if credit_dups > 0:
            self.cleaned_creditcard = self.cleaned_creditcard.drop_duplicates()
            logging.warning(f"Removed {credit_dups} duplicates from credit data")
        if ip_dups > 0:
            self.cleaned_ip = self.cleaned_ip.drop_duplicates()
            logging.warning(f"Removed {ip_dups} duplicates from IP data")

    def validate_data_consistency(self):
        """Step 4: Perform logical consistency checks"""
        logging.info("Performing consistency validation...")
        
        # Check purchase time validity
        invalid_purchases = (
            self.cleaned_fraud['time_to_purchase'] < 0
        ).sum()
        if invalid_purchases > 0:
            logging.warning(f"{invalid_purchases} purchases occurred before signup")

        # Check IP range validity
        invalid_ranges = (
            self.cleaned_ip['lower_bound_ip_address'] > 
            self.cleaned_ip['upper_bound_ip_address']
        ).sum()
        if invalid_ranges > 0:
            logging.warning(f"{invalid_ranges} invalid IP ranges found")

    def generate_target_distribution_plot(self):
        """Step 5: Create combined target distribution plot"""
        logging.info("Generating target distribution visualization...")
        
        fig = make_subplots(rows=1, cols=2,
                          specs=[[{'type':'domain'}, {'type':'domain'}]],
                          subplot_titles=['Fraud Dataset', 'Credit Card Dataset'])

        # Fraud data subplot
        fraud_counts = self.cleaned_fraud['class'].value_counts()
        fig.add_trace(
            go.Pie(labels=fraud_counts.index,
                   values=fraud_counts.values,
                   name='Fraud Data',
                   marker=dict(colors=['#66b3ff', '#ff6666'])),
            row=1, col=1
        )

        # Credit card data subplot
        credit_counts = self.cleaned_creditcard['Class'].value_counts()
        fig.add_trace(
            go.Pie(labels=credit_counts.index,
                   values=credit_counts.values,
                   name='Credit Card Data',
                   marker=dict(colors=['#66b3ff', '#ff6666'])),
            row=1, col=2
        )

        fig.update_layout(
            title_text='Fraud vs Credit Card Target Distribution',
            height=600,
            width=1000,
            showlegend=False
        )
        
        fig.write_html("plots/target_distribution.html")
        fig.show()

    # Display methods to show current data state
    def show_data_shapes(self):
        print("\nFraud Data Shape:", self.cleaned_fraud.shape)
        print("Credit Card Data Shape:", self.cleaned_creditcard.shape)
        print("IP Data Shape:", self.cleaned_ip.shape)

    def show_data_info(self):
        print("\nFraud Data Info:")
        print(self.cleaned_fraud.info())
        print("\nCredit Card Data Info:")
        print(self.cleaned_creditcard.info())
        print("\nIP Data Info:")
        print(self.cleaned_ip.info())

    def save_all_data(
        self,
        fraud_path: str = "cleaned_fraud.csv",
        credit_path: str = "cleaned_credit.csv",
        ip_path: str = "cleaned_ip.csv"
    ):
        """Save all cleaned datasets to CSV files"""
        try:
            # Save fraud data
            self.cleaned_fraud.to_csv(fraud_path, index=False)
            logging.info(f"Fraud data saved to {fraud_path}")

            # Save credit card data
            self.cleaned_creditcard.to_csv(credit_path, index=False)
            logging.info(f"Credit card data saved to {credit_path}")

            # Save IP data
            self.cleaned_ip.to_csv(ip_path, index=False)
            logging.info(f"IP data saved to {ip_path}")

            print(f"""
            Data saved successfully:
            - Fraud: {fraud_path}
            - Credit: {credit_path}
            - IP: {ip_path}
            """)
        except Exception as e:
            logging.error(f"Error saving data: {e}")
            raise
    # Properties to access cleaned data
    @property
    def fraud_data(self):
        return self.cleaned_fraud

    @property
    def creditcard_data(self):
        return self.cleaned_creditcard

    @property
    def ip_data(self):
        return self.cleaned_ip