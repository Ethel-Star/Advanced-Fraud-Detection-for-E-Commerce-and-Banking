import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/eda.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w"  # Overwrites the log file each run; use "a" to append
    )
class AdvancedEDA:
    def __init__(self, fraud_data, creditcard_data, ip_to_country, output_path):
        """
        Initialize the pipeline with datasets and configuration.
        """
        self.fraud_data = fraud_data
        self.creditcard_data = creditcard_data
        self.ip_to_country = ip_to_country
        self.output_path = output_path
        self.logger = self._setup_logging()

    def _setup_logging(self):
        """
        Configure logging for the pipeline.
        """
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(
            filename="logs/data_analysis_EDA.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filemode="w"  # Overwrites the log file each run; use "a" to append
        )
        return logging.getLogger()

    def plot_fraud_data_distributions(self):
        """
        Plot distributions for 'purchase_value', 'age', 'source', and 'browser' in fraud_data.
        """
        print("\nPlotting distributions for fraud_data...")
        self.logger.info("Plotting distributions for fraud_data.")
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            # Purchase Value Histogram
            sns.histplot(self.fraud_data['purchase_value'], bins=50, kde=True, color='skyblue', ax=axes[0, 0])
            axes[0, 0].set_title('Distribution of Purchase Value')
            axes[0, 0].set_xlabel('Purchase Value ($)')
            axes[0, 0].set_ylabel('Frequency')

            # Age Histogram
            sns.histplot(self.fraud_data['age'], bins=50, kde=True, color='lightgreen', ax=axes[0, 1])
            axes[0, 1].set_title('Distribution of Age')
            axes[0, 1].set_xlabel('Age')
            axes[0, 1].set_ylabel('Frequency')

            # Source Bar Chart
            sns.countplot(x='source', data=self.fraud_data, ax=axes[1, 0])
            axes[1, 0].set_title('Distribution of Source')
            axes[1, 0].set_xlabel('Source')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].tick_params(axis='x', rotation=45)

            # Browser Bar Chart
            sns.countplot(x='browser', data=self.fraud_data, ax=axes[1, 1])
            axes[1, 1].set_title('Distribution of Browser')
            axes[1, 1].set_xlabel('Browser')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.show()
            print("Successfully plotted fraud_data distributions.")
            self.logger.info("Successfully plotted fraud_data distributions.")
        except Exception as e:
            print(f"Error plotting fraud_data distributions: {e}")
            self.logger.error(f"Error plotting fraud_data distributions: {e}")

    def plot_creditcard_data_distributions(self):
        """
        Plot distributions for 'Time', 'Amount', and 'Class' in creditcard_data.
        """
        print("\nPlotting distributions for creditcard_data...")
        self.logger.info("Plotting distributions for creditcard_data.")
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # Time Histogram
            sns.histplot(self.creditcard_data['Time'], bins=50, kde=True, color='lightblue', ax=axes[0])
            axes[0].set_title('Distribution of Time', fontsize=14)
            axes[0].set_xlabel('Time (seconds)', fontsize=12)
            axes[0].set_ylabel('Frequency', fontsize=12)

            # Amount Histogram
            sns.histplot(self.creditcard_data['Amount'], bins=50, kde=True, color='lightgreen', ax=axes[1])
            axes[1].set_title('Distribution of Transaction Amount', fontsize=14)
            axes[1].set_xlabel('Amount ($)', fontsize=12)
            axes[1].set_ylabel('Frequency', fontsize=12)

            # Class Bar Chart
            sns.countplot(x='Class', data=self.creditcard_data, ax=axes[2])
            axes[2].set_title('Fraudulent vs Non-Fraudulent Transactions', fontsize=14)
            axes[2].set_xlabel('Class', fontsize=12)
            axes[2].set_ylabel('Count', fontsize=12)
            axes[2].set_xticks([0, 1])
            axes[2].set_xticklabels(['Non-Fraudulent', 'Fraudulent'], fontsize=12)

            # Add annotations
            for p in axes[2].patches:
                axes[2].annotate(f'{int(p.get_height())}', 
                                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                                 ha='center', va='center', 
                                 xytext=(0, 10), 
                                 textcoords='offset points',
                                 fontsize=12)

            plt.tight_layout()
            plt.show()
            print("Successfully plotted creditcard_data distributions.")
            self.logger.info("Successfully plotted creditcard_data distributions.")
        except Exception as e:
            print(f"Error plotting creditcard_data distributions: {e}")
            self.logger.error(f"Error plotting creditcard_data distributions: {e}")

    def plot_fraud_data_relationships(self):
        """
        Plot relationships between features in fraud_data.
        """
        print("\nPlotting relationships in fraud_data...")
        self.logger.info("Plotting relationships in fraud_data.")
        try:
            sns.set_style("whitegrid")
            fig, axes = plt.subplots(3, 2, figsize=(14, 12))

            # Purchase Value vs Age
            sns.scatterplot(x='age', y='purchase_value', hue='class', data=self.fraud_data, alpha=0.5, palette='coolwarm', ax=axes[0, 0])
            axes[0, 0].set_title('Purchase Value vs Age (Fraud & Non-Fraud)')
            axes[0, 0].set_xlabel('Age')
            axes[0, 0].set_ylabel('Purchase Value ($)')

            # Purchase Value vs Source
            sns.boxplot(x='source', y='purchase_value', hue='class', data=self.fraud_data, palette='viridis', ax=axes[0, 1])
            axes[0, 1].set_title('Purchase Value vs Source (Fraud & Non-Fraud)')
            axes[0, 1].set_xlabel('Source')
            axes[0, 1].set_ylabel('Purchase Value ($)')
            axes[0, 1].tick_params(axis='x', rotation=45)

            # Age vs Source
            sns.boxplot(x='source', y='age', hue='class', data=self.fraud_data, palette='coolwarm', ax=axes[1, 0])
            axes[1, 0].set_title('Age vs Source (Fraud & Non-Fraud)')
            axes[1, 0].set_xlabel('Source')
            axes[1, 0].set_ylabel('Age')
            axes[1, 0].tick_params(axis='x', rotation=45)

            # Age vs Browser
            sns.boxplot(x='browser', y='age', hue='class', data=self.fraud_data, palette='magma', ax=axes[1, 1])
            axes[1, 1].set_title('Age vs Browser (Fraud & Non-Fraud)')
            axes[1, 1].set_xlabel('Browser')
            axes[1, 1].set_ylabel('Age')
            axes[1, 1].tick_params(axis='x', rotation=90)

            # Fraud Distribution across Age
            sns.histplot(self.fraud_data, x="age", hue="class", multiple="stack", palette="coolwarm", kde=True, ax=axes[2, 0])
            axes[2, 0].set_title('Fraud Distribution across Age')
            axes[2, 0].set_xlabel('Age')
            axes[2, 0].set_ylabel('Count')

            # Purchase Time vs Class
            self.fraud_data['purchase_hour'] = pd.to_datetime(self.fraud_data['purchase_time']).dt.hour
            sns.histplot(self.fraud_data, x="purchase_hour", hue="class", multiple="stack", palette="viridis", kde=True, ax=axes[2, 1])
            axes[2, 1].set_title('Fraud vs Purchase Time')
            axes[2, 1].set_xlabel('Purchase Hour')
            axes[2, 1].set_ylabel('Count')

            plt.tight_layout()
            plt.show()
            print("Successfully plotted fraud_data relationships.")
            self.logger.info("Successfully plotted fraud_data relationships.")
        except Exception as e:
            print(f"Error plotting fraud_data relationships: {e}")
            self.logger.error(f"Error plotting fraud_data relationships: {e}")

    def plot_correlation_analysis(self):
        """
        Perform correlation analysis and plot heatmaps for fraud_data and creditcard_data.
        """
        print("\nPerforming correlation analysis...")
        self.logger.info("Performing correlation analysis.")
        try:
            numeric_fraud_data = self.fraud_data.select_dtypes(include=['number'])
            numeric_credit_data = self.creditcard_data.select_dtypes(include=['number'])

            # Fraud Data Heatmap and Scatter Plot
            fig1, axs1 = plt.subplots(1, 2, figsize=(16, 6))
            corr_fraud = numeric_fraud_data.corr()
            sns.heatmap(corr_fraud, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=axs1[0])
            axs1[0].set_title("Fraud Data Correlation Heatmap")

            sns.scatterplot(x=numeric_fraud_data['purchase_value'], y=numeric_fraud_data['age'], hue=numeric_fraud_data['class'], palette="viridis", alpha=0.7, ax=axs1[1])
            axs1[1].set_title("Fraud Data: Purchase Value vs Age (Color by Class)")
            plt.tight_layout()
            plt.show()

            # Credit Card Data Heatmap
            fig2, ax2 = plt.subplots(figsize=(14, 8))
            corr_credit = numeric_credit_data.corr()
            sns.heatmap(corr_credit, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax2)
            ax2.set_title("Credit Card Data Correlation Heatmap")
            plt.tight_layout()
            plt.show()

            print("Successfully performed correlation analysis.")
            self.logger.info("Successfully performed correlation analysis.")
        except Exception as e:
            print(f"Error performing correlation analysis: {e}")
            self.logger.error(f"Error performing correlation analysis: {e}")

    def map_ip_to_country(self):
        """
        Map IP addresses in fraud_data to countries using ip_to_country.
        """
        print("\nMapping IP addresses to countries...")
        self.logger.info("Mapping IP addresses to countries.")
        try:
            self.fraud_data['ip_address'] = self.fraud_data['ip_address'].astype(int)

            def find_country_by_ip(ip):
                matched_row = self.ip_to_country[(ip >= self.ip_to_country['lower_bound_ip_address']) & 
                                                 (ip <= self.ip_to_country['upper_bound_ip_address'])]
                return matched_row['country'].values[0] if not matched_row.empty else 'Unknown'

            self.fraud_data['country'] = self.fraud_data['ip_address'].apply(find_country_by_ip)
            print("Successfully mapped IP addresses to countries.")
            self.logger.info("Successfully mapped IP addresses to countries.")
        except Exception as e:
            print(f"Error mapping IP addresses to countries: {e}")
            self.logger.error(f"Error mapping IP addresses to countries: {e}")

    def save_data(self):
        """
        Save the processed fraud_data to a CSV file.
        """
        print(f"\nSaving processed data to {self.output_path}...")
        self.logger.info(f"Saving processed data to {self.output_path}.")
        try:
            self.fraud_data.to_csv(self.output_path, index=False)
            print(f"Data successfully saved to {self.output_path}.")
            self.logger.info(f"Data successfully saved to {self.output_path}.")
        except Exception as e:
            print(f"Error saving data to {self.output_path}: {e}")
            self.logger.error(f"Error saving data to {self.output_path}: {e}")

    def run_pipeline(self):
        """
        Execute the entire EDA pipeline.
        """
        print("\nStarting exploratory data analysis (EDA)...")
        self.logger.info("Starting exploratory data analysis (EDA).")
        self.plot_fraud_data_distributions()
        self.plot_creditcard_data_distributions()
        self.plot_fraud_data_relationships()
        self.plot_correlation_analysis()
        self.map_ip_to_country()
        self.save_data()
        print("\nExploratory data analysis (EDA) completed.")
        self.logger.info("Exploratory data analysis (EDA) completed.")
