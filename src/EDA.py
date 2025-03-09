import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

class FraudEDA:
    def __init__(
        self,
        fraud_path: str,
        creditcard_path: str,
        ip_path: str,
        output_path: Optional[str] = "processed_fraud_data.csv",
    ):
        """
        Initialize EDA processor with file paths
        :param fraud_path: Path to fraud dataset
        :param creditcard_path: Path to credit card dataset
        :param ip_path: Path to IP mapping dataset
        :param output_path: Output path for processed data
        """
        self.fraud_data = pd.read_csv(fraud_path)
        self.creditcard_data = pd.read_csv(creditcard_path)
        self.ip_to_country = pd.read_csv(ip_path)
        self.output_path = output_path

        # Setup directories and logging
        os.makedirs("logs", exist_ok=True)
        os.makedirs("plots", exist_ok=True)
        logging.basicConfig(
            filename="logs/eda.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filemode="w",
        )

    @property
    def fraud_data(self) -> pd.DataFrame:
        return self._fraud_data

    @fraud_data.setter
    def fraud_data(self, value: pd.DataFrame):
        self._validate_data(value, required_cols=["ip_address", "class"])
        self._fraud_data = value

    @property
    def creditcard_data(self) -> pd.DataFrame:
        return self._creditcard_data

    @creditcard_data.setter
    def creditcard_data(self, value: pd.DataFrame):
        self._validate_data(value, required_cols=["Class", "Amount"])
        self._creditcard_data = value

    def _validate_data(self, data: pd.DataFrame, required_cols: list):
        """Validate dataset integrity"""
        missing_cols = set(required_cols) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def plot_fraud_distributions(self):
        """Univariate analysis for fraud dataset"""
        logging.info("Starting fraud data distribution analysis")
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            # Purchase Value Distribution
            sns.histplot(
                data=self.fraud_data,
                x="purchase_value",
                bins=50,
                kde=True,
                color="skyblue",
                ax=axes[0, 0],
            ).set(title="Purchase Value Distribution", xlabel="Value ($)")

            # Age Distribution
            sns.histplot(
                data=self.fraud_data,
                x="age",
                bins=50,
                kde=True,
                color="lightgreen",
                ax=axes[0, 1],
            ).set(title="Age Distribution", xlabel="Age")

            # Source Distribution
            sns.countplot(
                data=self.fraud_data,
                x="source",
                ax=axes[1, 0],
            ).set(title="Source Distribution", xlabel="Source", ylabel="Count")
            axes[1, 0].tick_params(axis="x", rotation=45)

            # Browser Distribution
            sns.countplot(
                data=self.fraud_data,
                x="browser",
                ax=axes[1, 1],
            ).set(title="Browser Distribution", xlabel="Browser", ylabel="Count")
            axes[1, 1].tick_params(axis="x", rotation=45)

            plt.tight_layout()
            plt.savefig("plots/fraud_distributions.png")
            plt.show()
            logging.info("Completed fraud data distribution analysis")

        except Exception as e:
            logging.error(f"Distribution analysis failed: {str(e)}")
            raise

    def plot_creditcard_distributions(self):
        """Univariate analysis for credit card dataset"""
        logging.info("Starting credit card data distribution analysis")
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # Time Distribution
            sns.histplot(
                data=self.creditcard_data,
                x="Time",
                bins=50,
                kde=True,
                color="lightblue",
                ax=axes[0],
            ).set(title="Transaction Time Distribution", xlabel="Time (s)")

            # Amount Distribution
            sns.histplot(
                data=self.creditcard_data,
                x="Amount",
                bins=50,
                kde=True,
                color="lightgreen",
                ax=axes[1],
            ).set(title="Transaction Amount Distribution", xlabel="Amount ($)")

            # Class Distribution
            sns.countplot(
                data=self.creditcard_data,
                x="Class",
                ax=axes[2],
            ).set(title="Fraud Distribution", xlabel="Class", ylabel="Count")
            axes[2].set_xticks([0, 1])  # Fix for set_ticklabels warning
            axes[2].set_xticklabels(["Non-Fraud", "Fraud"])

            plt.tight_layout()
            plt.savefig("plots/creditcard_distributions.png")
            plt.show()
            logging.info("Completed credit card data distribution analysis")

        except Exception as e:
            logging.error(f"Distribution analysis failed: {str(e)}")
            raise

    def plot_fraud_relationships(self):
        """Bivariate analysis for fraud dataset"""
        logging.info("Starting fraud data relationship analysis")
        try:
            fig, axes = plt.subplots(3, 2, figsize=(14, 12))

            # Purchase Value vs Age
            sns.scatterplot(
                data=self.fraud_data,
                x="age",
                y="purchase_value",
                hue="class",
                palette="coolwarm",
                ax=axes[0, 0],
            ).set(title="Purchase Value vs Age")

            # Purchase Value vs Source
            sns.boxplot(
                data=self.fraud_data,
                x="source",
                y="purchase_value",
                hue="class",
                palette="viridis",
                ax=axes[0, 1],
            ).set(title="Purchase Value by Source")
            axes[0, 1].tick_params(axis="x", rotation=45)

            # Age vs Browser
            sns.boxplot(
                data=self.fraud_data,
                x="browser",
                y="age",
                hue="class",
                palette="magma",
                ax=axes[1, 0],
            ).set(title="Age Distribution by Browser")
            axes[1, 0].tick_params(axis="x", rotation=90)

            # Fraud by Time
            self.fraud_data["purchase_hour"] = pd.to_datetime(
                self.fraud_data["purchase_time"]
            ).dt.hour
            sns.histplot(
                data=self.fraud_data,
                x="purchase_hour",
                hue="class",
                multiple="stack",
                palette="viridis",
                kde=True,
                ax=axes[1, 1],
            ).set(title="Fraud Distribution by Hour")

            # Turn off empty subplots
            axes[2, 0].axis("off")
            axes[2, 1].axis("off")

            plt.tight_layout()
            plt.savefig("plots/fraud_relationships.png")
            plt.show()
            logging.info("Completed fraud data relationship analysis")

        except Exception as e:
            logging.error(f"Relationship analysis failed: {str(e)}")
            raise

    def plot_correlations(self):
        """Correlation analysis for both datasets"""
        logging.info("Starting correlation analysis")
        try:
            # Fraud Data Correlation
            fraud_corr = self.fraud_data.select_dtypes(include="number").corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                fraud_corr,
                annot=True,
                cmap="coolwarm",
                fmt=".2f",
                linewidths=0.5,
            ).set(title="Fraud Data Correlation Matrix")
            plt.savefig("plots/fraud_correlation.png")
            plt.show()

            # Credit Card Correlation
            credit_corr = self.creditcard_data.corr()
            plt.figure(figsize=(14, 12))
            sns.heatmap(
                credit_corr,
                annot=True,
                cmap="coolwarm",
                fmt=".2f",
                linewidths=0.5,
                annot_kws={"size": 10},
            ).set(title="Credit Card Data Correlation Matrix")
            plt.savefig("plots/creditcard_correlation.png")
            plt.show()
            logging.info("Completed correlation analysis")

        except Exception as e:
            logging.error(f"Correlation analysis failed: {str(e)}")
            raise

    def map_ip_addresses(self):
        """Map IP addresses to countries"""
        logging.info("Starting IP to country mapping")
        try:
            # Convert IP to integer (handle missing/invalid values)
            self.fraud_data["ip_address"] = self.fraud_data["ip_address"].apply(
                lambda x: int("".join([f"{int(num):03}" for num in str(x).split(".")]))
                if pd.notna(x) and isinstance(x, str)
                else None
            )

            # Create IP range intervals
            self.ip_to_country["ip_range"] = pd.IntervalIndex.from_arrays(
                self.ip_to_country["lower_bound_ip_address"],
                self.ip_to_country["upper_bound_ip_address"],
            )

            # Map countries
            self.fraud_data["country"] = self.fraud_data["ip_address"].apply(
                lambda ip: self.ip_to_country[
                    self.ip_to_country["ip_range"].contains(ip)
                ]["country"].values[0]
                if ip and any(self.ip_to_country["ip_range"].contains(ip))
                else "Unknown"
            )
            logging.info("Completed IP to country mapping")

        except Exception as e:
            logging.error(f"IP mapping failed: {str(e)}")
            raise

    def save_processed_data(self):
        """Save processed fraud data with country mappings"""
        try:
            self.fraud_data.to_csv(self.output_path, index=False)
            logging.info(f"Data saved successfully to {self.output_path}")
        except Exception as e:
            logging.error(f"Save failed: {str(e)}")
            raise

    def perform_full_eda(self):
        """Execute complete EDA pipeline"""
        logging.info("Starting full EDA pipeline")
        try:
            self.plot_fraud_distributions()
            self.plot_creditcard_distributions()
            self.plot_fraud_relationships()
            self.plot_correlations()
            self.map_ip_addresses()
            self.save_processed_data()
            logging.info("EDA pipeline completed successfully")
        except Exception as e:
            logging.error(f"EDA pipeline failed: {str(e)}")
            raise