import logging
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from pathlib import Path
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Paths to models and data using pathlib
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'data'
CREDIT_MODEL_PATH = MODEL_DIR / 'best_random_forest_credit.pkl'
FRAUD_MODEL_PATH = MODEL_DIR / 'best_gradient_boosting_fraud.pkl'
FEATURED_CREDIT_DATA_PATH = DATA_DIR / 'featured_credit_data.csv'
FEATURED_FRAUD_DATA_PATH = DATA_DIR / 'featured_fraud_data.csv'

# Load models
try:
    credit_model = joblib.load(CREDIT_MODEL_PATH)
    logger.info("Credit model loaded successfully")
except Exception as e:
    logger.error(f"Error loading credit model: {str(e)}")
    credit_model = None

try:
    fraud_model = joblib.load(FRAUD_MODEL_PATH)
    logger.info("Fraud model loaded successfully")
except Exception as e:
    logger.error(f"Error loading fraud model: {str(e)}")
    fraud_model = None

# Load feature data to get expected features
try:
    credit_df = pd.read_csv(FEATURED_CREDIT_DATA_PATH)
    credit_features = credit_df.drop(columns=['Class']).columns.tolist()
    logger.info(f"Credit features loaded: {credit_features}")
except Exception as e:
    logger.error(f"Error loading credit features: {str(e)}")
    credit_features = []

try:
    fraud_df = pd.read_csv(FEATURED_FRAUD_DATA_PATH)
    # Adjust columns to drop based on actual dataset
    columns_to_drop = ['class']
    # Remove columns that shouldn't be in features if they exist
    for col in ['user_id', 'device_id', 'ip_address', 'purchase_timestamp', 'signup_timestamp']:
        if col in fraud_df.columns:
            columns_to_drop.append(col)
    fraud_features = fraud_df.drop(columns=columns_to_drop).columns.tolist()
    logger.info(f"Fraud features loaded: {fraud_features}")
except Exception as e:
    logger.error(f"Error loading fraud features: {str(e)}")
    fraud_features = []

@app.route('/', methods=['GET'])
def get_features():
    return jsonify({
        'credit_features': credit_features,
        'fraud_features': fraud_features
    })

@app.route('/predict_credit', methods=['POST'])
def predict_credit():
    if credit_model is None:
        logger.error("Credit model not loaded")
        return jsonify({'error': 'Credit model not loaded'}), 500

    data = request.get_json()
    if not data:
        logger.error("No input data provided")
        return jsonify({'error': 'No input data provided'}), 400

    try:
        # Check for missing features
        missing = [f for f in credit_features if f not in data]
        if missing:
            logger.error(f"Missing features in credit prediction input: {missing}")
            return jsonify({'error': f'Missing features: {missing}'}), 400

        # Create DataFrame from input
        input_df = pd.DataFrame([data], columns=credit_features)
        prediction = credit_model.predict(input_df)[0]
        logger.info(f"Credit prediction made: {prediction}")
        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        logger.error(f"Error in credit prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_fraud', methods=['POST'])
def predict_fraud():
    if fraud_model is None:
        logger.error("Fraud model not loaded")
        return jsonify({'error': 'Fraud model not loaded'}), 500

    data = request.get_json()
    if not data:
        logger.error("No input data provided")
        return jsonify({'error': 'No input data provided'}), 400

    try:
        # Check for missing features
        missing = [f for f in fraud_features if f not in data]
        if missing:
            logger.error(f"Missing features in fraud prediction input: {missing}")
            return jsonify({'error': f'Missing features: {missing}'}), 400

        # Create DataFrame from input
        input_df = pd.DataFrame([data], columns=fraud_features)
        prediction = fraud_model.predict(input_df)[0]
        logger.info(f"Fraud prediction made: {prediction}")
        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        logger.error(f"Error in fraud prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask API server")
    app.run(host='0.0.0.0', port=5000, debug=False)