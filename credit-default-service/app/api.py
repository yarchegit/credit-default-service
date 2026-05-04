from flask import Flask, request, jsonify
import joblib
import os
import sys
import logging
from pythonjsonlogger import jsonlogger
import random
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import predict_default

app = Flask(__name__)

logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(message)s')
logHandler.setFormatter(formatter)
app.logger.addHandler(logHandler)
app.logger.setLevel(logging.INFO)

MODEL_V1_PATH = os.getenv('MODEL_V1_PATH', 'models/model_v1.pkl')
MODEL_V2_PATH = os.getenv('MODEL_V2_PATH', 'models/model_v2.pkl')
FEATURE_NAMES_PATH = 'models/feature_names.pkl'

try:
    model_v1 = joblib.load(MODEL_V1_PATH)
    model_v2 = joblib.load(MODEL_V2_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    app.logger.info("Models loaded")
except Exception as e:
    app.logger.error(f"Failed to load models: {e}")
    model_v1 = None
    model_v2 = None
    feature_names = None

AB_TEST_RATIO = float(os.getenv('AB_TEST_RATIO', '0.5'))

def generate_request_id():
    return f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{random.randint(1000, 9999)}"

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'service': 'Credit Default Prediction API',
        'version': '1.0.0',
        'author': 'yarchegit',
        'endpoints': {
            'health': 'GET /health',
            'predict': 'POST /predict',
            'predict_v1': 'POST /predict/v1',
            'predict_v2': 'POST /predict/v2'
        }
    }), 200

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy' if (model_v1 and model_v2) else 'unhealthy',
        'timestamp': datetime.now().isoformat(),
        'models': {
            'v1_loaded': model_v1 is not None,
            'v2_loaded': model_v2 is not None
        },
        'ab_test_ratio': AB_TEST_RATIO
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    request_id = generate_request_id()
    
    if not (model_v1 and model_v2 and feature_names):
        return jsonify({'error': 'Models not loaded', 'request_id': request_id}), 503
    
    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({'error': 'Missing features', 'request_id': request_id}), 400
        
        use_v2 = random.random() < AB_TEST_RATIO
        model = model_v2 if use_v2 else model_v1
        model_version = 'v2' if use_v2 else 'v1'
        
        result = predict_default(model, data['features'], feature_names)
        result['model_version'] = model_version
        result['request_id'] = request_id
        result['timestamp'] = datetime.now().isoformat()
        
        app.logger.info({
            'event': 'prediction',
            'request_id': request_id,
            'model_version': model_version,
            'prediction': result['prediction_label']
        })
        
        return jsonify(result), 200
        
    except Exception as e:
        app.logger.error(f"Error: {e}")
        return jsonify({'error': str(e), 'request_id': request_id}), 500

@app.route('/predict/v1', methods=['POST'])
def predict_v1():
    request_id = generate_request_id()
    if not (model_v1 and feature_names):
        return jsonify({'error': 'Model v1 not loaded'}), 503
    try:
        data = request.get_json()
        result = predict_default(model_v1, data.get('features', {}), feature_names)
        result['model_version'] = 'v1'
        result['request_id'] = request_id
        result['timestamp'] = datetime.now().isoformat()
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/v2', methods=['POST'])
def predict_v2():
    request_id = generate_request_id()
    if not (model_v2 and feature_names):
        return jsonify({'error': 'Model v2 not loaded'}), 503
    try:
        data = request.get_json()
        result = predict_default(model_v2, data.get('features', {}), feature_names)
        result['model_version'] = 'v2'
        result['request_id'] = request_id
        result['timestamp'] = datetime.now().isoformat()
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Credit Default Prediction API")
    print(f"A/B Test Ratio: {AB_TEST_RATIO}")
    app.run(host='0.0.0.0', port=5000, debug=False)
