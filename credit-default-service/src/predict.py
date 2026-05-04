import joblib
import pandas as pd

def load_model(model_path):
    """Загрузка модели из файла"""
    return joblib.load(model_path)

def predict_default(model, features_dict, feature_names):
    """
    Прогнозирование дефолта по кредитной карте
    
    Args:
        model: обученная модель scikit-learn
        features_dict: словарь с признаками клиента
        feature_names: список имен признаков в правильном порядке
    
    Returns:
        dict: результат прогноза с вероятностями
    """
    # Преобразуем словарь в DataFrame с правильным порядком колонок
    features_df = pd.DataFrame([features_dict], columns=feature_names)
    
    # Получаем прогноз и вероятности
    prediction = model.predict(features_df)[0]
    probability = model.predict_proba(features_df)[0]
    
    return {
        'prediction': int(prediction),
        'prediction_label': 'DEFAULT' if prediction == 1 else 'NO_DEFAULT',
        'probability_no_default': float(probability[0]),
        'probability_default': float(probability[1]),
        'risk_level': _get_risk_level(probability[1])
    }

def _get_risk_level(prob_default):
    """Определение уровня риска на основе вероятности дефолта"""
    if prob_default > 0.7:
        return 'HIGH'
    elif prob_default > 0.4:
        return 'MEDIUM'
    else:
        return 'LOW'
