import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import joblib
import os
import sys

def load_and_prepare_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    df = pd.read_csv(filepath)
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)
    
    X = df.drop('default.payment.next.month', axis=1)
    y = df['default.payment.next.month']
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_model_v1(X_train, y_train):
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_model_v2(X_train, y_train):
    model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6,
        min_samples_split=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, version):
    y_pred = model.predict(X_test)
    
    print(f"\n{'='*50}")
    print(f"Model {version} Performance")
    print('='*50)
    print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return {
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred)
    }

if __name__ == "__main__":
    print("Credit Default Model Training Pipeline\n")
    
    try:
        X_train, X_test, y_train, y_test = load_and_prepare_data('data/raw/UCI_Credit_Card.csv')
        print(f"   Training: {X_train.shape}")
        print(f"   Test: {X_test.shape}")
        
        print("\nTraining Model v1...")
        model_v1 = train_model_v1(X_train, y_train)
        metrics_v1 = evaluate_model(model_v1, X_test, y_test, "v1")
        
        print("\nTraining Model v2...")
        model_v2 = train_model_v2(X_train, y_train)
        metrics_v2 = evaluate_model(model_v2, X_test, y_test, "v2")
        
        print("\nSaving models...")
        os.makedirs('models', exist_ok=True)
        
        joblib.dump(model_v1, 'models/model_v1.pkl')
        joblib.dump(model_v2, 'models/model_v2.pkl')
        joblib.dump(X_train.columns.tolist(), 'models/feature_names.pkl')
        
        print("models/model_v1.pkl")
        print("models/model_v2.pkl")
        print("models/feature_names.pkl")
        
        print("A/B Test Comparison")
        print(f"v1 F1: {metrics_v1['f1']:.4f}")
        print(f"v2 F1: {metrics_v2['f1']:.4f}")
        improvement = ((metrics_v2['f1'] - metrics_v1['f1']) / metrics_v1['f1']) * 100
        print(f"Change: {improvement:+.2f}%")
        
        print("\nTraining complete!")
        
    except FileNotFoundError:
        print("\nDataset not found!")
        print("Copy UCI_Credit_Card.csv to data/raw/")
        sys.exit(1)
