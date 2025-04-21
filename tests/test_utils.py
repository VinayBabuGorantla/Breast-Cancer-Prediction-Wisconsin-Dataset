import numpy as np
from sklearn.linear_model import LinearRegression
from src.utils import evaluate_models

def test_evaluate_models():
    X_train = np.array([[1], [2], [3]])
    y_train = np.array([2, 4, 6])
    X_test = np.array([[4], [5]])
    y_test = np.array([8, 10])
    
    models = {
        "linear": LinearRegression()
    }
    params = {
        "linear": {}
    }

    report = evaluate_models(X_train, y_train, X_test, y_test, models, params)
    assert "linear" in report
    assert report["linear"] > 0.9  # R^2 score
