from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib

def train_model(X, y, test_size=0.2, random_state=42):
    """Train Linear Regression model and return model + metrics"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred)
    }
    
    return model, metrics

def save_model(model, filepath="model.pkl"):
    """Save the trained model"""
    joblib.dump(model, filepath)

def load_model(filepath="model.pkl"):
    """Load a saved model"""
    return joblib.load(filepath)
