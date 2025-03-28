import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

def train_arima_with_exog(train_X: pd.DataFrame, train_y: pd.DataFrame, test_X: pd.DataFrame, test_y: pd.DataFrame, target: str):
    # Fillna with mean -> Definitely not best method, but for simplicity
    train_X = train_X.fillna(train_X.mean())
    test_X = test_X.fillna(test_X.mean())
    
    # Standardize the exogenous variables
    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    test_X_scaled = scaler.transform(test_X)

    # Train Lasso model for feature selection
    lasso = Lasso(alpha=0.01)
    lasso.fit(train_X_scaled, train_y)
    
    # Select non-zero coefficients
    selected_features = train_X.columns[lasso.coef_ != 0]
    
    # Prepare exogenous variables for ARIMA
    train_exog = train_X[selected_features]
    test_exog = test_X[selected_features]
    
    # Train ARIMA model with exogenous variables
    arima_model = ARIMA(train_y, exog=train_exog, order=(5,1,0))
    arima_result = arima_model.fit()
    
    # Make predictions
    predictions = arima_result.predict(start=len(train_y), end=len(train_y) + len(test_y) - 1, exog=test_exog)
    
    return arima_result, predictions