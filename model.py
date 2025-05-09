import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

class GoldPricePredictor:
    """
    A class for predicting gold prices using machine learning models
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the GoldPricePredictor with a specified model type
        
        Args:
            model_type (str): Type of model to use - 'random_forest', 'gradient_boosting', or 'linear'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        elif model_type == 'linear':
            self.model = LinearRegression()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def train(self, X_train, y_train, optimize=False):
        """
        Train the prediction model
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training target values
            optimize (bool): Whether to optimize hyperparameters
            
        Returns:
            GoldPricePredictor: Self reference for method chaining
        """
        if X_train.shape[0] == 0 or y_train.shape[0] == 0:
            raise ValueError("Training data is empty")
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if optimize and self.model_type == 'random_forest':
            # Define hyperparameter grid
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            # Use time series split for validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Grid search with time series validation
            grid_search = GridSearchCV(
                estimator=self.model,
                param_grid=param_grid,
                cv=tscv,
                n_jobs=-1,
                scoring='neg_mean_squared_error'
            )
            
            grid_search.fit(X_train_scaled, y_train)
            
            # Update model with best parameters
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            # Train the model with default parameters
            self.model.fit(X_train_scaled, y_train)
        
        return self
    
    def predict(self, X_test):
        """
        Make price predictions using the trained model
        
        Args:
            X_test (numpy.ndarray): Test features
            
        Returns:
            numpy.ndarray: Predicted values
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Scale the test features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        predictions = self.model.predict(X_test_scaled)
        
        return predictions
    
    def evaluate(self, y_true, y_pred):
        """
        Evaluate model performance using various metrics
        
        Args:
            y_true (numpy.ndarray): True target values
            y_pred (numpy.ndarray): Predicted target values
            
        Returns:
            tuple: (mse, mae, r2) - Mean squared error, mean absolute error, and R^2 score
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return mse, mae, r2
    
    def get_feature_importance(self, feature_names):
        """
        Get the importance of each feature in the model
        
        Args:
            feature_names (list): Names of features
            
        Returns:
            pandas.DataFrame: Feature importance data
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        if hasattr(self.model, 'feature_importances_'):
            # Create a dataframe of feature importances
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            feature_importance = pd.DataFrame({
                'Feature': [feature_names[i] for i in indices],
                'Importance': importances[indices]
            })
            
            return feature_importance
        else:
            # For models without feature_importances_ attribute (like LinearRegression)
            if isinstance(self.model, LinearRegression):
                importances = np.abs(self.model.coef_)
                indices = np.argsort(importances)[::-1]
                
                feature_importance = pd.DataFrame({
                    'Feature': [feature_names[i] for i in indices],
                    'Importance': importances[indices]
                })
                
                return feature_importance
            else:
                return pd.DataFrame({'Feature': feature_names, 'Importance': 0})
    
    def forecast_future(self, last_data_point, days_ahead, feature_window):
        """
        Forecast gold prices for a specified number of days ahead
        
        Args:
            last_data_point (numpy.ndarray): Last known data point
            days_ahead (int): Number of days to forecast ahead
            feature_window (int): Number of days used for features
            
        Returns:
            numpy.ndarray: Forecasted prices
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Initialize array to store forecasts
        forecasts = np.zeros(days_ahead)
        
        # Make a copy of the last data point to update with predictions
        current_features = last_data_point.copy()
        
        # Forecast one day at a time
        for i in range(days_ahead):
            # Scale the features
            scaled_features = self.scaler.transform(current_features.reshape(1, -1))
            
            # Make prediction for the next day
            next_day_forecast = self.model.predict(scaled_features)[0]
            
            # Store the forecast
            forecasts[i] = next_day_forecast
            
            # Update the features for the next prediction
            # This depends on how the features are organized
            # Assuming the features are lagged prices, we update by shifting and adding the new prediction
            current_features = np.roll(current_features, -1)
            current_features[-1] = next_day_forecast
        
        return forecasts
