from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import re

# Get the absolute path to the current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# Load data using absolute paths
X_train = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_train_final.csv'))
X_test = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_test_final.csv'))
y_train = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_train.csv'))
y_test = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_test.csv'))

# Sanitize feature names for LightGBM compatibility
def sanitize_column(col):
    # Replace any character that is not alphanumeric or underscore with underscore
    return re.sub(r'[^0-9a-zA-Z_]', '_', col)
X_train.columns = [sanitize_column(col) for col in X_train.columns]
X_test.columns = [sanitize_column(col) for col in X_test.columns]

# Validate data shapes
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
print("y_train columns:", y_train.columns)
print("y_test columns:", y_test.columns)

# Select target column (replace 'actual_productivity_score' with the correct column name)
target_col = 'actual_productivity_score'  # Update based on y_train.columns output
y_train = y_train[target_col]
y_test = y_test[target_col]

# Validate shapes after selecting target column
if X_train.shape[0] != y_train.shape[0]:
    raise ValueError(f"Sample mismatch: X_train has {X_train.shape[0]} samples, y_train has {y_train.shape[0]} samples")
if X_test.shape[0] != y_test.shape[0]:
    raise ValueError(f"Sample mismatch: X_test has {X_test.shape[0]} samples, y_test has {y_test.shape[0]} samples")

# Split X_train and y_train into x_train, x_val, y_train, y_val
x_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

def train_model(model_type='linear', target_col=target_col):
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'decision_tree':
        model = DecisionTreeRegressor()
    elif model_type == 'xgb':
        model = XGBRegressor(objective='reg:squarederror', random_state=42)
    elif model_type == 'lgbm':
        model = LGBMRegressor(random_state=42)
    else:
        raise ValueError("Unsupported model type. Choose from 'linear', 'decision_tree', 'xgb', or 'lgbm'.")
    
    model.fit(x_train, y_train)
    return model

def evaluate_model(model):
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(x_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = mse_test ** 0.5
    mse_train = mean_squared_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)
    rmse_train = mse_train ** 0.5
    return {
        'test': {'mse': mse_test, 'rmse': rmse_test, 'r2': r2_test},
        'train': {'mse': mse_train, 'rmse': rmse_train, 'r2': r2_train}
    }

def visualize_residuals(model):
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residuals vs Predicted Values')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = model.coef_
    else:
        raise ValueError("Model does not have feature importances or coefficients.")

    feature_names = X_train.columns
    indices = importances.argsort()[::-1]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=feature_names[indices])
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()

def visualize_predictions(model):
    y_pred = model.predict(X_test)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.tight_layout()
    plt.show()

def tune_decision_tree():
    param_grid = {
        'max_depth': [3, 5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2', None]
    }
    grid = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(x_train, y_train)
    # Evaluate on validation set
    val_score = grid.score(x_val, y_val)
    return grid.best_params_, grid.best_score_, val_score

def tune_xgb():
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
    grid = GridSearchCV(XGBRegressor(objective='reg:squarederror', random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(x_train, y_train)
    val_score = grid.score(x_val, y_val)
    return grid.best_params_, grid.best_score_, val_score

def tune_lgbm():
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, -1],
        'learning_rate': [0.01, 0.1, 0.2],
        'num_leaves': [31, 50, 100]
    }
    grid = GridSearchCV(LGBMRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(x_train, y_train)
    val_score = grid.score(x_val, y_val)
    return grid.best_params_, grid.best_score_, val_score

def train_best_decision_tree():
    """
    Tune and train a DecisionTreeRegressor with best hyperparameters, save the model, and print scores.
    """
    best_params, best_cv_score, val_score = tune_decision_tree()
    print(f"Best Decision Tree Params: {best_params}")
    print(f"Best CV Score: {best_cv_score}")
    print(f"Validation Score: {val_score}")
    model = DecisionTreeRegressor(**best_params, random_state=42)
    model.fit(x_train, y_train)
    # Save the model
    joblib.dump(model, os.path.join(BASE_DIR, '../models/decision_tree_model.pkl'))
    # Evaluate
    metrics = evaluate_model(model)
    print('Train:', metrics['train'])
    print('Test:', metrics['test'])
    return model, metrics

def main():
    # Train models
    linear = train_model('linear')
    xgb = train_model('xgb')
    lgbm = train_model('lgbm')
    decision_tree = train_model('decision_tree')

    # Create directory if it doesn't exist
    model_dir = os.path.join(BASE_DIR, '..', 'models')
    os.makedirs(model_dir, exist_ok=True)

    # Save models
    try:
        joblib.dump(linear, os.path.join(model_dir, 'linear_model.pkl'))
        joblib.dump(xgb, os.path.join(model_dir, 'xgb_model.pkl'))
        joblib.dump(lgbm, os.path.join(model_dir, 'lgbm_model.pkl'))
        joblib.dump(decision_tree, os.path.join(model_dir, 'decision_tree_model.pkl'))
        print("Models saved successfully.")
    except Exception as e:
        print(f"Error saving models: {e}")

    return linear, xgb, lgbm, decision_tree

if __name__ == "__main__":
    linear, xgb, lgbm, decision_tree = main()
