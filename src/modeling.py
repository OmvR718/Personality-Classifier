from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Load data
X_train = pd.read_csv('Social-MediaVsProductivity/data/proccessed/X_train_final.csv')
X_test = pd.read_csv('Social-MediaVsProductivity/data/proccessed/X_test_final.csv')
y_train = pd.read_csv('Social-MediaVsProductivity/data/proccessed/y_train.csv')
y_test = pd.read_csv('Social-MediaVsProductivity/data/proccessed/y_test.csv')

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

def train_model(model_type='linear', target_col=target_col):
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'decision_tree':
        model = DecisionTreeRegressor()
    elif model_type == 'svm':
        model = SVR()
    else:
        raise ValueError("Unsupported model type. Choose from 'linear', 'decision_tree', or 'svm'.")
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
    return mse, r2

def visualize_residuals(model):
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residuals vs Predicted Values')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
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
    plt.show()

def visualize_predictions(model):
    y_pred = model.predict(X_test)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.show()

def main():
    # Train models
    linear = train_model('linear')
    svr = train_model('svm')
    decision_tree = train_model('decision_tree')

    # Create directory if it doesn't exist
    model_dir = 'Social-MediaVsProductivity/models'
    os.makedirs(model_dir, exist_ok=True)

    # Save models
    try:
        joblib.dump(linear, os.path.join(model_dir, 'linear_model.pkl'))
        joblib.dump(svr, os.path.join(model_dir, 'svr_model.pkl'))
        joblib.dump(decision_tree, os.path.join(model_dir, 'decision_tree_model.pkl'))
        print("Models saved successfully.")
    except Exception as e:
        print(f"Error saving models: {e}")

    return linear, svr, decision_tree

if __name__ == "__main__":
    linear, svr, decision_tree = main()
