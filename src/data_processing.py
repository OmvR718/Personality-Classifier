import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
scaler = RobustScaler()
ohe=OneHotEncoder(drop='first', sparse_output=False,handle_unknown='ignore')
# Identify column types based on the dataset
numerical_cols = [
    'age', 'daily_social_media_time', 'number_of_notifications', 'work_hours_per_day',
    'perceived_productivity_score', 'stress_level', 'sleep_hours', 'screen_time_before_sleep',
    'breaks_during_work', 'coffee_consumption_per_day', 'days_feeling_burnout_per_month',
    'weekly_offline_hours', 'job_satisfaction_score'
]
ordinal_cols = ['uses_focus_apps', 'has_digital_wellbeing_enabled']
nominal_cols = ['gender', 'job_type', 'social_platform_preference']

def load_data(file_path):
    """
    Load data from a CSV file.
    
    Parameters:
    file_path (str): The path to the CSV file.
    
    Returns:
    pd.DataFrame: The loaded data as a DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
def clean_data(data):
    """
    Clean the data by removing rows with missing values.
    
    Parameters:
    data (pd.DataFrame): The data to clean.
    
    Returns:
    pd.DataFrame: The cleaned data.
    """
    cleaned_data = data.dropna(subset=['actual_productivity_score'])
    return cleaned_data
def split_into_x_y(data):
    """
    Split the DataFrame into features (X) and target variable (y).
    
    Parameters:
    data (pd.DataFrame): The DataFrame to split.
    
    Returns:
    tuple: (X, y) where X is the features DataFrame and y is the target Series.
    """
    X = data.drop(columns=['actual_productivity_score'])
    y = data['actual_productivity_score']
    
    return X, y

def split_data(X , y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Parameters:
    data (pd.DataFrame): The data to split.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): Controls the shuffling applied to the data before applying the split.
    
    Returns:
    tuple: Training and testing DataFrames.
    """
    X_train, X_test,y_train,y_test = train_test_split(X,y, test_size=test_size, random_state=random_state)
    # Ensure the indices are sorted for consistency
    X_train = X_train.sort_index()
    X_test = X_test.sort_index()
    y_train = y_train.sort_index()
    y_test = y_test.sort_index()
    
    return X_train, X_test,y_train,y_test
def split_features(X_train, X_test):
    """
    Split features into numerical, ordinal, and nominal DataFrames.
    
    Parameters:
    X (pd.DataFrame): The DataFrame containing all features.
    
    Returns:
    tuple: (X_num, X_ord, X_nom) where each is a DataFrame of the respective feature type.
    """
    X_train_num = X_train[numerical_cols]
    X_test_num = X_test[numerical_cols]
    X_train_ord = X_train[ordinal_cols]
    X_test_ord = X_test[ordinal_cols]
    X_train_nom = X_train[nominal_cols]
    X_test_nom = X_test[nominal_cols]
    
    return X_train_num, X_test_num,X_train_ord, X_test_ord, X_train_nom, X_test_nom
def one_hot_encode_nominal(X_train_nom, X_test_nom):
    """
    One-hot encode nominal columns for train and test sets.
    
    Parameters:
    X_train_nom (pd.DataFrame): Training data with nominal columns
    X_test_nom (pd.DataFrame): Test data with nominal columns
    
    Returns:
    tuple: (X_train_nom_encoded, X_test_nom_encoded) as pandas DataFrames
    """
    # Initialize encoder
    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    
    # Fit and transform training data
    X_train_nom_encoded = pd.DataFrame(
        encoder.fit_transform(X_train_nom),
        index=X_train_nom.index,
        columns=encoder.get_feature_names_out(X_train_nom.columns)
    )
    
    # Transform test data
    X_test_nom_encoded = pd.DataFrame(
        encoder.transform(X_test_nom),
        index=X_test_nom.index,
        columns=encoder.get_feature_names_out(X_test_nom.columns)
    )
    
    return X_train_nom_encoded, X_test_nom_encoded

def preproccess_num(X_train_num, X_test_num):
    """
    Scale specified features in the DataFrame.
    
    Parameters:
    X_train_num, X_test_num (pd.DataFrame): DataFrames with numerical features.
    
    Returns:
    pd.DataFrame: DataFrames with scaled and imputed features.
    """
    columns = [
        'daily_social_media_time',
        'perceived_productivity_score',
        'stress_level',
        'sleep_hours',
        'screen_time_before_sleep',
        'job_satisfaction_score'
    ]
    # Impute missing values first
    X_train_num[columns] = imputer.fit_transform(X_train_num[columns])
    X_test_num[columns] = imputer.transform(X_test_num[columns])
    # Scale all numerical features
    X_train_num_scaled = pd.DataFrame(scaler.fit_transform(X_train_num), columns=X_train_num.columns, index=X_train_num.index)
    X_test_num_scaled = pd.DataFrame(scaler.transform(X_test_num), columns=X_test_num.columns, index=X_test_num.index)
    return X_train_num_scaled, X_test_num_scaled
def preproccess_ord(X_train_ord, X_test_ord):
    """
    Encode ordinal features in the DataFrame.
    
    Parameters:
    X_train_ord, X_test_ord (pd.DataFrame): DataFrames with ordinal features.
    
    Returns:
    pd.DataFrame: DataFrames with encoded ordinal features.
    """
    X_train_ord = X_train_ord.copy()
    X_test_ord = X_test_ord.copy()

    # Map boolean values directly for uses_focus_apps
    X_train_ord['uses_focus_apps'] = X_train_ord['uses_focus_apps'].map({True: 1, False: 0})
    X_test_ord['uses_focus_apps'] = X_test_ord['uses_focus_apps'].map({True: 1, False: 0})

    # Try both boolean and string mapping for has_digital_wellbeing_enabled
    if X_train_ord['has_digital_wellbeing_enabled'].dropna().isin([True, False]).all():
        X_train_ord['has_digital_wellbeing_enabled'] = X_train_ord['has_digital_wellbeing_enabled'].map({True: 1, False: 0})
        X_test_ord['has_digital_wellbeing_enabled'] = X_test_ord['has_digital_wellbeing_enabled'].map({True: 1, False: 0})
    else:
        X_train_ord['has_digital_wellbeing_enabled'] = X_train_ord['has_digital_wellbeing_enabled'].map({'Yes': 1, 'No': 0})
        X_test_ord['has_digital_wellbeing_enabled'] = X_test_ord['has_digital_wellbeing_enabled'].map({'Yes': 1, 'No': 0})

    return X_train_ord, X_test_ord
def combine_features(X_train_num, X_test_num, X_train_ord, X_test_ord, X_train_nom_encoded, X_test_nom_encoded):
    """
    Combine numerical, ordinal, and nominal features into a single DataFrame.
    
    Parameters:
    X_train_num (pd.DataFrame): Training data with numerical features
    X_test_num (pd.DataFrame): Test data with numerical features
    X_train_ord (pd.DataFrame): Training data with ordinal features
    X_test_ord (pd.DataFrame): Test data with ordinal features
    X_train_nom_encoded (pd.DataFrame): Training data with encoded nominal features
    X_test_nom_encoded (pd.DataFrame): Test data with encoded nominal features
    
    Returns:
    tuple: Combined training and test DataFrames
    """
    # Combine all features into a single DataFrame
    X_train_combined = pd.concat([X_train_num, X_train_ord, X_train_nom_encoded], axis=1)
    X_test_combined = pd.concat([X_test_num, X_test_ord, X_test_nom_encoded], axis=1)
    
    return X_train_combined, X_test_combined