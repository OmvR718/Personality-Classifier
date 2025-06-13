import streamlit as st
import pandas as pd
from src import data_processing, eda

st.set_page_config(page_title="Social Media vs Productivity", layout="wide")
st.title("Social Media vs Productivity Dashboard")

# Sidebar navigation
page = st.sidebar.selectbox(
    "Choose a section",
    ("Data Processing", "Exploratory Data Analysis", "Modeling")
)

if page == "Data Processing":
    st.header("Data Processing")
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file:
        st.subheader("Raw Data")
        st.code("""data = data_processing.load_data(uploaded_file)""")
        data = data_processing.load_data(uploaded_file)
        st.write(data)

        st.subheader("Cleaned Data")
        st.code("""cleaned = data_processing.clean_data(data)""")
        cleaned = data_processing.clean_data(data)
        st.write(cleaned)

        st.subheader("Features (X) and Target (y)")
        st.code("""X, y = data_processing.split_into_x_y(cleaned)""")
        X, y = data_processing.split_into_x_y(cleaned)
        st.write(X)
        st.write(y)

        st.subheader("Train/Test Split")
        st.code("""X_train, X_test, y_train, y_test = data_processing.split_data(X, y)""")
        X_train, X_test, y_train, y_test = data_processing.split_data(X, y)
        st.write(X_train)
        st.write(X_test)
        st.write(y_train)
        st.write(y_test)

        st.subheader("Split Features")
        st.code("""X_train_num, X_test_num, X_train_ord, X_test_ord, X_train_nom, X_test_nom = data_processing.split_features(X_train, X_test)""")
        X_train_num, X_test_num, X_train_ord, X_test_ord, X_train_nom, X_test_nom = data_processing.split_features(X_train, X_test)
        st.write(X_train_num)
        st.write(X_test_num)
        st.write(X_train_ord)
        st.write(X_test_ord)
        st.write(X_train_nom)
        st.write(X_test_nom)

        st.subheader("One-Hot Encode Nominal Features")
        st.code("""X_train_nom_encoded, X_test_nom_encoded = data_processing.one_hot_encode_nominal(X_train_nom, X_test_nom)""")
        X_train_nom_encoded, X_test_nom_encoded = data_processing.one_hot_encode_nominal(X_train_nom, X_test_nom)
        st.write(X_train_nom_encoded)
        st.write(X_test_nom_encoded)

        st.subheader("Process Numerical Features")
        st.code("""X_train_num_proc, X_test_num_proc = data_processing.preproccess_num(X_train_num, X_test_num)""")
        X_train_num_proc, X_test_num_proc = data_processing.preproccess_num(X_train_num, X_test_num)
        st.write(X_train_num_proc)
        st.write(X_test_num_proc)

        st.subheader("Process Ordinal Features")
        st.code("""X_train_ord_proc, X_test_ord_proc = data_processing.preproccess_ord(X_train_ord, X_test_ord)""")
        X_train_ord_proc, X_test_ord_proc = data_processing.preproccess_ord(X_train_ord, X_test_ord)
        st.write(X_train_ord_proc)
        st.write(X_test_ord_proc)

        st.subheader("Combine All Features")
        st.code("""X_train_final, X_test_final = data_processing.combine_features(
    X_train_num_proc, X_test_num_proc,
    X_train_ord_proc, X_test_ord_proc,
    X_train_nom_encoded, X_test_nom_encoded
)""")
        X_train_final, X_test_final = data_processing.combine_features(
            X_train_num_proc, X_test_num_proc,
            X_train_ord_proc, X_test_ord_proc,
            X_train_nom_encoded, X_test_nom_encoded
        )
        st.write(X_train_final)
        st.write(X_test_final)

elif page == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")
    df = pd.read_csv("src/data/raw/social_media_vs_productivity.csv")
    st.write("Dataset", df.head())

    eda_plot = st.selectbox(
        "Choose an EDA visualization",
        [
            "Correlation Matrix",
            "Univariate Analysis",
            "Productivity Gap",
            "Notifications per Hour",
            "Binned Notifications",
            "Balanced Violin Plots",
            "Job Type Counts",
            "Stress by Job Type",
            "Social Platform Pie",
            "Sleep Hours by Stress Group"
        ]
    )
    if eda_plot == "Correlation Matrix":
        st.subheader("Correlation Matrix")
        fig = eda.plot_correlation_matrix(df)
        st.pyplot(fig)
    elif eda_plot == "Univariate Analysis":
        st.subheader("Univariate Analysis")
        if hasattr(eda, 'plot_numeric_columns'):
            num_cols = [
                'age', 'daily_social_media_time', 'number_of_notifications', 'work_hours_per_day',
                'perceived_productivity_score', 'stress_level', 'sleep_hours', 'screen_time_before_sleep',
                'breaks_during_work', 'coffee_consumption_per_day', 'days_feeling_burnout_per_month',
                'weekly_offline_hours', 'job_satisfaction_score', 'actual_productivity_score'
            ]
            available_cols = [col for col in num_cols if col in df.columns]
            selected_col = st.selectbox("Select numeric column for univariate analysis", available_cols, key='univariate_col')
            fig = eda.plot_numeric_columns(df, selected_col)
            st.pyplot(fig)
        else:
            st.info("Univariate analysis function not implemented.")
    elif eda_plot == "Productivity Gap":
        st.subheader("Productivity Gap")
        if hasattr(eda, 'plot_productivity_gap'):
            fig = eda.plot_productivity_gap(df)
            if fig is not None:
                st.pyplot(fig)
            else:
                st.info("No figure returned by plot_productivity_gap.")
        else:
            st.info("Productivity gap plot function not implemented.")
    elif eda_plot == "Notifications per Hour":
        st.subheader("Notifications per Hour")
        if hasattr(eda, 'plot_notifications_per_hour'):
            fig = eda.plot_notifications_per_hour(df)
            st.pyplot(fig)
        else:
            st.info("Notifications per hour plot function not implemented.")
    elif eda_plot == "Binned Notifications":
        st.subheader("Binned Notifications")
        # Ensure 'notifications_per_hour' exists
        if 'notifications_per_hour' not in df.columns:
            df = df.copy()
            df["notifications_per_hour"] = (1/16) * (df["number_of_notifications"] / df["work_hours_per_day"])
        if hasattr(eda, 'plot_binned_notifications'):
            fig = eda.plot_binned_notifications(df)
            st.pyplot(fig)
        else:
            st.info("Binned notifications plot function not implemented.")
    elif eda_plot == "Balanced Violin Plots":
        st.subheader("Balanced Violin Plots with Correlation")
        # Clean data to avoid NaN/inf errors in violin/correlation plot
        required_cols = ["uses_focus_apps", "has_digital_wellbeing_enabled", "daily_social_media_time"]
        df_clean = df.copy()
        for col in required_cols:
            if col in df_clean.columns:
                df_clean = df_clean[pd.to_numeric(df_clean[col], errors='coerce').notnull()]
        df_clean = df_clean.dropna(subset=required_cols)
        df_clean = df_clean[~df_clean[required_cols].isin([float('inf'), float('-inf')]).any(axis=1)]
        if hasattr(eda, 'plot_balanced_violin_plots_with_correlation'):
            fig = eda.plot_balanced_violin_plots_with_correlation(df_clean)
            st.pyplot(fig)
        else:
            st.info("Balanced violin plots function not implemented.")
    elif eda_plot == "Job Type Counts":
        st.subheader("Job Type Counts")
        if hasattr(eda, 'plot_job_type_counts'):
            if 'job_type' in df.columns:
                job_type_counts = df['job_type'].value_counts()
                fig = eda.plot_job_type_counts(job_type_counts)
                st.pyplot(fig)
            else:
                st.info("job_type column not found in data.")
        else:
            st.info("Job type counts plot function not implemented.")
    elif eda_plot == "Stress by Job Type":
        st.subheader("Stress by Job Type")
        if hasattr(eda, 'plot_stress_by_job_type'):
            fig = eda.plot_stress_by_job_type(df)
            st.pyplot(fig)
        else:
            st.info("Stress by job type plot function not implemented.")
    elif eda_plot == "Social Platform Pie":
        st.subheader("Social Platform Pie Chart")
        if hasattr(eda, 'plot_social_platform_pie'):
            fig = eda.plot_social_platform_pie(df)
            st.pyplot(fig)
        else:
            st.info("Social platform pie chart function not implemented.")
    elif eda_plot == "Sleep Hours by Stress Group":
        st.subheader("Sleep Hours by Stress Group (Split at Mean Stress Level)")
        if hasattr(eda, 'plot_stress_vs_sleep_split_by_stress'):
            fig = eda.plot_stress_vs_sleep_split_by_stress(df)
            st.pyplot(fig)
        else:
            st.info("Sleep vs stress split plot function not implemented.")

elif page == "Modeling":
    st.header("Modeling")
    st.write("Select a model to load and evaluate:")
    model_choice = st.selectbox("Choose a regression model", ["Linear Regression", "XGBoost", "LightGBM", "Decision Tree"])
    model_path = None
    if model_choice == "Linear Regression":
        model_path = "models/linear_model.pkl"
    elif model_choice == "XGBoost":
        model_path = "models/xgb_model.pkl"
    elif model_choice == "LightGBM":
        model_path = "models/lgbm_model.pkl"
    elif model_choice == "Decision Tree":
        model_path = "models/decision_tree_model.pkl"

    if model_path:
        import joblib
        import matplotlib.pyplot as plt
        from src import modeling
        model = joblib.load(model_path)
        st.success(f"Loaded {model_choice} model.")
        # Robustly display model class and parameters
        st.write(f"Model: {type(model).__name__}")
        try:
            params = model.get_params()
            st.write("Parameters:", params)
        except Exception as e:
            st.info(f"Could not display model parameters due to version mismatch or missing attributes. ({e})")
        # Use modeling.py's evaluate_model (which loads X_test, y_test internally)
        metrics = modeling.evaluate_model(model)
        st.subheader('Test Set Metrics')
        st.metric("MSE (Test)", f"{metrics['test']['mse']:.3f}")
        st.metric("RMSE (Test)", f"{metrics['test']['rmse']:.3f}")
        st.metric("R2 Score (Test)", f"{metrics['test']['r2']:.3f}")
        st.subheader('Train Set Metrics')
        st.metric("MSE (Train)", f"{metrics['train']['mse']:.3f}")
        st.metric("RMSE (Train)", f"{metrics['train']['rmse']:.3f}")
        st.metric("R2 Score (Train)", f"{metrics['train']['r2']:.3f}")

        st.subheader("Residuals vs Predicted Values")
        modeling.visualize_residuals(model)
        st.pyplot(plt.gcf())

        st.subheader("Feature Importance / Coefficients")
        try:
            modeling.plot_feature_importance(model)
            st.pyplot(plt.gcf())
        except Exception as e:
            st.info(f"No feature importances or coefficients available. ({e})")

        st.subheader("Actual vs Predicted Values")
        modeling.visualize_predictions(model)
        st.pyplot(plt.gcf())
