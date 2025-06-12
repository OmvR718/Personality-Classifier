import matplotlib.pyplot as plt
import seaborn as sns
from src.data_processing import load_data
import pandas as pd
from scipy.stats import pointbiserialr
from sklearn.utils import resample

numerical_cols = [
    'age', 'daily_social_media_time', 'number_of_notifications', 'work_hours_per_day',
    'perceived_productivity_score', 'stress_level', 'sleep_hours', 'screen_time_before_sleep',
    'breaks_during_work', 'coffee_consumption_per_day', 'days_feeling_burnout_per_month',
    'weekly_offline_hours', 'job_satisfaction_score','actual_productivity_score'
]
def plot_correlation_matrix(df):
    """
    Create a heatmap to visualize the correlation matrix of the DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the numeric columns
    
    Returns:
    fig: The matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    # Use only the specified numerical columns for correlation
    numeric_df = df[[col for col in numerical_cols if col in df.columns]]
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8}, ax=ax)
    ax.set_title('Correlation Matrix')
    return fig
def plot_numeric_columns(df, col):
    """
    Create a histogram and boxplot for a single numeric column.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the numeric column
    col (str): Column name to plot
    
    Returns:
    fig: The matplotlib figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # Plot histogram with KDE
    sns.histplot(df[col], kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title(f'Histogram of {col}')
    # Plot boxplot
    sns.boxplot(x=df[col], ax=axes[1], color='lightgreen')
    axes[1].set_title(f'Boxplot of {col}')
    plt.tight_layout()
    return fig
def plot_productivity_gap(df, sample_frac=0.5, random_state=42, window=100):
    """
    Create a scatter plot of productivity gap vs actual productivity score with a rolling mean trend line.
    Returns a matplotlib figure for Streamlit.
    """
    # Calculate productivity gap
    df = df.copy()
    df["productivity_gap"] = df["perceived_productivity_score"] - df['actual_productivity_score']
    # Use a random sample for better visualization
    sample = df.sample(frac=sample_frac, random_state=random_state)
    actual_scores_sample = df['actual_productivity_score'].loc[sample.index]
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=actual_scores_sample, y=sample["productivity_gap"], alpha=0.6, color="teal", ax=ax)
    # Calculate and plot rolling mean for trend line
    sample_sorted = sample.copy()
    sample_sorted["actual_productivity_score"] = actual_scores_sample
    sample_sorted = sample_sorted.sort_values("actual_productivity_score")
    rolling_mean = sample_sorted["productivity_gap"].rolling(window=window, min_periods=1).mean()
    sns.lineplot(
        x=sample_sorted["actual_productivity_score"],
        y=rolling_mean,
        color="red",
        label=f"Rolling Mean (window={window})",
        ax=ax
    )
    ax.set_title("Productivity Gap vs Actual Productivity Score")
    ax.set_xlabel("Actual Productivity Score")
    ax.set_ylabel("Productivity Gap (Perceived - Actual)")
    ax.legend()
    fig.tight_layout()
    return fig
    
def plot_notifications_per_hour(df, wh_perday=1/16, bins=30):
    """
    Create a histogram of notifications per hour with KDE.
    Parameters:
    df (pd.DataFrame): DataFrame containing number_of_notifications and work_hours_per_day
    wh_perday (float): Weight for notifications per day calculation (default: 1/16)
    bins (int): Number of bins for histogram (default: 30)
    """
    df = df.copy()
    df["notifications_per_hour"] = wh_perday * (df["number_of_notifications"] / df["work_hours_per_day"])
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["notifications_per_hour"], kde=True, bins=bins, color='purple', ax=ax)
    ax.set_title("Distribution of Notifications per Hour")
    ax.set_xlabel("Notifications per Hour")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig

def plot_binned_notifications(df, bins=[-2, -1, 0, 1, 2], labels=["Very Low", "Low", "Moderate", "High", "Very High"]):
    """
    Create a count plot of binned notifications per hour.
    Parameters:
    df (pd.DataFrame): DataFrame containing notifications_per_hour column
    bins (list): Bin edges for notifications per hour (default: [-2, -1, 0, 1, 2, max])
    labels (list): Labels for bins (default: ["Very Low", "Low", "Moderate", "High", "Very High"])
    """
    df = df.copy()
    bins = bins + [df["notifications_per_hour"].max()] if bins[-1] != df["notifications_per_hour"].max() else bins
    df["notif_bin"] = pd.cut(
        df["notifications_per_hour"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x="notif_bin", data=df, palette="viridis", hue="notif_bin", ax=ax)
    ax.set_title("Distribution of Notifications per Hour Binned")
    ax.set_xlabel("Notifications per Hour (Binned)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig

def balance_classes(df, col, y_col):
    """
    Balance classes for a binary column by downsampling the majority class.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    col (str): Binary column to balance
    y_col (str): Target column for analysis
    
    Returns:
    pd.DataFrame: Balanced DataFrame
    """
    class_0 = df[df[col] == 0]
    class_1 = df[df[col] == 1]
    n_samples = min(len(class_0), len(class_1))
    class_0_bal = resample(class_0, replace=False, n_samples=n_samples, random_state=42)
    class_1_bal = resample(class_1, replace=False, n_samples=n_samples, random_state=42)
    return pd.concat([class_0_bal, class_1_bal])

def plot_balanced_violin_plots_with_correlation(df, focus_col='uses_focus_apps', wellbeing_col='has_digital_wellbeing_enabled', y_col='daily_social_media_time'):
    """
    Create violin plots for daily social media time by focus apps and digital wellbeing usage with balanced classes,
    including point-biserial correlation statistics.
    """
    from matplotlib import pyplot as plt
    # Clean data to avoid NaN/inf errors
    df = df.copy()
    for col in [focus_col, wellbeing_col, y_col]:
        df = df[pd.to_numeric(df[col], errors='coerce').notnull()]
    df = df.dropna(subset=[focus_col, wellbeing_col, y_col])
    df = df[~df[[focus_col, wellbeing_col, y_col]].isin([float('inf'), float('-inf')]).any(axis=1)]
    class_0_focus = df[df[focus_col] == 0]
    class_1_focus = df[df[focus_col] == 1]
    n_samples_focus = min(len(class_0_focus), len(class_1_focus))
    class_0_bal_focus = resample(class_0_focus, replace=False, n_samples=n_samples_focus, random_state=42)
    class_1_bal_focus = resample(class_1_focus, replace=False, n_samples=n_samples_focus, random_state=42)
    balanced_focus = pd.concat([class_0_bal_focus, class_1_bal_focus])
    class_0_wellbeing = df[df[wellbeing_col] == 0]
    class_1_wellbeing = df[df[wellbeing_col] == 1]
    n_samples_wellbeing = min(len(class_0_wellbeing), len(class_1_wellbeing))
    class_0_bal_wellbeing = resample(class_0_wellbeing, replace=False, n_samples=n_samples_wellbeing, random_state=42)
    class_1_bal_wellbeing = resample(class_1_wellbeing, replace=False, n_samples=n_samples_wellbeing, random_state=42)
    balanced_wellbeing = pd.concat([class_0_bal_wellbeing, class_1_bal_wellbeing])
    corr_focus, p_corr_focus = pointbiserialr(balanced_focus[focus_col], balanced_focus[y_col])
    corr_wellbeing, p_corr_wellbeing = pointbiserialr(balanced_wellbeing[wellbeing_col], balanced_wellbeing[y_col])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.violinplot(data=balanced_focus, x=focus_col, y=y_col, split=True, inner='quart', palette=['steelblue', 'orange'], hue=focus_col, ax=axes[0])
    axes[0].set_title('Daily Social Media Time by Focus App Usage (Balanced)')
    axes[0].set_xlabel('Uses Focus Apps')
    axes[0].set_ylabel('Daily Social Media Time')
    groups_focus = balanced_focus[focus_col].unique()
    for i, group in enumerate(sorted(groups_focus)):
        group_data = balanced_focus[balanced_focus[focus_col] == group][y_col]
        mean = group_data.mean()
        std = group_data.std()
        axes[0].text(i, mean + std + 0.2, f"μ={mean:.4f}\nσ={std:.4f}", ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')
    axes[0].text(0.5, -0.15, f"r={corr_focus:.3f}, p={p_corr_focus:.4f}", ha='center', va='top', fontsize=10, color='black', fontweight='bold', transform=axes[0].transAxes)
    sns.violinplot(data=balanced_wellbeing, x=wellbeing_col, y=y_col, split=True, inner='quart', palette=['steelblue', 'orange'], hue=wellbeing_col, ax=axes[1])
    axes[1].set_title('Daily Social Media Time by Digital Wellbeing (Balanced)')
    axes[1].set_xlabel('Digital Wellbeing Enabled')
    axes[1].set_ylabel('Daily Social Media Time')
    groups_wellbeing = balanced_wellbeing[wellbeing_col].unique()
    for i, group in enumerate(sorted(groups_wellbeing)):
        group_data = balanced_wellbeing[balanced_wellbeing[wellbeing_col] == group][y_col]
        mean = group_data.mean()
        std = group_data.std()
        axes[1].text(i, mean + std + 0.2, f"μ={mean:.4f}\nσ={std:.4f}", ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')
    axes[1].text(0.5, -0.15, f"r={corr_wellbeing:.3f}, p={p_corr_wellbeing:.4f}", ha='center', va='top', fontsize=10, color='black', fontweight='bold', transform=axes[1].transAxes)
    fig.tight_layout()
    return fig

def plot_job_type_counts(job_type_counts):
    fig, ax = plt.subplots(figsize=(8, 4))
    # Assign x to hue and set legend=False to avoid deprecation warning
    sns.barplot(x=job_type_counts.index, y=job_type_counts.values, hue=job_type_counts.index, palette="viridis", ax=ax, legend=False)
    ax.set_title("Count of Each Job Type in Training Data")
    ax.set_xlabel("Job Type")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    fig.tight_layout()
    return fig

def plot_stress_by_job_type(df, job_type_col='job_type', stress_col='stress_level'):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x=df[job_type_col], y=df[stress_col], palette="coolwarm", hue=df[job_type_col], ax=ax)
    ax.set_title("Stress Level by Job Type")
    ax.set_xlabel("Job Type")
    ax.set_ylabel("Stress Level")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    fig.tight_layout()
    return fig

def plot_social_platform_pie(df, platform_col='social_platform_preference'):
    platform_counts = df[platform_col].value_counts()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(platform_counts, labels=platform_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    ax.set_title('Most Favored Social Platform (Training Set)')
    ax.axis('equal')
    return fig

def plot_stress_vs_sleep_split_by_stress(df, stress_col='stress_level', sleep_col='sleep_hours'):
    """
    Plot sleep hours for low vs high stress groups, split by the mean of stress_level.
    Shows boxplots and swarmplots for each group (swarmplot is sampled for speed).
    Adds group means and stds as text annotations.
    """
    import numpy as np
    df = df.copy()
    mean_stress = df[stress_col].mean()
    df['Stress Group'] = np.where(df[stress_col] > mean_stress, 'High Stress', 'Low Stress')
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='Stress Group', y=sleep_col, data=df, palette=['#4C72B0', '#DD8452'], ax=ax)
    # Sample for swarmplot to avoid lag
    if len(df) > 500:
        df_swarm = df.sample(n=500, random_state=42)
    else:
        df_swarm = df
    sns.swarmplot(x='Stress Group', y=sleep_col, data=df_swarm, color='k', alpha=0.5, ax=ax)
    # Add group means and stds as text
    for i, group in enumerate(['Low Stress', 'High Stress']):
        group_data = df[df['Stress Group'] == group][sleep_col]
        mean = group_data.mean()
        std = group_data.std()
        count = group_data.count()
        ax.text(i, mean + std + 0.2, f"n={count}\nμ={mean:.2f}\nσ={std:.2f}",
                ha='center', va='bottom', fontsize=11, color='black', fontweight='bold')
    ax.set_title('Sleep Hours by Stress Group (Split at Mean Stress Level)')
    ax.set_xlabel('Stress Group')
    ax.set_ylabel('Sleep Hours')
    fig.tight_layout()
    return fig