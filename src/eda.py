import matplotlib.pyplot as plt
import seaborn as sns
from data_processing import load_data
import pandas as pd
from scipy.stats import pointbiserialr
from sklearn.utils import resample
df=pd.read_csv('C:\Users\omara\OneDrive\Desktop\Assignments Course\Repos\Course\Social Media\Social-MediaVsProductivity\data\raw\social_media_vs_productivity.csv')
numerical_cols = [
    'age', 'daily_social_media_time', 'number_of_notifications', 'work_hours_per_day',
    'perceived_productivity_score', 'stress_level', 'sleep_hours', 'screen_time_before_sleep',
    'breaks_during_work', 'coffee_consumption_per_day', 'days_feeling_burnout_per_month',
    'weekly_offline_hours', 'job_satisfaction_score'
]
def plot_correlation_matrix(df):
    """
    Create a heatmap to visualize the correlation matrix of the DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the numeric columns
    """
    plt.figure(figsize=(12, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix')
    plt.show()
def plot_numeric_columns(df, columns):
    """
    Create histograms and boxplots for specified numeric columns.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the numeric columns
    num_cols (list): List of column names to plot
    """
    for col in columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot histogram with KDE
        sns.histplot(df[col], kde=True, ax=axes[0], color='skyblue')
        axes[0].set_title(f'Histogram of {col}')
        
        # Plot boxplot
        sns.boxplot(x=df[col], ax=axes[1], color='lightgreen')
        axes[1].set_title(f'Boxplot of {col}')
        
        plt.tight_layout()
        plt.show() 
def plot_productivity_gap(df,sample_frac=0.5, random_state=42, window=100):
    """
    Create a scatter plot of productivity gap vs actual productivity score with a rolling mean trend line.
    
    Parameters:
    X_train_final (pd.DataFrame): DataFrame containing perceived productivity scores
    y_train (pd.Series): Series containing actual productivity scores
    sample_frac (float): Fraction of data to sample for visualization (default: 0.5)
    random_state (int): Random seed for reproducibility (default: 42)
    window (int): Window size for rolling mean (default: 100)
    """
    # Calculate productivity gap
    df["productivity_gap"] = df["perceived_productivity_score"] - df['actual_productivity_score']
    
    # Use a random sample for better visualization
    sample = df.sample(frac=sample_frac, random_state=random_state)
    actual_scores_sample = df['actual_productivity_score'].loc[sample.index]
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=actual_scores_sample, y=sample["productivity_gap"], alpha=0.6, color="teal")
    
    # Calculate and plot rolling mean for trend line
    sample_sorted = sample.copy()
    sample_sorted["actual_productivity_score"] = actual_scores_sample
    sample_sorted = sample_sorted.sort_values("actual_productivity_score")
    rolling_mean = sample_sorted["productivity_gap"].rolling(window=window, min_periods=1).mean()
    sns.lineplot(
        x=sample_sorted["actual_productivity_score"],
        y=rolling_mean,
        color="red",
        label=f"Rolling Mean (window={window})"
    )
    
    # Set plot labels and title
    plt.title("Productivity Gap vs Actual Productivity Score")
    plt.xlabel("Actual Productivity Score")
    plt.ylabel("Productivity Gap (Perceived - Actual)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def plot_notifications_per_hour(df, wh_perday=1/16, bins=30):
    """
    Create a histogram of notifications per hour with KDE.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing number_of_notifications and work_hours_per_day
    wh_perday (float): Weight for notifications per day calculation (default: 1/16)
    bins (int): Number of bins for histogram (default: 30)
    """
    # Calculate notifications per hour
    df["notifications_per_hour"] = wh_perday * (df["number_of_notifications"] / df["work_hours_per_day"])
    
    # Create histogram with KDE
    sns.histplot(df["notifications_per_hour"], kde=True, bins=bins, color='purple')
    
    # Set plot labels and title
    plt.title("Distribution of Notifications per Hour")
    plt.xlabel("Notifications per Hour")
    plt.ylabel("Frequency")
    plt.show()
    
def plot_binned_notifications(df, bins=[-2, -1, 0, 1, 2], labels=["Very Low", "Low", "Moderate", "High", "Very High"]):
    """
    Create a count plot of binned notifications per hour.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing notifications_per_hour column
    bins (list): Bin edges for notifications per hour (default: [-2, -1, 0, 1, 2, max])
    labels (list): Labels for bins (default: ["Very Low", "Low", "Moderate", "High", "Very High"])
    """
    # Ensure max value is included in bins
    bins = bins + [df["notifications_per_hour"].max()] if bins[-1] != df["notifications_per_hour"].max() else bins
    
    # Bin notifications per hour
    df["notif_bin"] = pd.cut(
        df["notifications_per_hour"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )
    
    # Create count plot
    sns.countplot(x="notif_bin", data=df, palette="viridis", hue="notif_bin")
    
    # Set plot labels and title
    plt.title("Distribution of Notifications per Hour Binned")
    plt.xlabel("Notifications per Hour (Binned)")
    plt.ylabel("Count")
    plt.show()
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
    
    Parameters:
    df (pd.DataFrame): Input DataFrame containing the columns
    focus_col (str): Column name for focus apps (default: 'uses_focus_apps')
    wellbeing_col (str): Column name for digital wellbeing (default: 'has_digital_wellbeing_enabled')
    y_col (str): Target column for y-axis (default: 'daily_social_media_time')
    """
    # Balance classes for focus apps
    class_0_focus = df[df[focus_col] == 0]
    class_1_focus = df[df[focus_col] == 1]
    n_samples_focus = min(len(class_0_focus), len(class_1_focus))
    class_0_bal_focus = resample(class_0_focus, replace=False, n_samples=n_samples_focus, random_state=42)
    class_1_bal_focus = resample(class_1_focus, replace=False, n_samples=n_samples_focus, random_state=42)
    balanced_focus = pd.concat([class_0_bal_focus, class_1_bal_focus])
    
    # Balance classes for digital wellbeing
    class_0_wellbeing = df[df[wellbeing_col] == 0]
    class_1_wellbeing = df[df[wellbeing_col] == 1]
    n_samples_wellbeing = min(len(class_0_wellbeing), len(class_1_wellbeing))
    class_0_bal_wellbeing = resample(class_0_wellbeing, replace=False, n_samples=n_samples_wellbeing, random_state=42)
    class_1_bal_wellbeing = resample(class_1_wellbeing, replace=False, n_samples=n_samples_wellbeing, random_state=42)
    balanced_wellbeing = pd.concat([class_0_bal_wellbeing, class_1_bal_wellbeing])
    
    # Calculate point-biserial correlations
    corr_focus, p_corr_focus = pointbiserialr(balanced_focus[focus_col], balanced_focus[y_col])
    corr_wellbeing, p_corr_wellbeing = pointbiserialr(balanced_wellbeing[wellbeing_col], balanced_wellbeing[y_col])
    
    # Initialize figure
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Focus Apps (balanced)
    plt.subplot(1, 2, 1)
    sns.violinplot(data=balanced_focus, 
                   x=focus_col, 
                   y=y_col,
                   split=True,
                   inner='quart',
                   palette=['steelblue', 'orange'],
                   hue=focus_col)
    plt.title('Daily Social Media Time by Focus App Usage (Balanced)')
    plt.xlabel('Uses Focus Apps')
    plt.ylabel('Daily Social Media Time')
    
    # Annotate means, std deviations, and correlation for Focus Apps
    groups_focus = balanced_focus[focus_col].unique()
    for i, group in enumerate(sorted(groups_focus)):
        group_data = balanced_focus[balanced_focus[focus_col] == group][y_col]
        mean = group_data.mean()
        std = group_data.std()
        plt.text(i, mean + std + 0.2, f"μ={mean:.4f}\nσ={std:.4f}", 
                 ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')
    # Add correlation annotation
    plt.text(0.5, -0.15, f"r={corr_focus:.3f}, p={p_corr_focus:.4f}", 
             ha='center', va='top', fontsize=10, color='black', fontweight='bold', transform=plt.gca().transAxes)
    
    # Plot 2: Digital Wellbeing (balanced)
    plt.subplot(1, 2, 2)
    sns.violinplot(data=balanced_wellbeing, 
                   x=wellbeing_col, 
                   y=y_col,
                   split=True,
                   inner='quart',
                   palette=['steelblue', 'orange'],
                   hue=wellbeing_col)
    plt.title('Daily Social Media Time by Digital Wellbeing (Balanced)')
    plt.xlabel('Digital Wellbeing Enabled')
    plt.ylabel('Daily Social Media Time')
    
    # Annotate means, std deviations, and correlation for Digital Wellbeing
    groups_wellbeing = balanced_wellbeing[wellbeing_col].unique()
    for i, group in enumerate(sorted(groups_wellbeing)):
        group_data = balanced_wellbeing[balanced_wellbeing[wellbeing_col] == group][y_col]
        mean = group_data.mean()
        std = group_data.std()
        plt.text(i, mean + std + 0.2, f"μ={mean:.4f}\nσ={std:.4f}", 
                 ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')
    # Add correlation annotation
    plt.text(0.5, -0.15, f"r={corr_wellbeing:.3f}, p={p_corr_wellbeing:.4f}", 
             ha='center', va='top', fontsize=10, color='black', fontweight='bold', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.show()
def plot_job_type_counts(job_type_counts):
    """
    Create a bar plot of job type counts.
    
    Parameters:
    job_type_counts (pd.Series): Series containing counts of each job type
    """
    plt.figure(figsize=(8, 4))
    sns.barplot(x=job_type_counts.index, y=job_type_counts.values, palette="viridis")
    plt.title("Count of Each Job Type in Training Data")
    plt.xlabel("Job Type")
    plt.ylabel("Count")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.show()
def plot_stress_by_job_type(df, job_type_col='job_type', stress_col='stress_level'):
    """
    Create a box plot of stress level by job type.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing job type and stress level columns
    job_type_col (str): Column name for job type (default: 'job_type')
    stress_col (str): Column name for stress level (default: 'stress_level')
    """
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df[job_type_col], y=df[stress_col], palette="coolwarm", hue=df[job_type_col])
    plt.title("Stress Level by Job Type")
    plt.xlabel("Job Type")
    plt.ylabel("Stress Level")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.show()
def plot_social_platform_pie(df, platform_col='social_platform_preference'):
    """
    Create a pie chart of the most favored social platform.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the social platform preference column
    platform_col (str): Column name for social platform preference (default: 'social_platform_preference')
    """
    # Calculate platform counts
    platform_counts = df[platform_col].value_counts()
    
    # Create pie chart
    plt.figure(figsize=(7, 7))
    plt.pie(platform_counts, labels=platform_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    plt.title('Most Favored Social Platform (Training Set)')
    plt.axis('equal')
    plt.show()