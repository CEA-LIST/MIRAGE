import os
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib.lines import Line2D
from matplotlib import rcParams


def plot_scalar_accuracies_from_df(df, group_by, title="", figsize=(14, 9), save_path=None, dpi=300):
    if 'max_val_accuracy_mean_std' not in df.columns:
        raise ValueError("The dataframe must contain 'max_val_accuracy_mean_std' column.")
    if not all(col in df.columns for col in group_by):
        missing = [col for col in group_by if col not in df.columns]
        raise ValueError(f"The following group_by columns are missing from the dataframe: {missing}")

    has_test_acc = 'test_accuracy_mean_std' in df.columns
    has_train_acc = 'max_accuracy_mean_std' in df.columns

    def format_label(values, max_len=65):
        label = ', '.join(str(v).replace('_', ' ').capitalize() for v in values)
        if len(label) <= max_len:
            return label
        split_idx = max_len
        while split_idx > 0 and label[split_idx] != ' ':
            split_idx -= 1
        if split_idx == 0:
            split_idx = max_len
        label = label[:split_idx].strip() + '\n' + label[split_idx:].strip()
        return label

    point_labels = df[group_by].apply(lambda row: format_label(row.values), axis=1)
    x_positions = np.arange(len(df))
    cmap = cm.get_cmap('tab20', len(df))

    rcParams.update({
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': dpi
    })

    fig, ax = plt.subplots(figsize=figsize)

    # Shape legend (order: train, validation, test)
    shape_legend = [
        Line2D([0], [0], marker='s', color='w', label='Train Accuracy',
               markerfacecolor='black', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Validation Accuracy',
               markerfacecolor='black', markersize=10)
    ]
    if has_test_acc and not df['test_accuracy_mean_std'].isna().all() and (
        df['test_accuracy_mean_std'].str.split(' ± ').str[0].astype(float) > 0).any():
        shape_legend.append(Line2D([0], [0], marker='^', color='w', label='Test Accuracy',
                                   markerfacecolor='black', markersize=10))

    color_handles = {}

    for idx, (x, raw_label) in enumerate(zip(x_positions, point_labels)):
        row = df.iloc[idx]
        color = cmap(idx)
        label = raw_label

        # Parse and plot validation point
        val_acc_mean, val_acc_std = map(float, row['max_val_accuracy_mean_std'].split(' ± '))
        ax.errorbar(x, val_acc_mean, yerr=val_acc_std, fmt='o', color=color, alpha=1, markersize=8,
                    capsize=6, capthick=1.5, linewidth=2)

        # Train accuracy if available
        if has_train_acc and not pd.isna(row['max_accuracy_mean_std']):
            train_acc_mean, train_acc_std = map(float, row['max_accuracy_mean_std'].split(' ± '))
            ax.errorbar(x, train_acc_mean, yerr=train_acc_std, fmt='s', color=color, alpha=0.6, markersize=8,
                        capsize=6, capthick=1.5, linewidth=2)

        # Test accuracy if available
        if has_test_acc and not pd.isna(row['test_accuracy_mean_std']):
            test_acc_mean, test_acc_std = map(float, row['test_accuracy_mean_std'].split(' ± '))
            if test_acc_mean > 0:
                ax.errorbar(x, test_acc_mean, yerr=test_acc_std, fmt='^', color=color, alpha=0.6, markersize=8,
                            capsize=6, capthick=1.5, linewidth=2)

        if label not in color_handles:
            color_handles[label] = Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=color, label=label, markersize=10)

    # Shape legend
    shape_legend_obj = ax.legend(handles=shape_legend, 
                                 title="Accuracy Type", loc='upper left',
                                 bbox_to_anchor=(1.02, 1.0),
                                 title_fontproperties={'weight': 'bold'})
    ax.add_artist(shape_legend_obj)

    # Color legend
    def clean_group_title(group_by):
        return ', '.join(
            word.replace('_', ' ').capitalize() for word in group_by
        )

    ax.legend(
        handles=list(color_handles.values()), 
        title=clean_group_title(group_by),
        loc='upper left',
        bbox_to_anchor=(1.02, 0.8),
        title_fontproperties={'weight': 'bold'}
    )

    ax.set_xticks([])
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Run')
    ax.set_title(title or 'Validation, Train, and Test Accuracies')
    ax.set_xlim(-0.5, len(df) - 0.5)

    # Set Y limits dynamically
    all_vals = []
    for idx in range(len(df)):
        val_mean, val_std = map(float, df.iloc[idx]['max_val_accuracy_mean_std'].split(' ± '))
        val_std = val_std if not pd.isna(val_std) else 0
        all_vals.extend([val_mean + val_std, val_mean - val_std])
        if has_train_acc and not pd.isna(df.iloc[idx]['max_accuracy_mean_std']):
            train_mean, train_std = map(float, df.iloc[idx]['max_accuracy_mean_std'].split(' ± '))
            train_std = train_std if not pd.isna(train_std) else 0
            all_vals.extend([train_mean + train_std, train_mean - train_std])
        if has_test_acc and not pd.isna(df.iloc[idx]['test_accuracy_mean_std']):
            test_mean, test_std = map(float, df.iloc[idx]['test_accuracy_mean_std'].split(' ± '))
            test_std = test_std if not pd.isna(test_std) else 0
            if test_mean > 0:
                all_vals.extend([test_mean + test_std, test_mean - test_std])
    ymin = max(min(all_vals) - 0.02, 0)
    ymax = min(max(all_vals) + 0.02, 1.05)
    ax.set_ylim(ymin, ymax)

    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()


def summarize_tensorboard_logs_with_hparams(folder_path, test_folder_name="test"):
    """
    Summarize TensorBoard logs in a folder and its subfolders, including hyperparameters.

    For each training run, report the max accuracy (acc), validation accuracy (val_acc),
    test accuracy (test_acc), test loss (test_loss), and test top-3 accuracy (test_top3_acc),
    along with the associated loss, validation loss, and hyperparameters.

    Args:
        folder_path (str): Path to the folder containing TensorBoard logs.

    Returns:
        pd.DataFrame: Summary of the TensorBoard logs with hyperparameters.
    """
    summary_data = []

    # Walk through the folder and its subfolders
    for root, _, files in os.walk(folder_path):
        if "test" in root or root.endswith("/test") or "val_clean" in root or "val_noisy" in root:
            continue
        
        for file in files:
            if "events.out.tfevents" in file:  # Identify TensorBoard event files
                file_path = os.path.join(root, file)
                event_acc = EventAccumulator(file_path)
                event_acc.Reload()  # Load the event file

                # Extract scalar data
                scalars = {}
                for tag in event_acc.Tags().get("scalars", []):
                    scalars[tag] = event_acc.Scalars(tag)

                # Initialize summary for this log file
                max_acc = max_val_acc = max_test_acc = max_test_top3_acc = None
                acc_loss = val_acc_loss = test_loss = None
                acc_step = val_acc_step = test_acc_step = None

                # Find max validation accuracy and its step
                if "val_acc" in scalars:
                    max_val_acc_event = max(scalars["val_acc"], key=lambda x: x.value)
                    max_val_acc = max_val_acc_event.value
                    val_acc_step = max_val_acc_event.step

                    # Extract acc and loss at the same step as max val_acc
                    if "acc" in scalars:
                        acc_event = next((x for x in scalars["acc"] if x.step == val_acc_step), None)
                        max_acc = acc_event.value if acc_event else None
                        acc_step = val_acc_step
                    if "loss" in scalars:
                        acc_loss_event = next((x for x in scalars["loss"] if x.step == val_acc_step), None)
                        acc_loss = acc_loss_event.value if acc_loss_event else None
                    if "val_loss" in scalars:
                        val_acc_loss_event = next((x for x in scalars["val_loss"] if x.step == val_acc_step), None)
                        val_acc_loss = val_acc_loss_event.value if val_acc_loss_event else None
                else:
                    # Fallback: use max acc and associated loss as before
                    if "acc" in scalars:
                        max_acc_event = max(scalars["acc"], key=lambda x: x.value)
                        max_acc = max_acc_event.value
                        acc_step = max_acc_event.step
                    if "loss" in scalars and max_acc is not None:
                        acc_loss_event = next((x for x in scalars["loss"] if x.step == acc_step), None)
                        acc_loss = acc_loss_event.value if acc_loss_event else None
                    if "val_loss" in scalars and max_val_acc is not None:
                        val_acc_loss_event = next((x for x in scalars["val_loss"] if x.step == val_acc_step), None)
                        val_acc_loss = val_acc_loss_event.value if val_acc_loss_event else None

                # Check if a "test" subfolder exists
                test_folder = os.path.join(root, test_folder_name)
                if os.path.exists(test_folder):
                    for test_file in os.listdir(test_folder):
                        if "events.out.tfevents" in test_file:
                            test_file_path = os.path.join(test_folder, test_file)
                            test_event_acc = EventAccumulator(test_file_path)
                            test_event_acc.Reload()

                            # Extract test metrics
                            if "test_acc" in test_event_acc.Tags().get("scalars", []):
                                test_acc_values = test_event_acc.Scalars("test_acc")
                                max_test_acc_event = max(test_acc_values, key=lambda x: x.value)
                                max_test_acc = max_test_acc_event.value
                                test_acc_step = max_test_acc_event.step

                            if "test_top3_acc" in test_event_acc.Tags().get("scalars", []):
                                test_top3_acc_values = test_event_acc.Scalars("test_top3_acc")
                                max_test_top3_acc_event = max(test_top3_acc_values, key=lambda x: x.value)
                                max_test_top3_acc = max_test_top3_acc_event.value

                            if "test_loss" in test_event_acc.Tags().get("scalars", []):
                                test_loss_values = test_event_acc.Scalars("test_loss")
                                test_loss_event = next((x for x in test_loss_values if x.step == test_acc_step), None)
                                test_loss = test_loss_event.value if test_loss_event else None

                # Load hyperparameters from hparams.yaml
                hparams = {}
                hparams_file = os.path.join(root, "hparams.yaml")
                if os.path.exists(hparams_file):
                    with open(hparams_file, "r") as f:
                        hparams = yaml.safe_load(f)

                # Append summary for this log file
                summary_entry = {
                    "run": root,
                    "max_accuracy": max_acc,
                    "loss": acc_loss,
                    "max_val_accuracy": max_val_acc,
                    "val_loss": val_acc_loss,
                    "test_accuracy": max_test_acc,
                    "test_loss": test_loss,
                    "test_top3_accuracy": max_test_top3_acc,
                }
                summary_entry.update(hparams)  # Add hyperparameters to the summary
                summary_data.append(summary_entry)

    # Convert summary to DataFrame
    return pd.DataFrame(summary_data)

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def plot_training_curves(log_dict, title, figsize=(15, 6)):
    """
    Plots training, validation, and test accuracy and loss curves for multiple training runs.

    Parameters:
    log_dict (dict): Dictionary where keys are curve names and values are paths to the log directories containing TensorBoard event files.
    title (str): Title for the entire figure.
    figsize (tuple): Size of the figure (default is (15, 6)).

    Returns:
    None
    """
    def extract_metrics(log_dir):
        event_acc = event_accumulator.EventAccumulator(log_dir)
        event_acc.Reload()  # Load the events from the directory

        # Extract train and validation metrics
        train_acc_tag = 'acc'
        train_loss_tag = 'loss'
        val_acc_tag = 'val_acc'
        val_loss_tag = 'val_loss'

        train_acc_values = event_acc.Scalars(train_acc_tag)
        train_loss_values = event_acc.Scalars(train_loss_tag)
        val_acc_values = event_acc.Scalars(val_acc_tag)
        val_loss_values = event_acc.Scalars(val_loss_tag)

        # Separate out the steps and values for all metrics
        train_steps = [entry.step for entry in train_acc_values]
        train_acc = [entry.value for entry in train_acc_values]
        train_loss = [entry.value for entry in train_loss_values]

        val_steps = [entry.step for entry in val_acc_values]
        val_acc = [entry.value for entry in val_acc_values]
        val_loss = [entry.value for entry in val_loss_values]

        # Initialize test metrics
        test_steps, test_acc, test_top3_acc, test_loss = [], [], [], []

        # Check if a "test" subfolder exists
        test_log_dir = os.path.join(log_dir, "test")
        if os.path.exists(test_log_dir):
            test_event_acc = event_accumulator.EventAccumulator(test_log_dir)
            test_event_acc.Reload()

            # Extract test metrics
            test_acc_tag = 'test_acc'
            test_top3_acc_tag = 'test_top3_acc'
            test_loss_tag = 'test_loss'

            if test_acc_tag in test_event_acc.Tags().get("scalars", []):
                test_acc_values = test_event_acc.Scalars(test_acc_tag)
                test_steps = [entry.step for entry in test_acc_values]
                test_acc = [entry.value for entry in test_acc_values]

            if test_top3_acc_tag in test_event_acc.Tags().get("scalars", []):
                test_top3_acc_values = test_event_acc.Scalars(test_top3_acc_tag)
                test_top3_acc = [entry.value for entry in test_top3_acc_values]

            if test_loss_tag in test_event_acc.Tags().get("scalars", []):
                test_loss_values = test_event_acc.Scalars(test_loss_tag)
                test_loss = [entry.value for entry in test_loss_values]

        return train_steps, train_acc, train_loss, val_steps, val_acc, val_loss, test_steps, test_acc, test_top3_acc, test_loss

    # Create subplots: one for accuracy and one for loss
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Iterate through the log dictionary and plot the curves
    for i, (curve_name, log_dir) in enumerate(log_dict.items()):
        # Extract metrics for the current training run
        train_steps, train_acc, train_loss, val_steps, val_acc, val_loss, test_steps, test_acc, test_top3_acc, test_loss = extract_metrics(log_dir)

        # Use a different color for each run
        color = plt.cm.get_cmap('tab10')(i % 10)  # Using a colormap to cycle colors

        # Plot train and validation accuracy
        axes[0].plot(train_steps, train_acc, label=f'{curve_name} Train Acc', linestyle='--', color=color)
        axes[0].plot(val_steps, val_acc, label=f'{curve_name} Val Acc', linestyle='-', color=color)

        # Plot test accuracy if available
        if test_acc:
            axes[0].scatter(test_steps, test_acc, label=f'{curve_name} Test Acc', marker='o', color=color, edgecolor='black', s=100)

        # Plot train and validation loss
        axes[1].plot(train_steps, train_loss, label=f'{curve_name} Train Loss', linestyle='--', color=color)
        axes[1].plot(val_steps, val_loss, label=f'{curve_name} Val Loss', linestyle='-', color=color)

        # Plot test loss if available
        if test_loss:
            axes[1].scatter(test_steps, test_loss, label=f'{curve_name} Test Loss', marker='o', color=color, edgecolor='black', s=100)

        # Plot test top-3 accuracy if available
        if test_top3_acc:
            axes[0].scatter(test_steps, test_top3_acc, label=f'{curve_name} Test Top-3 Acc', marker='^', color=color, edgecolor='black', s=100)

    # Set titles, labels, and legends for the plots
    axes[0].set_xlabel('Steps')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Train, Validation, and Test Accuracy')
    axes[0].legend(fontsize='small')
    axes[0].grid(True)

    axes[1].set_xlabel('Steps')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Train, Validation, and Test Loss')
    axes[1].legend(fontsize='small')
    axes[1].grid(True)

    # Set the common title for the entire figure
    fig.suptitle(title, fontsize=16)

    # Adjust the layout to prevent overlapping
    fig.tight_layout()

    # Show the plot
    plt.show()

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
# import seaborn as sns

def compute_and_plot_param_importance(summary_raw_df, group_keys, target="max_val_accuracy_mean"):
    df = summary_raw_df.copy()

    # Drop rows with missing or inf values
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[target] + group_keys)

    X = df[group_keys].copy()
    y = df[target].values

    # Encode categorical variables
    for col in X.columns:
        if X[col].dtype == "object" or isinstance(X[col].iloc[0], (bool, str, tuple)):
            X[col] = X[col].astype(str)
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Hyperparameter": X.columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    # Plot
    plt.figure(figsize=(8, 5))
    sns.barplot(data=importance_df, x="Importance", y="Hyperparameter", palette="mako", hue="Hyperparameter")
    plt.title("Hyperparameter Importance")
    plt.tight_layout()
    plt.show()

    return importance_df


def process_tensorboard_logs(folder_paths, group_keys, test_folder_name="test", save_path=None):
    """
    Processes TensorBoard logs and generates a summary DataFrame.

    Args:
        folder_paths (list): List of folder paths containing TensorBoard logs.
        group_keys (list): List of keys to group the data.
        save_path (str, optional): Path to save the resulting DataFrame as a CSV file.

    Returns:
        pd.DataFrame: Processed summary DataFrame.
    """
    summary = summarize_and_format_tensorboard_logs(folder_paths, group_keys, test_folder_name)
    summary = filter_and_clean_summary(summary, group_keys)
    summary = sort_summary(summary)
    if save_path:
        save_dataframe_to_csv(summary, save_path)

    return summary

def filter_and_clean_summary(summary, group_keys):
    """
    Filters and cleans the summary DataFrame.

    Args:
        summary (pd.DataFrame): Summary DataFrame.
        group_keys (list): List of keys to group the data.

    Returns:
        pd.DataFrame: Filtered and cleaned DataFrame.
    """

    # List of metrics to remove (mean/std columns)
    metrics = ['max_accuracy', 'loss', 'max_val_accuracy', 'val_loss',
               'test_accuracy', 'test_loss', 'test_top3_accuracy']

    # Build column names to drop, only if they exist
    drop_cols = []
    for metric in metrics:
        for suffix in ['_mean', '_std']:
            col_name = metric + suffix
            if col_name in summary.columns:
                drop_cols.append(col_name)

    summary = summary.drop(columns=drop_cols)

    return summary



def sort_summary(summary):
    """
    Sorts the summary DataFrame.

    Args:
        summary (pd.DataFrame): Summary DataFrame.

    Returns:
        pd.DataFrame: Sorted DataFrame.
    """
    return summary.sort_values(by="max_val_accuracy_mean_std", ascending=False)


def save_dataframe_to_csv(df, save_path):
    """
    Saves a DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame to save.
        save_path (str): Path to save the CSV file.

    Returns:
        None
    """
    df.to_csv(save_path, index=False)


def summarize_and_format_tensorboard_logs(folder_paths, group_keys=[], test_folder_name="test"):
    """
    Summarizes TensorBoard logs from multiple folder paths and formats the results.

    Args:
        folder_paths (list): List of folder paths containing TensorBoard logs.
        group_keys (list): List of columns to group the data by.

    Returns:
        pd.DataFrame: Formatted summary DataFrame with mean and standard deviation for metrics.
    """
    summaries = [summarize_tensorboard_logs_with_hparams(folder, test_folder_name) for folder in folder_paths]
    combined_summary = combine_and_normalize_summaries(summaries)
    # combined_summary.to_csv("combined_summary.csv")

    if not group_keys:
        return combined_summary

    grouped_summary = group_and_aggregate_summary(combined_summary, group_keys)
    grouped_summary = format_mean_std_columns(grouped_summary)

    return grouped_summary


def combine_and_normalize_summaries(summaries):
    """
    Combines multiple summaries into a single DataFrame and normalizes data types.

    Args:
        summaries (list): List of DataFrames containing summaries.

    Returns:
        pd.DataFrame: Combined and normalized summary DataFrame.
    """
    combined_summary = pd.concat(summaries, ignore_index=True)
    combined_summary = combined_summary.fillna(False)
    combined_summary = combined_summary.applymap(
        lambda x: tuple(x) if isinstance(x, list) else (None if pd.isnull(x) else x)
    )
    return combined_summary


def group_and_aggregate_summary(combined_summary, group_keys):
    """
    Groups the summary DataFrame by specified keys and calculates mean and standard deviation.

    Args:
        combined_summary (pd.DataFrame): Combined summary DataFrame.
        group_keys (list): List of columns to group the data by.

    Returns:
        pd.DataFrame: Grouped and aggregated summary DataFrame.
    """
    try:
        metrics = ['max_accuracy', 'loss', 'max_val_accuracy', 'val_loss', 'test_accuracy', 'test_loss', 'test_top3_accuracy']
        existing_metrics = [metric for metric in metrics if metric in combined_summary.columns]

        grouped_summary = (
            combined_summary
            .groupby(group_keys)[existing_metrics]
            .agg(['mean', 'std'])
        )
        grouped_summary.columns = ['_'.join(col) for col in grouped_summary.columns]
        grouped_summary = grouped_summary.reset_index()
        # grouped_summary.to_csv("grouped_summary.csv")
        return grouped_summary
    except Exception as e:
        print(f"An error occurred while grouping and aggregating the summary: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error


def format_mean_std_columns(grouped_summary):
    """
    Formats mean and standard deviation into a single column for each metric.

    Args:
        grouped_summary (pd.DataFrame): Grouped summary DataFrame.

    Returns:
        pd.DataFrame: Updated DataFrame with formatted mean ± std columns.
    """
    def format_mean_std(mean, std):
        return f"{mean:.3f} ± {std:.3f}"

    for metric in ['max_accuracy', 'loss', 'max_val_accuracy', 'val_loss', 'test_accuracy', 'test_loss', 'test_top3_accuracy']:
        if f"{metric}_mean" in grouped_summary.columns and f"{metric}_std" in grouped_summary.columns:
            grouped_summary[f"{metric}_mean_std"] = grouped_summary.apply(
                lambda row: format_mean_std(row[f"{metric}_mean"], row[f"{metric}_std"]),
                axis=1
            )
    return grouped_summary


def adjust_plot_layout_and_show(fig):
    """
    Adjusts the layout of a plot to prevent overlapping and displays it.

    Args:
        fig (matplotlib.figure.Figure): Matplotlib figure object.

    Returns:
        None
    """
    fig.tight_layout()
    plt.show()