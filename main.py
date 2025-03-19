from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from typing import Literal, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
import constants as const



def accuracy(trisomy_type: Literal[const.trisomy_type_18, const.trisomy_type_21, const.trisomy_type_13],
             true_positives: int,
             false_positives: int,
             true_negatives: int,
             false_negatives: int):
    prior = const.trisomy_prior[trisomy_type]
    actual_positives = true_positives + false_negatives
    actual_negatives = true_negatives + false_positives
    negative_rate = true_negatives / actual_negatives
    positive_rate = true_positives / actual_positives
    return negative_rate * (1 - prior) + positive_rate * prior


def balanced_accuracy(true_positives,
                      false_positives,
                      true_negatives,
                      false_negatives):
    actual_positives = true_positives + false_negatives
    actual_negatives = true_negatives + false_positives
    negative_rate = true_negatives / actual_negatives
    positive_rate = true_positives / actual_positives
    return (negative_rate + positive_rate) / 2


def precision(true_positives: int,
              false_positives: int,
              true_negatives: int,
              false_negatives: int):
    return true_positives / (true_positives + false_positives)


def recall(true_positives: int,
           false_positives: int,
           true_negatives: int,
           false_negatives: int):
    return true_positives / (true_positives + false_negatives)


def metrics(trisomy_type, true_positives, false_positives, true_negatives, false_negatives):
    return {
        'accuracy': accuracy(trisomy_type, true_positives, false_positives, true_negatives, false_negatives),
        'balanced accuracy': balanced_accuracy(true_positives, false_positives, true_negatives, false_negatives),
        'precision': precision(true_positives, false_positives, true_negatives, false_negatives),
        'recall': recall(true_positives, false_positives, true_negatives, false_negatives),
        'f1': f1_score(true_positives, false_positives, true_negatives, false_negatives)
    }


def predict(data, model):
    return data.swifter.apply(lambda row: model.predict(row), axis=1)


def score(data, model):
    tqdm.pandas()
    # Split the entire data into batches of 1000 rows

    # Process each batch
    scores = data.progress_apply(lambda row: model.score(row), axis=1)
    data['bayesian score'] = scores.values  # Update the bayesian score for the batch
    data.to_csv('data.csv')
    print("All batches processed and saved.")
def plot_roc_curve(data, likelihood_ratio, path, week, trisomy_type, model):
    plt.rcParams['font.family'] = 'Georgia'
    plt.rcParams['font.size'] = 11  # Set the default font size

    coverage_levels = sorted(data[const.feature_coverage_depth].unique())
    for i, depth in enumerate(coverage_levels):
        condition = (data[const.feature_coverage_depth] == depth)
        # Extract likelihood ratios and labels
        depth_likelihood_ratio = likelihood_ratio[condition].values
        depth_likelihood_ratio = np.clip(depth_likelihood_ratio, -1e15, 1e14)
        labels = data.loc[condition, const.feature_trisomy_label].values

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(labels, depth_likelihood_ratio)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve for this coverage depth
        plt.plot(fpr, tpr, label=f"{round(depth, 2)}x (AUC = {roc_auc:.2f})",linewidth=2)
    # Add diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess", linewidth=1)

    # Labels and title
    plt.ylabel("True Positive Rate (TPR)")
    plt.xlabel("False Positive Rate (FPR)")
    plt.title(f"ROC Curves, gestational age: {week} weeks, T{trisomy_type}, {model} model", fontsize=14)
    plt.legend(loc="lower right",
               fontsize=9,
               ncol=2,
               columnspacing=0.1,  # Reduces spacing between columns
               handletextpad=0.05,  # Reduces space between the marker and text
               )
    plt.savefig(path)
    plt.close()

def plot_roc(data, lr, gestational_age, trisomy_type, model_type, path: Optional[str] = None):
    # Rename variable to avoid conflict with built-in `filter`
    condition = ((data[const.feature_age] == gestational_age) &
                 (data[const.feature_trisomy_type] == trisomy_type))

    actual_lr = lr[condition]
    if path is None:
        path = f'plots/roc_curves/{model_type}.week_{gestational_age}.chr_{trisomy_type}.png'
    plot_roc_curve(data[condition], actual_lr, path, gestational_age, trisomy_type, model_type)


def fixed_accuracy(trisomy_type, true_positives, false_positives, true_negatives, false_negatives):
    """Compute Fixed Accuracy using the trisomy prior."""
    prior = const.trisomy_prior[trisomy_type]
    actual_positives = true_positives + false_negatives
    actual_negatives = true_negatives + false_positives
    negative_rate = true_negatives / actual_negatives
    positive_rate = true_positives / actual_positives
    return negative_rate * (1 - prior) + positive_rate * prior

# ========================== Likelihood Ratio Calculation ========================== #
def compute_lrt(likelihood_ratio: float) -> float:
    """Compute the LRT test statistic (positive log) and p-value."""
    D = 2 * np.log(likelihood_ratio)  # Positive log version
    p_value = chi2.sf(D, df=1)       # Compute p-value from chi-square distribution
    return p_value


def apply_lrt(data: pd.DataFrame) -> pd.DataFrame:
    """Apply LRT p-value calculation to Bayesian and deterministic models."""
    data['bayesian_p_value'] = data['bayesian score'].apply(compute_lrt)
    data['deterministic_p_value'] = data['regular_lr'].apply(compute_lrt)
    return data


# ========================== Prediction Logic ========================== #
def make_predictions(data: pd.DataFrame, alpha: float = 0.05) -> tuple:
    """Generate predictions for both Bayesian and deterministic models."""
    bayesian_predictions = data['bayesian_p_value'] < alpha
    deterministic_predictions = data['deterministic_p_value'] < alpha
    return bayesian_predictions, deterministic_predictions


# ========================== Metric Calculation ========================== #
def compute_metrics(y_true, y_pred, trisomy_type):
    """Compute various performance metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'True Negatives (TN)': tn,
        'False Positives (FP)': fp,
        'False Negatives (FN)': fn,
        'True Positives (TP)': tp,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
        'Fixed Accuracy': fixed_accuracy(trisomy_type, tp, fp, tn, fn),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred)
    }


# ========================== Confusion Matrix Visualization ========================== #
def plot_confusion_matrix(conf_matrix, metrics, title):
    conf_matrix = conf_matrix.astype('float') / conf_matrix.sum()
    """Plot confusion matrix with TP, TN, FP, FN labels and evaluation metrics."""
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap="Blues",
                     xticklabels=["Non-Trisomy (Predicted)", "Trisomy (Predicted)"],
                     yticklabels=["Non-Trisomy (Actual)", "Trisomy (Actual)"])

    # Get counts for each category
    TN, FP, FN, TP = conf_matrix.ravel()

    # Define quadrant labels
    labels = np.array([
        [f"TN: {TN}", f"FP: {FP}"],
        [f"FN: {FN}", f"TP: {TP}"]
    ])
    # Set proper tick positions
    ax.set_xticks([])
    ax.set_yticks([])

    # Define quadrant labels next to the axes
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)

    plt.title(title)
    plt.savefig(f"{title}.png")

    plt.show()


# ========================== Main Workflow ========================== #
def confusion_matrix_pipline(data, week, depth, trisomy_type):
    for coverage_depth in data[const.feature_coverage_depth].unique():
        filtered_data = data[
            (data[const.feature_age] == 7) &
            (data[const.feature_coverage_depth] == coverage_depth) &
            (data[const.feature_trisomy_type] == 21)
        ]

        # Get true labels
        y_true = filtered_data[const.feature_trisomy_label]

        # Apply LRT and make predictions
        filtered_data = apply_lrt(filtered_data)
        bayesian_predictions, deterministic_predictions = make_predictions(filtered_data)

        # Compute confusion matrices
        conf_matrix_bayesian = confusion_matrix(y_true, bayesian_predictions)
        conf_matrix_deterministic = confusion_matrix(y_true, deterministic_predictions)

        plot_confusion_matrix(conf_matrix_bayesian, None,
                              f"Bayesian Model, Weeks 7, Coverage Depth {round(coverage_depth, 2)}, T21")


        # Save updated dataset with LRT p-values
        print("\n✅ **LRT-based classification completed. Results and metrics displayed.**")


if __name__ == '__main__':
    data = pd.read_csv('data.csv')
    print(data.columns)
    confusion_matrix_pipline(data, 10, 0.2, 21)



