import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, ConfusionMatrixDisplay, accuracy_score, precision_score,
                             recall_score, f1_score)


def fix_datatypes(df):
    """
    Converts specific columns to int8 for memory efficiency.
    """
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()

    int_cols = [
        'displaced', 'special_needs', 'debtor', 'tuition_fees_up_to_date',
        'gender', 'scholarship_holder', 'international', 'daytime_evening',
        'age_at_enrollment', 'cu1_credited', 'cu1_enrolled', 'cu1_evaluations',
        'cu1_approved', 'cu1_no_evaluations', 'cu2_credited', 'cu2_enrolled',
        'cu2_evaluations', 'cu2_approved', 'cu2_no_evaluations', 'total_approved_units'
    ]

    unknown_flags = [
        'father_occupation_is_unknown', 'mother_occupation_is_unknown',
        'father_qualification_is_unknown', 'mother_qualification_is_unknown',
        'prev_qualification_is_unknown', 'any_info_missing'
    ]

    # Convert columns if they exist in the dataframe
    for cols in [int_cols, unknown_flags]:
        existing_cols = [c for c in cols if c in df.columns]
        df[existing_cols] = df[existing_cols].astype('int8')

    if 'target_encoded' in df.columns:
        df['target_encoded'] = df['target_encoded'].astype('int8')

    return df


def evaluate_classification_model(y_test, y_pred, model_name="Model", target_names=['Dropout', 'Graduate']):
    """
    Prints a classification report and displays a confusion matrix.
    """
    print(f"{model_name}: evaluation results")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Create the display
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=target_names,
        cmap='Blues',
        text_kw={'color': 'black', 'fontweight': 'bold', 'fontsize': 14}
    )

    plt.title(f'{model_name} Confusion Matrix (Recall Focus)')
    plt.grid(False)
    plt.show()


def get_model_metrics(y_test, y_pred, model_name):
    """Extract classification metrics and return as a dictionary (one row)."""
    print("hello")
    return {
        'Model': model_name,
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'Precision (Dropout)': round(precision_score(y_test, y_pred, pos_label=0), 4),
        'Recall (Dropout)': round(recall_score(y_test, y_pred, pos_label=0), 4),
        'F1-Score (Dropout)': round(f1_score(y_test, y_pred, pos_label=0), 4),
        'Precision (Graduate)': round(precision_score(y_test, y_pred, pos_label=1), 4),
        'Recall (Graduate)': round(recall_score(y_test, y_pred, pos_label=1), 4),
        'F1-Score (Graduate)': round(f1_score(y_test, y_pred, pos_label=1), 4),
    }