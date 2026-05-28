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


def predict_with_threshold(model, X_test, y_test, threshold=0.4, model_name="Model"):
    """
    Predicts labels based on a custom probability threshold and evaluates the model.

    Parameters:
    - model: The trained scikit-learn model or pipeline.
    - X_test: Test features.
    - y_test: True labels.
    - threshold: Probability threshold for the positive class (default 0.5).
    - model_name: String for reporting/labeling.
    """
    # Get probabilities for all classes
    # predict_proba returns [prob_class_0, prob_class_1]
    probs = model.predict_proba(X_test)

    # Assuming class 0 is 'Dropout' and class 1 is 'Graduate'
    # We want to flag 'Dropout' (0) if the probability of 'Dropout' is > (1 - threshold)
    # Or simply: if probability of being class 0 is high, label as 0.

    # In your code snippet: probs[:, 0] is the probability of class 0 (Dropout).
    # If prob of Dropout > 0.6 (or if you prefer the Graduate threshold logic):
    # Let's keep it consistent with your thresholding logic:
    dropout_probs = probs[:, 0]

    # Apply custom threshold for class 0 (Dropout)
    # e.g., if dropout_prob > 0.6 (which is equivalent to graduate_prob <= 0.4)
    # The snippet you had used: (probs <= 0.4).astype(int)
    # Let's generalize this:
    y_pred_custom = (dropout_probs >= (1 - threshold)).astype(int)

    # Evaluate
    evaluate_classification_model(
        y_test, y_pred_custom, model_name=f"{model_name} (Threshold {threshold})"
    )

    return y_pred_custom