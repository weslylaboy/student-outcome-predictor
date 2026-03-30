Here is the updated markdown table where the output descriptions have been merged into the **Output File** column for a cleaner, more compact layout:

| Order | Notebook | Input File | Output File (Description) | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **1** | `01_data_cleaning.ipynb` | `data_raw.csv` | `data_clean.csv` - Full dataset with engineered features.<br>`X_processed.csv` - Independent variables (predictors) only.<br>`y_class.csv` - Binary target labels (Dropout/Graduate).<br>`y_gpa.csv` - Numeric target values (`avg_grade`). | Technical cleaning and feature engineering. |
| **2** | `02_eda.ipynb` | `data_clean.csv` | `Visualizations/Insights` - Charts and summary of findings. | Understanding student behavior and correlations. |
| **3** | `03_modeling_classification.ipynb` | `X_processed.csv`, `y_class.csv` | `model_classifier.joblib` - Trained classification model object. | Predicting student dropout risk. |
| **4** | `04_modeling_regression.ipynb` | `X_processed.csv`, `y_gpa.csv` | `model_regressor.joblib` - Trained regression model object. | Predicting GPA and identifying risk factors. |