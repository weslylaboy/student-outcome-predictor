# Problem Statement

## Research Questions
This project will use machine learning to answer two questions that academic advisors face every day:

1. Which currently enrolled students are most likely to drop out, and how can we identify them early enough to intervene?  
2. For students flagged as dropout risk, which factors are causing their low academic performance, and where should advisors focus their support to give those students the best chance of succeeding?

## Dataset
The dataset is available at the UC Irvine Machine Learning Repository:  
[https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

## Techniques to be used
- **Data preparation**  
- **Exploratory Data Analysis (EDA)** with visualizations to understand the data before modeling  
- **Feature encoding and scaling**: Encode categorical features and scale numeric features  
- **Handle class imbalance** between “Dropout” and “Graduate” students using SMOTE or class weight adjustment  
- **Train/test split**: 80/20 train-test split with 5-fold stratified cross-validation  

## Classification Model
**Goal:** Predict whether a student will *Dropout* or *Graduate*

### Algorithms
- Logistic Regression (baseline)  
- Decision Tree  
- Random Forest  
- K-Nearest Neighbors (KNN)  
- Support Vector Machines (SVM)  

**Training:** The model will be trained using data from students with known outcomes (Graduate or Dropout).

### Evaluation
- Accuracy  
- F1-score  
- Confusion matrix  
- ROC-AUC  
- Recall  

> **Note:** Recall will be prioritized since missing a dropout (False Negative) is worse than flagging a student as at risk when it is not the case (False Positive).

**Deployment:**  
The model will be used on currently enrolled students to flag those who are at risk.

## Regression Model
**Goal:** For students flagged as at risk of Dropout, predict their GPA and identify what is causing the students low performance.

### Algorithms
- Linear Regression (baseline)  
- Ridge/Lasso  
- Random Forest Regressor  
- XGBoost Regressor  

### Feature Importance
Identify which factors such as financial stress, academic struggles, or personal background are most associated with low GPA among students at risk of Dropout.

### Evaluation
- Root Mean Square Error (RMSE)  
- Mean Absolute Error (MAE)  
- R² (Coefficient of determination)  

## Expected Results
The **classification model** is expected to correctly identify students who are at risk of dropping out.  
It will be evaluated using accuracy, F1-score, and ROC-AUC, but **Recall** will be prioritized for the *Dropout* class. Recall measures how many of the at-risk students the model successfully identified. Missing a student who needed help and never got it (False Negative) is worse than offering help to a student who is doing fine (False Positive). Therefore, the model will be tuned to maximize Recall, even if that means flagging a student unnecessarily. Once trained, it will be applied to currently enrolled students to identify those needing immediate intervention.

The **regression model** will then be applied to students flagged as at risk by the classification model.  
It will predict each student’s GPA and highlight the factors contributing most to their low performance. The most important part of the output will not be the exact GPA value but the feature importance results. By showing whether a student’s predicted GPA is mainly affected by financial stress, such as tuition debt or lack of scholarship, by academic struggles, such as few courses passed, or by personal background, the model will give advisors a starting point for deciding what kind of support to offer to each student.

Together, these models aim to:  
- Help advisors identify which students are at risk of dropping out  
- Clarify *why* they are struggling  
- Suggest where support efforts should be focused  

## Why are these questions important?
Most universities only find out a student is struggling too late, after something has already gone wrong, maybe a failed course, a missed payment, or when the students stop showing up to class, in other words, the students have already dropped out. Too often, advisors are trying to help students already in crisis rather than identifying problems early and assigning resources to help them before it is too late. With this project, I’ll use available data to build a model that identifies early warning signs that a student may be struggling. The classification model will tell advisors which students need help right now while the regression model will tell advisors what kind of help the students are most likely to need. Together, these models will give institutions an chance to help students before it’s too late.

Ultimately, not only will students benefit by getting the help they really need, but institutions will also benefit, since dropout rates significantly affect universities through financial losses, reputation damage, and additional operational costs.

