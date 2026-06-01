# Students Outcome Predictor

**Author:** Wesly Laboy

---

## Executive summary
This project is about building a system to find students who might `Dropout` and understand why they are struggling academically. 
Applying EDA to data from 4,424 students, I saw that how many classes they pass (academic momentum) and if they have money problems 
(tuition status and scholarships) are the biggest signs of success. The EDA showed that students who don't pass any classes 
in their first year or owe tuition money are at very high risk. I want to use these findings to help advisors reach out 
to students early, before they decide to `Dropout`.

---

## Why should anyone care about this question?

Most universities only notice a student is in trouble when it’s too late, after they already failed a class or stopped showing up. 
This is bad for the student and the school. If we can spot the "red flags" early, like a sudden drop in grades or missing a payment, 
the university can help the student fix the problem before it gets too big to handle.

---

## Rationale

This project uses machine learning to answer two questions that academic advisors face every day:

1. Which currently enrolled students are most likely to drop out, and how can we identify them early enough to intervene?  
2. For students flagged as dropout risk, which factors are causing their low academic performance, and where should advisors focus their support to give those students the best chance of succeeding?

---

## Data Source

The dataset is available at the UC Irvine Machine Learning Repository:  
[https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

---

#### Methodology
1. **Cleaning the Data:** I checked for missing info and fixed some columns. For example, I made a special "flag" for students who didn't provide certain info, as this can sometimes be a sign of risk.
2. **Looking for Patterns (EDA):** I made charts to see how things like age, debt, and grades affect whether a student stays or leaves.
3. **Creating New Features:** I combined some data to get better insights, like:
   - `total_approved_units`: Adding up all classes passed in the first year.
   - `grade_progression`: Checking if grades went up or down between semesters.
   - `age_group`: Grouping students by age to see which groups struggle the most.
4. **Baseline Model:** I started with a simple model (Logistic Regression) to get a basic idea of how well we can predict dropouts. I focused on "Recall," which means we want to make sure I don't miss any students who actually need help.

---

## Algorithms

- Logistic Regression (baseline)
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machines (SVM)
- XGBoost
- Neural Network (MLP)

---

## Results

***From EDA:***
*   **Passing Classes is key:** This is the biggest sign. Graduates usually pass **10–12 classes** in their first year, while most dropouts pass **0 classes**.
*   **Money is important:** Students who owe tuition or have debt drop out much more often. On the other hand, **76% of students with scholarships graduate**, compared to only 41% of those without one.
*   **The Age Factor:** The dropout rate peaks for students between 26 and 30 years old.
*   **Missing Info:** If a student has missing administrative info, they are twice as likely to drop out. It might mean they aren't very "connected" to the school.

***From Baseline Model (Logistic Regression):***
The baseline model achieved an overall **Accuracy of 88%**, establishing a strong starting point for predictions.
*   **Recall (Dropout): 0.82** – The model successfully caught 82% of all actual dropouts.
*   **Precision (Dropout): 0.86** – When the model flags a student at risk, it is right 86% of the time.
*   **Confusion Matrix Analysis:** The model correctly identified **234 True Positives** (dropouts caught) while missing 50 students (False Negatives). Reducing the False Negatives is the primary goal for future, more advanced models.
*   **Technical Finding:** Using `class_weight='balanced'` was critical to account for the smaller number of dropout entries, ensuring the model take into consideration class imbalance.

---

## Outline of project

- [Data Cleaning Notebook](./notebooks/01_data_cleaning.ipynb): Data cleaning process and preparation of the data.
- [EDA Notebook](./notebooks/02_exploratory_data_analysis.ipynb): The charts, insights, and our first basic model.
- [Classification Model](notebooks/03_classification_model.ipynb): The baseline model and its evaluation.
- [Final Dataset](./data/processed/02.7_data_final.csv): The clean dataset we used for our analysis.

---

## Classification Model Report

### The Problem I Was Trying to Solve

Universities have a hard time knowing which students are going to dropout. By the time a student stops showing up, it is usually too late to help them. In the first part of this project, I tried to answer one question: **can we look at a student's information right now and tell if they are likely to leave before finishing?**

The model was used on students who were already enrolled and had not yet graduated or dropped out, 794 students in total. The model was trained on students whose outcomes we already know (Graduate or Dropout), and then we used it to make predictions for the enrolled group.

### How I Built the Model

I trained 23 different versions of models, using 7 different methods:

| Algorithm | Configurations Tested |
|---|---|
| Logistic Regression | Baseline, Tuned, with custom threshold |
| Decision Tree | Baseline, Tuned, with SMOTE |
| K-Nearest Neighbors | Tuned, with SMOTE |
| Support Vector Machine | Baseline, Tuned, with SMOTE |
| Random Forest | Baseline, Tuned (GridSearch), Tuned (RandomizedSearch), with SMOTE |
| XGBoost | Baseline, with SMOTE, with RandomizedSearch |
| Neural Network | Baseline, Tuned |

For each model, I measured how many at risk students it correctly caught. This is called **Recall for the Dropout class**, the main metric I used to compare models.

I chose Recall as the priority because **missing a student who is going to drop out is worse than flagging a student who is actually going to graduate**. It is better to offer help to someone who does not need it than to ignore someone who does.

---

### Why I Used a Custom Threshold

By default, most models say "Dropout" only when they are more than 50% sure. I changed this so that the model flags a student as at risk when it is more than 40% sure. This makes the model more careful, and it catches more students who need help, even if it also flags a few extra students who would have graduated anyway.

---

### The Model I Picked

After comparing all 23 versions, I chose the **XGBoost Baseline model with a threshold of 0.40**.

Here is how it performed on the test group (726 students the model had never seen before):

| Result | Number of Students |
|---|---|
| Correctly identified Dropout students | 245 out of 284 |
| Dropout students missed | 39 |
| Graduate students flagged incorrectly | 21 |
| Correctly identified Graduate students | 421 out of 442 |

This model found the most at risk students while having the fewest false alarms. It had a **Dropout Recall of 0.86** and an overall accuracy of **90%**.

---


### What the Model Learned

After training, I looked at which pieces of information had the biggest effect on predictions. The results fell into four clear groups:

**Academic Performance (36% of total influence)**
- The single most important signal is how many courses a student passed in their second semester (`cu2_approved`). This one feature alone accounted for 27% of the model's decisions.
- Total courses approved across both semesters (`total_approved_units`) was also very important.
- Students who are falling behind in class are the easiest to identify as at-risk.

**Course / Program Type (30% of total influence)**
- Several specific programs appeared repeatedly in the top features.
- This means dropout risk is different across all departments. Some programs have noticeably higher risk patterns than others.
- This is useful for institutions, it shows where to focus program level support.

**Financial Situation (15% of total influence)**
- Whether a student has their tuition paid up to date (`tuition_fees_up_to_date`) was the second most important single feature.
- Scholarship status and whether a student has debt also showed up clearly.
- Financial stress is a real and measurable warning sign.

**Demographics (3% of total influence)**
- Age at enrollment had a small but measurable effect.
- Overall, what a student does, how many courses they pass, whether they pay tuition is more predictive than who they are.

---
### What I Found When I Applied the Model to Enrolled Students

I loaded the saved model and ran it on all 794 currently enrolled students. These students were never seen during training, the model had no idea about their outcomes when it was trained.

The results:

| Prediction | Number of Students | Percentage |
|---|---|---|
| Likely Graduate | 403 | 50.8% |
| Dropout Risk | 391 | 49.2% |

**391 students were flagged as Dropout Risk.**

This is a high number, but it makes sense for two reasons:
1. The threshold is set at 0.40, which makes the model more aggressive in catching students at risk.
2. Many enrolled students may show early warning signs, low course approvals, overdue tuition, that look similar to patterns seen in students who eventually dropped out.

This list of 391 students is **not a verdict**. It is a starting point for advisors to begin conversations to check in, ask questions, and offer support before it is too late.

---

### What Institutions Can Do With This

Based on the model results, we suggest the following actions:

- **Reach out early** — Contact the 391 flagged students before they fall behind further. A simple checkin conversation can make a difference.
- **Focus academic support** — Prioritize students with low second semester course approvals. This is the strongest signal the model found.
- **Focus financial support** — Look at students with overdue tuition or no scholarship. Financial stress is a clear and manageable risk factor.
- **Look at high-risk programs** — Certain programs appeared repeatedly in the feature importance results. Advisors in those departments should be especially proactive.
- **Use the model regularly** — Running this prediction at the start of each semester allows institutions to catch new at risk students before problems get worse.

---


##### Contact and Further Information
If you have any questions about this work, feel free to reach out to Wesly Laboy.
