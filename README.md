### Students Outcome Predictor

**Author:** Wesly Laboy

#### Executive summary
This project is about building a system to find students who might `Dropout` and understand why they are struggling academically. 
Applying EDA to data from 4,424 students, I saw that how many classes they pass (academic momentum) and if they have money problems 
(tuition status and scholarships) are the biggest signs of success. The EDA showed that students who don't pass any classes 
in their first year or owe tuition money are at very high risk. I want to use these findings to help advisors reach out 
to students early, before they decide to `Dropout`.

#### Rationale
Why should anyone care about this question?

Most universities only notice a student is in trouble when it’s too late, after they already failed a class or stopped showing up. 
This is bad for the student and the school. If we can spot the "red flags" early, like a sudden drop in grades or missing a payment, 
the university can help the student fix the problem before it gets too big to handle.

#### Research Question
This project will use machine learning to answer two questions that academic advisors face every day:

1. Which currently enrolled students are most likely to drop out, and how can we identify them early enough to intervene?  
2. For students flagged as dropout risk, which factors are causing their low academic performance, and where should advisors focus their support to give those students the best chance of succeeding?

#### Data Sources
The dataset is available at the UC Irvine Machine Learning Repository:  
[https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

#### Methodology
1. **Cleaning the Data:** I checked for missing info and fixed some columns. For example, I made a special "flag" for students who didn't provide certain info, as this can sometimes be a sign of risk.
2. **Looking for Patterns (EDA):** I made charts to see how things like age, debt, and grades affect whether a student stays or leaves.
3. **Creating New Features:** I combined some data to get better insights, like:
   - `total_approved_units`: Adding up all classes passed in the first year.
   - `grade_progression`: Checking if grades went up or down between semesters.
   - `age_group`: Grouping students by age to see which groups struggle the most.
4. **Baseline Model:** I started with a simple model (Logistic Regression) to get a basic idea of how well we can predict dropouts. I focused on "Recall," which means we want to make sure I don't miss any students who actually need help.


#### Algorithms
- Logistic Regression (baseline)  
- Decision Tree  
- Random Forest  
- K-Nearest Neighbors (KNN)  
- Support Vector Machines (SVM)  

#### Results
- From EDA:
*   **Passing Classes is key:** This is the biggest sign. Graduates usually pass **10–12 classes** in their first year, while most dropouts pass **0 classes**.
*   **Money is important:** Students who owe tuition or have debt drop out much more often. On the other hand, **76% of students with scholarships graduate**, compared to only 41% of those without one.
*   **The Age Factor:** The dropout rate peaks for students between 26 and 30 years old.
*   **Missing Info:** If a student has missing administrative info, they are twice as likely to drop out. It might mean they aren't very "connected" to the school.

#### Next steps
1. **Better Models:** I want to try more advanced algorithm to get more accurate predictions.
2. **Multiple model:** I plan to build a multiple model that predicts a student's GPA and tells advisors the main reason for their struggle.
3. **Fixing the Balance:** Since there are fewer dropouts than graduates in the data, I will use techniques to make sure the model learns about both groups equally.


#### Outline of project

- [Data Cleaning Notebook](./notebooks/01_data_cleaning.ipynb): Data cleaning process and preparation of the data.
- [EDA Notebook](./notebooks/02_exploratory_data_analysis.ipynb): The charts, insights, and our first basic model.
- [Final Dataset](./data/processed/02.7_data_final.csv): The clean dataset we used for our analysis.


##### Contact and Further Information
If you have any questions about this work, feel free to reach out to Wesly Laboy.
