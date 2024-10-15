# ML_Zoom-Camp-Hmwk-3
This repository contains the solution to Week 3 homework for the ML Zoomcamp, where the task is to build and train machine learning models using the Bank Marketing Dataset.

Dataset

The dataset used in this homework is the Bank Marketing dataset, which consists of data related to direct marketing campaigns by a Portuguese banking institution. The goal is to predict whether a client will subscribe to a term deposit (y variable).

Download the dataset here.
Extract and use the bank-full.csv file.
Features Used

The following columns from the dataset are used:

age

job

marital

education

balance

housing

contact

day

month

duration

campaign

pdays

previous

poutcome

y (target variable)


Tasks

1. Data Exploration
Analyze the dataset and check for missing values.
Determine the most frequent observation (mode) for the education column.
2. Correlation Analysis
Create a correlation matrix for the numerical features to identify which features are most strongly correlated.
3. Target Encoding
Encode the target variable y (yes/no) as 1/0.
4. Data Splitting
Split the dataset into train, validation, and test sets with a 60%/20%/20% ratio.
5. Mutual Information
Calculate the mutual information score between the target variable y and other categorical features.
6. Logistic Regression Model
Train a logistic regression model using one-hot encoding for the categorical features.
Use the LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42) model.
7. Feature Elimination
Identify the least useful feature using the feature elimination technique by training the model without each feature and recording the accuracy.
8. Regularized Logistic Regression
Train a regularized logistic regression with different values of parameter C and evaluate the accuracy on the validation set.
Instructions

Download and Unzip Dataset:
bash
Copy code
wget https://archive.ics.uci.edu/static/public/222/bank+marketing.zip
unzip bank+marketing.zip
Run the Code: The solution can be found in the ML_Zoom Camp Hmwk 3.py file, where each task is implemented step by step.
Dependencies:
Python 3.x
pandas
scikit-learn
matplotlib
seaborn
