#!/usr/bin/env python
# coding: utf-8

# In[838]:


import pandas as pd
import os
os.getcwd()
os.chdir('/Users/utente/downloads')


# In[839]:


get_ipython().system('wget https://archive.ics.uci.edu/static/public/222/bank+marketing.zip')
    


# In[840]:


import zipfile

# Unzipping the file
with zipfile.ZipFile('bank+marketing.zip', 'r') as zip_ref:
    zip_ref.extractall()
    
# Load the data
data = pd.read_csv('bankfull.csv', sep=';')


# In[841]:


data


# In[842]:


data.columns


# In[843]:


data.tail().T


# # Data preparation

# In[844]:


df = data[['age', 'job', 'marital', 'education', 'balance', 'housing',
       'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
       'previous', 'poutcome', 'y']]


# In[845]:


df.columns


# In[846]:


df['y'] = df['y'].replace({'yes': 1, 'no': 0})


# In[847]:


df.isnull().sum()


# ## Question 1

# What is the most frequent observation (mode) for the column education?
# 
# 
# The most frequesnt observation is secondary

# In[848]:


df['education'].mode()


# ## Question 2

# What are the two features that have the biggest correlation?
# 
# 
# pdays and previous

# In[849]:


df.dtypes


# In[850]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


numerical_columns =df.select_dtypes(include=['int64', 'float64'])

# Compute the correlation matrix
correlation_matrix = numerical_columns.corr()
 
#visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Numerical Features")
plt.show()


# ## Question 3

# Which of these variables has the biggest mutual information score?
# 
# poutcome

# In[851]:


# Loop through each column and print unique values
for column in df.columns:
    print(f"Unique values in '{column}':")
    print(df[column].unique())
    print("\n")


# In[852]:


from sklearn.model_selection import train_test_split
df_full_train, df_test = train_test_split(df, train_size =0.2, random_state = 42)


# In[853]:


len(df_full_train),len(df_test)


# In[854]:


df_val, df_train = train_test_split(df_full_train, train_size =0.25, random_state = 42)


# In[855]:


len(df_train),len(df_val),len( df_test)


# In[856]:


df_train = df_train.reset_index(drop = True)
df_val = df_val.reset_index(drop = True)
df_test = df_test.reset_index(drop = True)


# In[857]:


df_full_train.fillna(0,inplace=True)


# In[858]:


df_full_train.isnull().sum()


# In[859]:


y_train = df_train['y'].values
y_val = df_val['y'].values
y_test = df_test['y'].values





# In[860]:


del df_train['y']
del df_val['y']
del df_test['y']


# In[861]:


df_full_train.reset_index(drop = True)


# In[862]:


df_full_train.y.value_counts()


# In[863]:


Global_subscription_rate = df_full_train.y.mean()
Global_subscription_rate
round(Global_subscription_rate, 2)


# In[864]:


Numerical = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
Categorical = ['job','marital','education','housing','loan', 'contact','month',
       'previous', 'poutcome']


# In[865]:


for c in Categorical:
    df_group = df_full_train.groupby (c).y.agg(['mean','count'])
    df_group['diff'] = df_group['mean'] - Global_subscription_rate
    df_group['risk'] = df_group['mean'] / Global_subscription_rate
    from IPython.display import display
    print(f"\nSummary for {c}:")
    display(df_group)
    
  


# In[866]:


for c in Numerical:
    df_group = df_full_train.groupby (c).y.agg(['mean','count'])
    df_group['diff'] = df_group['mean'] - Global_subscription_rate
    df_group['risk'] = df_group['mean'] / Global_subscription_rate
    from IPython.display import display
    print(f"\nSummary for {c}:")
    display(df_group)
    


# ## Mutual Information

# In[867]:


from sklearn.metrics import mutual_info_score
mutual_info_score(df_full_train.job,df_full_train.y )


# In[868]:


def mutual_info_y_score(series):
    return mutual_info_score(series, df_full_train.y)
mi = df_full_train[Categorical].apply(mutual_info_y_score)
mi.sort_values(ascending = False)


# ## Question 4

# In[869]:


from sklearn.feature_extraction import DictVectorizer
'''train_dict = df_train[Categorical + Numerical].to_dict(orient = 'records')
dv = DictVectorizer(sparse = False)
x_train = dv.fit_transform(train_dict)
x_train'''


# Fit DictVectorizer on training data
train_dict = df_train[Categorical + Numerical].to_dict(orient='records')
dv = DictVectorizer(sparse=False)
x_train = dv.fit_transform(train_dict)

# Transform the validation set using the same DictVectorizer
val_dict = df_val[Categorical + Numerical].to_dict(orient='records')
x_val = dv.transform(val_dict)  


# In[870]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)

model.fit(x_train, y_train)


# In[871]:


y_pred = model.predict(x_val)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.2f}")


# ## Question 5

# In[872]:


# Function to train the model and calculate accuracy
def train_and_evaluate(df_train, df_val, y_train, y_val, features):
    train_dict = df_train[features].to_dict(orient='records')
    val_dict = df_val[features].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    x_train = dv.fit_transform(train_dict)
    x_val = dv.transform(val_dict)

    # Train logistic regression model
    model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
    model.fit(x_train, y_train)

    # Predict on validation set
    y_pred = model.predict(x_val)

    # Calculate accuracy
    return accuracy_score(y_val, y_pred)

# List of all features
Categorical = ['job','marital','education','housing','loan', 'contact','month', 'previous', 'poutcome']
Numerical = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
all_features = Categorical + Numerical

# Base model with all features
base_accuracy = train_and_evaluate(df_train, df_val, y_train, y_val, all_features)
print(f"Base model accuracy with all features: {base_accuracy:.4f}")

# Features to evaluate
features_to_evaluate = ['age', 'balance', 'marital', 'previous']

# Store results
results = {}

# Iterate through each feature and exclude it
for feature in features_to_evaluate:
    features_subset = [f for f in all_features if f != feature]
    accuracy = train_and_evaluate(df_train, df_val, y_train, y_val, features_subset)
    accuracy_diff = base_accuracy - accuracy
    results[feature] = accuracy_diff
    print(f"Accuracy without {feature}: {accuracy:.4f} (Difference: {accuracy_diff:.4f})")

# Find the feature with the smallest difference
least_useful_feature = min(results, key=results.get)
print(f"\nFeature with the smallest difference in accuracy: {least_useful_feature}")


# ## Question 6

# In[873]:


# Function to train and evaluate the model for a given value of C
def train_and_evaluate_regularized(df_train, df_val, y_train, y_val, features, C_value):
    train_dict = df_train[features].to_dict(orient='records')
    val_dict = df_val[features].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    x_train = dv.fit_transform(train_dict)
    x_val = dv.transform(val_dict)

    # Train logistic regression model with regularization parameter C
    model = LogisticRegression(solver='liblinear', C=C_value, max_iter=1000, random_state=42)
    model.fit(x_train, y_train)

    # Predict on validation set
    y_pred = model.predict(x_val)

    # Calculate accuracy
    return accuracy_score(y_val, y_pred)

# List of all features (same as in Q4)
Categorical = ['job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'previous', 'poutcome']
Numerical = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
all_features = Categorical + Numerical

# List of C values to try
C_values = [0.01, 0.1, 1, 10, 100]

# Store results
best_accuracy = 0
best_C = None
results = {}

# Iterate over each C value
for C in C_values:
    accuracy = train_and_evaluate_regularized(df_train, df_val, y_train, y_val, all_features, C)
    accuracy_rounded = round(accuracy, 3)
    results[C] = accuracy_rounded
    print(f"Accuracy with C={C}: {accuracy_rounded}")

    # Track the best accuracy and corresponding C
    if accuracy_rounded > best_accuracy:
        best_accuracy = accuracy_rounded
        best_C = C

print(f"\nBest C value: {best_C} with accuracy: {best_accuracy:.3f}")


# In[ ]:




