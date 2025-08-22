#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# #Environment setup and installations#

# In[2]:


# Record the start time
import time
start_time = time.time()


get_ipython().system('pip install pandas openpyxl scikit-learn')
get_ipython().system('pip install graphviz pydotplus')
get_ipython().system('pip install matplotlib seaborn')
# !pip install keras
# !pip install tensorflow
get_ipython().system('pip install shap')
get_ipython().system('pip install lime')
get_ipython().system('pip install pdpbox')
get_ipython().system('pip install xgboost')
get_ipython().system('pip install scikit-optimize')
get_ipython().system('pip install keras==2.12.0')
# !pip uninstall tensorflow
get_ipython().system('pip install tensorflow==2.12.0')
get_ipython().system('pip install --upgrade nbconvert notebook')
get_ipython().system('pip show tensorflow')
get_ipython().system('pip install ipython-autotime')
get_ipython().system('pip install jax jaxlib')
get_ipython().run_line_magic('load_ext', 'autotime')


# In[3]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
import graphviz
import pydotplus
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image, display
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, explained_variance_score
from sklearn.inspection import permutation_importance
import shap
from sklearn.inspection import partial_dependence
from pdpbox import pdp
from lime.lime_tabular import LimeTabularExplainer
from joblib import Parallel, delayed
import tensorflow as tf
import xgboost as xgb
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.linear_model import LinearRegression
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
# from keras.optimizers import Adam, RMSprop
from keras.layers import LeakyReLU
from keras.layers import ReLU
import joblib
from joblib import Parallel, delayed


# In[4]:


pd.set_option('display.max_columns', None)  # To display all columns
pd.set_option('display.max_rows', None)     # To display all rows
pd.set_option('display.max_colwidth', None) # To prevent truncation of column contents


# #Exploratory data analysis#
# *Data Cleaning and Preprocessing*
# 

# In[5]:


# Get the name of the uploaded file
file_name = "/content/drive/MyDrive/>5GB/data_science/manali_master_thesis/parameter_analysis/Cura_Parameter_Rawdata_v0.03.xlsx"
df = pd.read_excel(file_name, sheet_name='Sheet1')
df.head()


# In[6]:


df.head()


# In[7]:


# df.dtypes


# In[8]:


df.shape


# In[9]:


df.describe()


# In[10]:


print(df.isnull().sum()[df.isnull().sum() > 0])


# In[11]:


# unique_values = {col: df[col].unique() for col in df.columns}
# unique_counts = {col: df[col].nunique() for col in df.columns}

# # Display the unique values and counts for each column
# for col in df.columns:
#     print(f"Column: {col}")
#     print(f"Unique Values: {unique_values[col]}")
#     print(f"Number of Unique Values: {unique_counts[col]}")
#     print("-" * 40)


# In[12]:


# df.nunique()


# In[13]:


unique_counts_series=df.nunique()
unique_counts_series[unique_counts_series == 1].index


# In[14]:


# Print the duplicate rows
# df[df.duplicated()]


# In[15]:


df[df.duplicated()].shape


# In[16]:


df.shape


# In[17]:


df_unique=df.drop_duplicates()
df_unique.shape


# In[18]:


df_unique = df_unique.reset_index(drop=True)


# In[19]:


# df_unique.tail()


# In[20]:


df1=df_unique.copy()
# df1.head()


# In[21]:


df2=df_unique.copy()
# df2.head()


# #Experiment 1#
# Modeling using Random forest with target variable as numerical data

# In[22]:


# Separate features and target
X = df1.drop('ET-mm', axis=1)
y = df1['ET-mm']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing pipelines for both numerical and categorical data
numerical_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Create and evaluate the pipeline
model_1 = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Not using the stratified split because target variable has
# more unique classes
# sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# for train_index, test_index in sss.split(X, y):
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]


# Train the model
model_1.fit(X_train, y_train)

# Predict on the test data
y_pred = model_1.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)


# In[23]:


print(y.value_counts())


# In[24]:


y.unique()


# In[25]:


value_counts = y.value_counts()
# Plot the distribution
plt.figure(figsize=(10, 6))
sns.barplot(x=value_counts.index, y=value_counts.values)
plt.xlabel('Target Classes')
plt.ylabel('Count')
plt.title('Distribution of Target Variable')
plt.show()


# In[26]:


value_counts.shape


# In[27]:


# Access the trained model
model_trained = model_1.named_steps['classifier']

# Get feature importances
importances = model_trained.feature_importances_

# Get feature names from the preprocessor
feature_names = model_1.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out()
all_feature_names = list(numerical_cols) + list(feature_names)

# Create a DataFrame to display feature importances
importances_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("Feature Importances:")
print(importances_df)


# In[28]:


# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(40, 30))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'], linewidths=2)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[29]:


# Perform cross-validation
cv_scores = cross_val_score(model_1, X, y, cv=5)

print(f'Cross-validation scores: {cv_scores}')
print(f'Average cross-validation score: {cv_scores.mean()}')


# In[30]:


# Training accuracy
train_accuracy = model_1.score(X_train, y_train)
print(f'Training Accuracy: {train_accuracy}')

# Test accuracy
test_accuracy = model_1.score(X_test, y_test)
print(f'Test Accuracy: {test_accuracy}')


# In[31]:


# Define hyperparameters to search
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20, 30]
}

# Grid search
grid_search = GridSearchCV(model_1, param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best Score: {grid_search.best_score_}')


# In[32]:


# Extract a single decision tree from the Random Forest
# Note: Index `0` refers to the first tree; you can choose another index if you prefer
tree = model_1.named_steps['classifier'].estimators_[0]

# Get the number of classes from the fitted tree
num_classes = len(tree.classes_)

# Define class names based on the number of classes
class_names = [f'Class {i}' for i in range(num_classes)]

# Export the tree to a DOT file
dot_data = export_graphviz(tree, out_file=None,
                           feature_names=model_1.named_steps['preprocessor'].get_feature_names_out(),
                           class_names=class_names,
                           filled=True, rounded=True,
                           special_characters=True)

# Create a Graphviz source object
graph = pydotplus.graph_from_dot_data(dot_data)

# Render the tree
graph.write_png('random_forest_classification_tree_experiment_1.png')

# Display the tree
with open('random_forest_classification_tree_experiment_1.png', 'rb') as f:
    display(Image(f.read()))


# In[33]:


# y_test


# In[34]:


y_pred


# In[35]:


y.sort_values().unique()


# In[36]:


# y.value_counts().sort_index()


# Converting target variable 'ET-mm' to categorical for bagging based algorithm modeling such as random forest and decision tree.

# In[37]:


y_sorted_unique = np.sort(np.unique(y))
y_sorted_unique


# In[38]:


num_bins = 16
bin_edges = np.linspace(0, len(y_sorted_unique)-1, num_bins + 1).astype(int)

# Get the actual bin edges from the sorted unique values
bins = np.append(y_sorted_unique[bin_edges], 99999)

# Generate the labels
labels = [f'{bins[i]}_to_{bins[i+1]}' for i in range(len(bins)-2)] + [f'{bins[-2]}_to_99999']

# Print the bins and labels
print("Bins:", bins)
print("Labels:", labels)


# In[39]:


df2.head()


# In[40]:


df2.shape


# In[41]:


df2['ET-mm_range'] = pd.cut(df2['ET-mm'], bins=bins, labels=labels, right=False)


# In[42]:


df2.head()


# In[43]:


df2.shape


# In[44]:


df2['ET-mm_range'].value_counts()


# In[45]:


df2[['ET-mm','ET-mm_range']]


# In[46]:


# [['ET-mm','ET-mm_range']]
df2=df2.drop('ET-mm', axis=1)
df2.shape


# In[47]:


df2.head()


# #Experiment 2#
# Modeling using Random forest with target variable as categorical data

# In[48]:


# Separate features and target
X = df2.drop('ET-mm_range', axis=1)
y = df2['ET-mm_range']
# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing pipelines for both numerical and categorical data
numerical_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Create and evaluate the pipeline
model_2 = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Not using the stratified split because target variable has
# more unique classes
# sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# for train_index, test_index in sss.split(X, y):
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]


# Train the model
model_2.fit(X_train, y_train)

# Predict on the test data
y_pred = model_2.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)


# In[49]:


# Access the trained model
model_trained = model_2.named_steps['classifier']

# Get feature importances
importances = model_trained.feature_importances_

# Get feature names from the preprocessor
feature_names = model_2.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out()
all_feature_names = list(numerical_cols) + list(feature_names)

# Create a DataFrame to display feature importances
importances_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("Feature Importances:")
print(importances_df)


# In[50]:


# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(20, 15))  # Adjust the size as needed
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, linewidths=2)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[51]:


# Perform cross-validation
cv_scores = cross_val_score(model_2, X, y, cv=5)

print(f'Cross-validation scores: {cv_scores}')
print(f'Average cross-validation score: {cv_scores.mean()}')


# In[52]:


# Training accuracy
train_accuracy = model_2.score(X_train, y_train)
print(f'Training Accuracy: {train_accuracy}')

# Test accuracy
test_accuracy = model_2.score(X_test, y_test)
print(f'Test Accuracy: {test_accuracy}')


# In[53]:


# Define hyperparameters to search
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20, 30]
}

# Grid search
grid_search = GridSearchCV(model_2, param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best Score: {grid_search.best_score_}')


# In[54]:


# Extract a single decision tree from the Random Forest
# Note: Index `0` refers to the first tree; you can choose another index if you prefer
tree = model_2.named_steps['classifier'].estimators_[0]

# Get the number of classes from the fitted tree
num_classes = len(tree.classes_)

# Define class names based on the number of classes
# class_names = [f'Class {i}' for i in range(num_classes)]

# Export the tree to a DOT file
dot_data = export_graphviz(tree, out_file=None,
                           feature_names=model_2.named_steps['preprocessor'].get_feature_names_out(),
                           class_names=labels,
                           filled=True, rounded=True,
                           special_characters=True)

# Create a Graphviz source object
graph = pydotplus.graph_from_dot_data(dot_data)

# Render the tree
graph.write_png('random_forest_classification_tree_experiment_2.png')

# Display the tree
with open('random_forest_classification_tree_experiment_2.png', 'rb') as f:
    display(Image(f.read()))


# #Experiment 3#
# Modeling using Decision tree with target variable as categorical data

# In[55]:


# Assuming df2 is your DataFrame
# Separate features and target
X = df2.drop('ET-mm_range', axis=1)
y = df2['ET-mm_range']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing pipelines for both numerical and categorical data
numerical_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Create and evaluate the pipeline
model_3 = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', DecisionTreeClassifier(random_state=42))])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model_3.fit(X_train, y_train)

# Predict on the test data
y_pred = model_3.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

# Access the trained model
model_trained = model_3.named_steps['classifier']

# Feature importances are not as meaningful for a single decision tree as for an ensemble, but we can still display them
importances = model_trained.feature_importances_

# Get feature names from the preprocessor
feature_names = model_3.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out()
all_feature_names = list(numerical_cols) + list(feature_names)

# Create a DataFrame to display feature importances
importances_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("Feature Importances:")
print(importances_df)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Define labels for the confusion matrix
labels = ['2_to_13', '13_to_28', '28_to_150', '150_to_174', '174_to_219', '219_to_301', '301_to_391', '391_to_455', '455_to_465', '465_to_481', '481_to_506', '506_to_557', '557_to_859', '859_to_1049', '1049_to_1094', '1094_to_2734', '2734_to_99999']

# Plot confusion matrix
plt.figure(figsize=(20, 15))  # Adjust the size as needed
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, linewidths=2)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Perform cross-validation
cv_scores = cross_val_score(model_3, X, y, cv=5)

print(f'Cross-validation scores: {cv_scores}')
print(f'Average cross-validation score: {cv_scores.mean()}')

# Training accuracy
train_accuracy = model_3.score(X_train, y_train)
print(f'Training Accuracy: {train_accuracy}')

# Test accuracy
test_accuracy = model_3.score(X_test, y_test)
print(f'Test Accuracy: {test_accuracy}')

# Define hyperparameters to search
param_grid = {
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10]
}

# Grid search
grid_search = GridSearchCV(model_3, param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best Score: {grid_search.best_score_}')

# Extract the best decision tree from the grid search
best_tree = grid_search.best_estimator_.named_steps['classifier']

# Export the tree to a DOT file
dot_data = export_graphviz(best_tree, out_file=None,
                           feature_names=model_3.named_steps['preprocessor'].get_feature_names_out(),
                           class_names=labels,
                           filled=True, rounded=True,
                           special_characters=True)

# Create a Graphviz source object
graph = pydotplus.graph_from_dot_data(dot_data)

# Render the tree
graph.write_png('decision_tree.png')

# Display the tree
with open('decision_tree.png', 'rb') as f:
    display(Image(f.read()))


# #Experiment 4#
# Modeling using **XGBoost** with target variable as numerical data

# In[56]:


df4=df1.copy()
df4 = pd.get_dummies(df4, drop_first=True)
df4 = df4.astype(float)

X = df4.drop('ET-mm', axis=1)  # Features
y = df4['ET-mm']  # Target

# Split the data into training+validation and testing sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split the training+validation set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Create an XGBoost regressor
model_7 = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6)

# Train the model
model_7.fit(X_train, y_train)

# Make predictions
y_pred_train = model_7.predict(X_train)
y_pred_val = model_7.predict(X_val)
y_pred_test = model_7.predict(X_test)

# Evaluate the model
# Training data
mse_train = mean_squared_error(y_train, y_pred_train)
rmse_train = np.sqrt(mse_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

# Validation data
mse_val = mean_squared_error(y_val, y_pred_val)
rmse_val = np.sqrt(mse_val)
mae_val = mean_absolute_error(y_val, y_pred_val)
r2_val = r2_score(y_val, y_pred_val)

# Test data
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print(f"Training Mean Squared Error: {mse_train}")
print(f"Training Root Mean Squared Error: {rmse_train}")
print(f"Training Mean Absolute Error: {mae_train}")
print(f"Training R² Score: {r2_train}")

print(f"Validation Mean Squared Error: {mse_val}")
print(f"Validation Root Mean Squared Error: {rmse_val}")
print(f"Validation Mean Absolute Error: {mae_val}")
print(f"Validation R² Score: {r2_val}")

print(f"Test Mean Squared Error: {mse_test}")
print(f"Test Root Mean Squared Error: {rmse_test}")
print(f"Test Mean Absolute Error: {mae_test}")
print(f"Test R² Score: {r2_test}")

# Plot feature importance
xgb.plot_importance(model_7)
plt.show()


# #Explainable AI of Experiment 4#

# In[57]:


# Compute SHAP values
explainer = shap.Explainer(model_7)
shap_values = explainer(X_train)

# Plot feature importance
shap.summary_plot(shap_values, X_train)


# In[58]:


# Convert feature names to a list
feature_names = X_train.columns.tolist()

# Plot a decision plot
shap.decision_plot(
    base_value=explainer.expected_value,
    shap_values=shap_values.values,
    features=X_train,
    feature_names=feature_names
)


# In[59]:


# Generate a force plot for a specific instance
instance_idx = 0  # Choose an index of the instance you want to visualize
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[instance_idx].values, X_train.iloc[instance_idx])


# In[59]:





# In[59]:





# In[59]:





# #Experiment 5#
# Modeling using **XGBoost** with target variable as numerical data along with bayesian optimization hyperparameter tuning technique

# In[60]:


df4=df1.copy()
df4 = pd.get_dummies(df4, drop_first=True)
df4 = df4.astype(float)

# Load your data
X = df4.drop('ET-mm', axis=1)  # Features
y = df4['ET-mm']  # Target

# Split the data into training and testing sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# Define the model
model_8 = xgb.XGBRegressor(objective='reg:squarederror')

# Define the parameter search space
param_space = {
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'max_depth': Integer(3, 10),
    'n_estimators': Integer(50, 300),
    'subsample': Real(0.6, 1.0),
    'colsample_bytree': Real(0.6, 1.0),
    'gamma': Real(0, 10),
    'reg_alpha': Real(0, 10),
    'reg_lambda': Real(0, 10)
}

# Define the BayesSearchCV
opt = BayesSearchCV(
    estimator=model_8,
    search_spaces=param_space,
    n_iter=50,  # Number of parameter settings sampled
    scoring='neg_mean_squared_error',
    cv=3,  # 3-fold cross-validation
    n_jobs=-1,
    random_state=42
)

# Perform the search
opt.fit(X_train, y_train)

# Best parameters found
print("Best parameters found: ", opt.best_params_)

# Train the final model with the best parameters
best_model = opt.best_estimator_
best_model.fit(X_train, y_train)

# Evaluate the model
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return mse, rmse, mae, r2

mse_train, rmse_train, mae_train, r2_train = evaluate_model(best_model, X_train, y_train)
mse_val, rmse_val, mae_val, r2_val = evaluate_model(best_model, X_val, y_val)
mse_test, rmse_test, mae_test, r2_test = evaluate_model(best_model, X_test, y_test)

print(f"Training Mean Squared Error: {mse_train}")
print(f"Training Root Mean Squared Error: {rmse_train}")
print(f"Training Mean Absolute Error: {mae_train}")
print(f"Training R² Score: {r2_train}")

print(f"Validation Mean Squared Error: {mse_val}")
print(f"Validation Root Mean Squared Error: {rmse_val}")
print(f"Validation Mean Absolute Error: {mae_val}")
print(f"Validation R² Score: {r2_val}")

print(f"Test Mean Squared Error: {mse_test}")
print(f"Test Root Mean Squared Error: {rmse_test}")
print(f"Test Mean Absolute Error: {mae_test}")
print(f"Test R² Score: {r2_test}")


# In[61]:


plt.figure(figsize=(14, 10))  # Adjust the width and height as needed
xgb.plot_importance(best_model, importance_type='weight', title='Feature Importance', xlabel='Importance', ylabel='Features')
plt.yticks(fontsize=5)
plt.show()


# #Explainable AI of Experiment 5#

# In[62]:


# Compute SHAP values using TreeExplainer
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_train)

# Plot summary plot
shap.summary_plot(shap_values, X_train)


# In[63]:


# Convert feature names to a list
feature_names = X_train.columns.tolist()

# Plot a decision plot
shap.decision_plot(
    base_value=explainer.expected_value,
    shap_values=shap_values,
    features=X_train,
    feature_names=feature_names
)


# In[64]:


# Generate a force plot for a specific instance
instance_idx = 0  # Choose an index of the instance you want to visualize
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[instance_idx], X_train.iloc[instance_idx])


# In[64]:





# In[64]:





# In[64]:





# In[64]:





# #Experiment 6#
# Modeling using ANN with target variable as numerical data
# 
# Dense(64,relu) -> Dense(32,relu) -> Dense(1)

# In[65]:


df3=df1.copy()


# In[66]:


df3.shape


# In[67]:


# df3.isnull().sum()


# In[68]:


has_null_values = df3.isnull().values.any()

if has_null_values:
    print("There are null values in the dataframe.")
else:
    print("There are no null values in the dataframe.")


# In[69]:


# Identify categorical columns
categorical_columns = df3.select_dtypes(include=['object', 'category']).columns

# Print unique values for each categorical column
for column in categorical_columns:
    unique_values = df1[column].unique()
    print(f"Unique values in '{column}': {unique_values}")


# In[70]:


# categorical_columns = df3.select_dtypes(include=['object', 'category']).columns

# # Count the columns that have more than 2 unique values
# count_more_than_two = 0

# for column in categorical_columns:
#     unique_values = df3[column].unique()
#     # print(f"Unique values in '{column}': {unique_values}")
#     if len(unique_values) > 2:
#         count_more_than_two += 1

# print(f"Number of categorical columns with more than 2 unique values: {count_more_than_two}")


# In[71]:


# Identify categorical columns
categorical_columns = df3.select_dtypes(include=['object', 'category']).columns

# Print columns with more than 2 unique values
print("Columns with more than 2 unique values:")
for column in categorical_columns:
    unique_values = df3[column].unique()
    if len(unique_values) > 2:
        print(f"{column}: {unique_values}")


# In[71]:





# In[72]:


# df4.shape


# In[72]:





# In[73]:


# df4.dtypes


# In[74]:


df4=df1.copy()
df4 = pd.get_dummies(df4, drop_first=True)
df4 = df4.astype(float)
scaler = StandardScaler()
numerical_cols = df4.select_dtypes(include=['float64', 'int64']).columns
df4[numerical_cols] = scaler.fit_transform(df4[numerical_cols])

X = df4.drop('ET-mm', axis=1)  # Replace 'ET-mm' with your actual target column name
y = df4['ET-mm']

# Split the original data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define and compile the model (use your actual model definition here)
model_4 = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])
model_4.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model and save the history
history = model_4.fit(X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_val, y_val),  # Pass validation data
                    verbose=1)

# Evaluate the model
y_pred = model_4.predict(X_test).flatten()

# Compute Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

# Print training, validation, and test metrics
train_loss, train_mae = model_4.evaluate(X_train, y_train, verbose=0)
val_loss, val_mae = model_4.evaluate(X_val, y_val, verbose=0)
test_loss, test_mae = model_4.evaluate(X_test, y_test)

print(f"Training Mean Absolute Error: {train_mae}")
print(f"Validation Mean Absolute Error: {val_mae}")
print(f"Test Mean Absolute Error: {test_mae}")

# Plot training and validation loss
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 6))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()


# In[75]:


# Predict on the test set
y_pred = model_4.predict(X_test).flatten()

# Compute Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Compute Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Compute R² Score
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")


# In[76]:


# Compute training accuracy
train_loss, train_mae = model_4.evaluate(X_train, y_train, verbose=0)
print(f"Training Mean Absolute Error: {train_mae}")

# Compute validation accuracy
val_loss, val_mae = model_4.evaluate(X_val, y_val, verbose=0)
print(f"Validation Mean Absolute Error: {val_mae}")

# Compute test accuracy
test_loss, test_mae = model_4.evaluate(X_test, y_test)
print(f"Test Mean Absolute Error: {test_mae}")

# Since your model is using MAE, convert it to accuracy
train_accuracy = 1 - train_mae
val_accuracy = 1 - val_mae
test_accuracy = 1 - test_mae

print(f"Training Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")
print(f"Test Accuracy: {test_accuracy}")


# #Experiment 7#
# Modeling using ANN with target variable as numerical data
# 
# Dense(512,relu)-> Dense(256,relu)-> Dense(128,relu)-> Dense(64,relu) -> Dense(32,relu) -> Dense(1)

# In[76]:





# In[77]:


df4=df1.copy()
df4 = pd.get_dummies(df4, drop_first=True)
df4 = df4.astype(float)
scaler = StandardScaler()
numerical_cols = df4.select_dtypes(include=['float64', 'int64']).columns
df4[numerical_cols] = scaler.fit_transform(df4[numerical_cols])

X = df4.drop('ET-mm', axis=1)  # Replace 'ET-mm' with your actual target column name
y = df4['ET-mm']

# Split the original data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define and compile the model (use your actual model definition here)
model_5 = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])
model_5.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model and save the history
history = model_5.fit(X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_val, y_val),  # Pass validation data
                    verbose=1)

# Evaluate the model
y_pred = model_5.predict(X_test).flatten()

# Compute Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

# Print training, validation, and test metrics
train_loss, train_mae = model_5.evaluate(X_train, y_train, verbose=0)
val_loss, val_mae = model_5.evaluate(X_val, y_val, verbose=0)
test_loss, test_mae = model_5.evaluate(X_test, y_test)

print(f"Training Mean Absolute Error: {train_mae}")
print(f"Validation Mean Absolute Error: {val_mae}")
print(f"Test Mean Absolute Error: {test_mae}")

# Plot training and validation loss
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 6))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()


# In[78]:


# Predict on the test set
y_pred = model_5.predict(X_test).flatten()

# Compute Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Compute Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Compute R² Score
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")


# In[79]:


# Compute training accuracy
train_loss, train_mae = model_5.evaluate(X_train, y_train, verbose=0)
print(f"Training Mean Absolute Error: {train_mae}")

# Compute validation accuracy
val_loss, val_mae = model_5.evaluate(X_val, y_val, verbose=0)
print(f"Validation Mean Absolute Error: {val_mae}")

# Compute test accuracy
test_loss, test_mae = model_5.evaluate(X_test, y_test)
print(f"Test Mean Absolute Error: {test_mae}")

# Since your model is using MAE, convert it to accuracy
train_accuracy = 1 - train_mae
val_accuracy = 1 - val_mae
test_accuracy = 1 - test_mae

print(f"Training Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")
print(f"Test Accuracy: {test_accuracy}")


# In[80]:


# !cat /proc/cpuinfo
# !cat /proc/meminfo


# #Experiment 8#
# Modeling using ANN with target variable as numerical data
# 
# Dense(256,relu)-> Dense(128,relu)-> Dense(64,relu) -> Dense(32,relu) -> Dense(1)
# 

# In[81]:


df1.shape


# In[82]:


df4=df1.copy()
df4 = pd.get_dummies(df4, drop_first=True)
df4 = df4.astype(float)
scaler = StandardScaler()
numerical_cols = df4.select_dtypes(include=['float64', 'int64']).columns
df4[numerical_cols] = scaler.fit_transform(df4[numerical_cols])

X = df4.drop('ET-mm', axis=1)  # Replace 'ET-mm' with your actual target column name
y = df4['ET-mm']

# Split the original data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define and compile the model (use your actual model definition here)
model_6 = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])
model_6.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model and save the history
history = model_6.fit(X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_val, y_val),  # Pass validation data
                    verbose=1)

# Evaluate the model
y_pred = model_6.predict(X_test).flatten()

# Compute Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

# Print training, validation, and test metrics
train_loss, train_mae = model_6.evaluate(X_train, y_train, verbose=0)
val_loss, val_mae = model_6.evaluate(X_val, y_val, verbose=0)
test_loss, test_mae = model_6.evaluate(X_test, y_test)

print(f"Training Mean Absolute Error: {train_mae}")
print(f"Validation Mean Absolute Error: {val_mae}")
print(f"Test Mean Absolute Error: {test_mae}")

# Plot training and validation loss
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 6))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()


# Predict on the test set
y_pred = model_6.predict(X_test).flatten()

# Compute Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Compute Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Compute R² Score
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")


# Compute training accuracy
train_loss, train_mae = model_6.evaluate(X_train, y_train, verbose=0)
print(f"Training Mean Absolute Error: {train_mae}")

# Compute validation accuracy
val_loss, val_mae = model_6.evaluate(X_val, y_val, verbose=0)
print(f"Validation Mean Absolute Error: {val_mae}")

# Compute test accuracy
test_loss, test_mae = model_6.evaluate(X_test, y_test)
print(f"Test Mean Absolute Error: {test_mae}")

# Since your model is using MAE, convert it to accuracy
train_accuracy = 1 - train_mae
val_accuracy = 1 - val_mae
test_accuracy = 1 - test_mae

print(f"Training Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")
print(f"Test Accuracy: {test_accuracy}")


# In[83]:


print(X_train.shape)
X_train[:1]


# #Explainable AI of Experiment 8#

# In[84]:


n=10 # number of rows to consider in an array
# Initialize the SHAP explainer for the neural network model
explainer = shap.KernelExplainer(model_6.predict, X_train[:n])  # Use a subset of training data for initialization

# Compute SHAP values for the test set
shap_values = explainer.shap_values(X_test[:n])  # Use a subset of test data for SHAP values

# Create a sample DataFrame with the same columns as your data for visualization
X_test_sample = pd.DataFrame(X_test[:n], columns=df4.drop('ET-mm', axis=1).columns)

# Choose a specific row to visualize
row = 0  # You can change this to any row index you want to visualize

# Extract the SHAP values for the specific row
single_shap_values = shap_values[row]

# Extract base value and convert it to a numpy scalar if needed
base_value = explainer.expected_value
if isinstance(base_value, list) or isinstance(base_value, np.ndarray):
    base_value = base_value[0]  # Use the first element if it's a list or array
if tf.is_tensor(base_value):
    base_value = base_value.numpy()  # Convert tensor to numpy array

# Ensure that we are selecting a single explanation
# Check the shape of single_shap_values
if single_shap_values.ndim > 1:
    single_shap_values = single_shap_values[:, 0]  # Select the first column if necessary


# In[85]:


# Plot the waterfall chart for the specific row
shap.initjs()
explanation=shap.Explanation(values=single_shap_values, base_values=base_value, data=X_test_sample.iloc[row, :])
shap.plots.waterfall(explanation, max_display=10)


# In[86]:


# Optionally, plot a summary plot
shap.summary_plot(shap_values, X_test_sample)


# In[87]:


# Optionally, plot a force plot for the specific row
shap.initjs()
shap.force_plot(base_value, single_shap_values, X_test_sample.iloc[row, :])


# In[87]:





# In[87]:





# #Validating real 3D printer experimental data with model#

# In[88]:


# read sheet1 into pandas dataframe
real_data_file="/content/drive/MyDrive/>5GB/data_science/manali_master_thesis/parameter_analysis/real_experiments_data_for_validation.xlsx"
df_real=pd.read_excel(real_data_file,sheet_name="Sheet1")
df_real.head()


# In[89]:


df_real.shape


# In[90]:


# do a linear regression model for the variables in 2 columns
X = df_real['Print duration in seconds'].values.reshape(-1, 1)
y = df_real['Energy consumption in Wh'].values
model_linear = LinearRegression()
model_linear.fit(X, y)
y_pred = model_linear.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)
print("Weight:", model_linear.coef_[0])
print("Bias:", model_linear.intercept_)

plt.scatter(X, y, label='Actual')
plt.plot(X, y_pred, color='red', label='Predicted')
plt.xlabel('Print duration in seconds')
plt.ylabel('Energy consumption in Wh')
plt.legend()
plt.show()


# Formulas
# 
# Convert watt-hours to kilowatt-hours: Energy in kWh= Energy in Wh / 1000
# 
# Convert joules to kilowatt-hours: Energy in kWh=Energy in J/(3.6×10^6)
# 
# Calculate CO2 emissions: CO2 emissions (g)=Energy in kWh x Carbon Intensity (gCO2/kWh)
# 
# Carbon Intensity (gCO2/kWh) = 300

# In[90]:





# In[91]:


def predict_energy_consumption_in_sec(test_duration_in_sec):
    """
    Predict the energy consumption and CO2 emissions based on the test_duration.

    Parameters:
    - test_duration_in_sec: Print duration in seconds.

    Returns:
    - A dictionary with energy consumption and CO2 emissions values.
    """

    # Ensure test_duration_in_sec is a float
    test_duration_in_sec = float(test_duration_in_sec)

    # Reshape to a 2D array as the model expects
    test_duration_array = np.array([[test_duration_in_sec]])

    # Predict the energy consumption
    predicted_energy_Wh = model_linear.predict(test_duration_array)

    energy_in_Wh = predicted_energy_Wh[0]
    energy_in_kWh = predicted_energy_Wh[0] / 1000
    energy_in_J = predicted_energy_Wh[0] * 3.6e3
    energy_in_kJ = predicted_energy_Wh[0] * 3.6
    carbon_emissions_in_grams = energy_in_kWh * 300

    # Print the predicted energy consumption
    print("Predicted Energy Consumption (Wh):", round(energy_in_Wh, 3))
    print("Predicted Energy Consumption (kWh):", round(energy_in_kWh, 3))
    print("Predicted Energy Consumption (Joules):", round(energy_in_J, 3))
    print("Predicted Energy Consumption (Kilo Joules):", round(energy_in_kJ, 3))
    print("Predicted CO2 Emissions (grams):", round(carbon_emissions_in_grams, 3))

    return {
        'Energy_Wh': round(energy_in_Wh, 3),
        'Energy_kWh': round(energy_in_kWh, 3),
        'Energy_J': round(energy_in_J, 3),
        'Energy_kJ': round(energy_in_kJ, 3),
        'CO2_emissions_grams': round(carbon_emissions_in_grams, 3)
    }

def predict_energy_consumption_in_min(test_duration_in_min):
    """
    Predict the energy consumption and CO2 emissions based on the test_duration.

    Parameters:
    - test_duration_in_min: Print duration in minutes.

    Returns:
    - A dictionary with energy consumption and CO2 emissions values.
    """

    # Ensure test_duration_in_min is a float
    test_duration_in_min = float(test_duration_in_min)

    # Convert minutes to seconds
    test_duration_in_sec = test_duration_in_min * 60

    # Reshape to a 2D array as the model expects
    test_duration_array = np.array([[test_duration_in_sec]])

    # Predict the energy consumption
    predicted_energy_Wh = model_linear.predict(test_duration_array)

    energy_in_Wh = predicted_energy_Wh[0]
    energy_in_kWh = predicted_energy_Wh[0] / 1000
    energy_in_J = predicted_energy_Wh[0] * 3.6e3
    energy_in_kJ = predicted_energy_Wh[0] * 3.6
    carbon_emissions_in_grams = energy_in_kWh * 300

    # Print the predicted energy consumption
    print("Predicted Energy Consumption (Wh):", round(energy_in_Wh, 3))
    print("Predicted Energy Consumption (kWh):", round(energy_in_kWh, 3))
    print("Predicted Energy Consumption (Joules):", round(energy_in_J, 3))
    print("Predicted Energy Consumption (Kilo Joules):", round(energy_in_kJ, 3))
    print("Predicted CO2 Emissions (grams):", round(carbon_emissions_in_grams, 3))

    return {
        'Energy_Wh': round(energy_in_Wh, 3),
        'Energy_kWh': round(energy_in_kWh, 3),
        'Energy_J': round(energy_in_J, 3),
        'Energy_kJ': round(energy_in_kJ, 3),
        'CO2_emissions_grams': round(carbon_emissions_in_grams, 3)
    }

# Example usage
# Input the print duration you want to test
# test_duration = float(input("Enter the print duration in seconds: "))
test_duration = 1800
results = predict_energy_consumption_in_sec(test_duration)
results = predict_energy_consumption_in_min(test_duration/60)


# Preprocessing the real time data from printer

# In[92]:


df1_real=df_real.copy()
df1_real.shape


# In[93]:


# df1_real.columns.tolist()


# In[94]:


df2_real=df1_real.copy()
df2_real['ET-mm']=df1_real['Estimated time (mm)']
# df2_real.columns.tolist()


# In[95]:


df3_real=df2_real.copy()
df3_real.shape
# List of columns to drop
columns_to_drop = [
    'CO2 emissions (grams)',
    'End Time (hh:mm)',
    'Energy consumption in Joules',
    'Energy consumption in Wh',
    'Estimated time (mm)',
    'Experiment No',
    'L-m',
    'Print duration in seconds',
    'Print Time (mm:ss)',
    'Printing Start Time (hh:mm:ss)',
    'Skirt/Brim Flow',
    'Support Infill Speed',
    'Support Line Width',
    'Support Speed',
    'Wt-g'
]

# Drop the specified columns from df3_real
df4_real = df3_real.drop(columns=columns_to_drop, errors='ignore')
df4_real.shape


# In[96]:


print(df4.shape)
print(df4_real.shape)


# In[97]:


df1.head()


# In[98]:


df4.head()


# In[99]:


# Compare columns in df5 and df4_real
same_columns = set(df4.columns) == set(df4_real.columns)

# Output the result
if same_columns:
    print("The columns in df4 and df4_real are the same.")
else:
    print("The columns in df4 and df4_real are different.")


# In[100]:


# Identify columns unique to each DataFrame
columns_in_df4_not_in_df4_real = set(df4.columns) - set(df4_real.columns)
columns_in_df4_real_not_in_df4 = set(df4_real.columns) - set(df4.columns)

# Print the differences
if columns_in_df4_not_in_df4_real:
    print("Columns in df4 but not in df4_real:")
    print(sorted(columns_in_df4_not_in_df4_real))

if columns_in_df4_real_not_in_df4:
    print("Columns in df4_real but not in df4:")
    print(sorted(columns_in_df4_real_not_in_df4))

if not columns_in_df4_not_in_df4_real and not columns_in_df4_real_not_in_df4:
    print("The columns in df4 and df4_real are identical.")


# In[101]:


# Identify columns unique to each DataFrame
columns_in_df4_not_in_df1 = set(df4.columns) - set(df1.columns)
columns_in_df1_not_in_df4 = set(df1.columns) - set(df4.columns)

# Print the differences
if columns_in_df4_not_in_df1:
    print("Columns in df4 but not in df1:")
    print(sorted(columns_in_df4_not_in_df1))

if columns_in_df1_not_in_df4:
    print("Columns in df1 but not in df4:")
    print(sorted(columns_in_df1_not_in_df4))

if not columns_in_df4_not_in_df1 and not columns_in_df1_not_in_df4:
    print("The columns in df4 and df1 are identical.")


# In[102]:


# Identify columns unique to each DataFrame
columns_in_df1_not_in_df4_real = set(df1.columns) - set(df4_real.columns)
columns_in_df4_real_not_in_df1 = set(df4_real.columns) - set(df1.columns)

# Print the differences
if columns_in_df1_not_in_df4_real:
    print("Columns in df1 but not in df4_real:")
    print(sorted(columns_in_df1_not_in_df4_real))

if columns_in_df4_real_not_in_df1:
    print("Columns in df4_real but not in df1:")
    print(sorted(columns_in_df4_real_not_in_df1))

if not columns_in_df1_not_in_df4_real and not columns_in_df4_real_not_in_df1:
    print("The columns in df1 and df4_real are identical.")


# In[103]:


# append df4_real dataframe to df1 by rows
print(df1.shape)
print(df4_real.shape)
df5=pd.concat([df1, df4_real], ignore_index=True)
print(df5.shape)


# In[104]:


df1.head()


# In[105]:


df4_real.head()


# In[106]:


df5.head()


# In[107]:


df1.tail()


# In[108]:


df4_real.tail()


# In[109]:


# df5.tail(20)
df5.tail()


# Predicting print estimated time using model in experiment 8 neural network model
# for testing of real data from 3D-printer

# In[110]:


df6=df5.copy()
df6 = pd.get_dummies(df6, drop_first=True)
df6 = df6.astype(float)
scaler = StandardScaler()
numerical_cols = df6.select_dtypes(include=['float64', 'int64']).columns
df6[numerical_cols] = scaler.fit_transform(df6[numerical_cols])

X = df6.drop('ET-mm', axis=1)  # Replace 'ET-mm' with your actual target column name
y = df6['ET-mm']

# Split the original data into training and test sets
# Ensure the last 15 rows are used for testing
X_train = X.iloc[:-15]
X_test = X.iloc[-15:]
y_train = y.iloc[:-15]
y_test = y.iloc[-15:]

# Further split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define and compile the model (use your actual model definition here)
model_9 = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])
model_9.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model and save the history
history = model_9.fit(X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_val, y_val),  # Pass validation data
                    verbose=1)

# Evaluate the model
y_pred = model_9.predict(X_test).flatten()

# Compute Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

# Print training, validation, and test metrics
train_loss, train_mae = model_9.evaluate(X_train, y_train, verbose=0)
val_loss, val_mae = model_9.evaluate(X_val, y_val, verbose=0)
test_loss, test_mae = model_9.evaluate(X_test, y_test)

print(f"Training Mean Absolute Error: {train_mae}")
print(f"Validation Mean Absolute Error: {val_mae}")
print(f"Test Mean Absolute Error: {test_mae}")

# Plot training and validation loss
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 6))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()


# Predict on the test set
y_pred = model_9.predict(X_test).flatten()

# Compute Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Compute Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Compute R² Score
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")

# Compute training accuracy
train_loss, train_mae = model_9.evaluate(X_train, y_train, verbose=0)
print(f"Training Mean Absolute Error: {train_mae}")

# Compute validation accuracy
val_loss, val_mae = model_9.evaluate(X_val, y_val, verbose=0)
print(f"Validation Mean Absolute Error: {val_mae}")

# Compute test accuracy
test_loss, test_mae = model_9.evaluate(X_test, y_test)
print(f"Test Mean Absolute Error: {test_mae}")

# Since your model is using MAE, convert it to accuracy
train_accuracy = 1 - train_mae
val_accuracy = 1 - val_mae
test_accuracy = 1 - test_mae

print(f"Training Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")
print(f"Test Accuracy: {test_accuracy}")


# In[111]:


print(y_test)
print(y_pred)


# Using Dropout and batchnormalization to avoid over fitting

# In[112]:


df6=df5.copy()
df6 = pd.get_dummies(df6, drop_first=True)
df6 = df6.astype(float)
scaler = StandardScaler()
numerical_cols = df6.select_dtypes(include=['float64', 'int64']).columns
df6[numerical_cols] = scaler.fit_transform(df6[numerical_cols])

X = df6.drop('ET-mm', axis=1)  # Replace 'ET-mm' with your actual target column name
y = df6['ET-mm']

# Split the original data into training and test sets
# Ensure the last 15 rows are used for testing
X_train = X.iloc[:-15]
X_test = X.iloc[-15:]
y_train = y.iloc[:-15]
y_test = y.iloc[-15:]

# Further split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define and compile the model (use your actual model definition here)
model_9 = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    BatchNormalization(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)
])
model_9.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model and save the history
history = model_9.fit(X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_val, y_val),  # Pass validation data
                    verbose=1)

# Evaluate the model
y_pred = model_9.predict(X_test).flatten()

# Compute Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

# Print training, validation, and test metrics
train_loss, train_mae = model_9.evaluate(X_train, y_train, verbose=0)
val_loss, val_mae = model_9.evaluate(X_val, y_val, verbose=0)
test_loss, test_mae = model_9.evaluate(X_test, y_test)

print(f"Training Mean Absolute Error: {train_mae}")
print(f"Validation Mean Absolute Error: {val_mae}")
print(f"Test Mean Absolute Error: {test_mae}")

# Plot training and validation loss
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 6))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()


# Predict on the test set
y_pred = model_9.predict(X_test).flatten()

# Compute Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Compute Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Compute R² Score
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")

# Compute training accuracy
train_loss, train_mae = model_9.evaluate(X_train, y_train, verbose=0)
print(f"Training Mean Absolute Error: {train_mae}")

# Compute validation accuracy
val_loss, val_mae = model_9.evaluate(X_val, y_val, verbose=0)
print(f"Validation Mean Absolute Error: {val_mae}")

# Compute test accuracy
test_loss, test_mae = model_9.evaluate(X_test, y_test)
print(f"Test Mean Absolute Error: {test_mae}")

# Since your model is using MAE, convert it to accuracy
train_accuracy = 1 - train_mae
val_accuracy = 1 - val_mae
test_accuracy = 1 - test_mae

print(f"Training Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")
print(f"Test Accuracy: {test_accuracy}")


# Using Dropout, batchnormalization, l2 regularizers to avoid over fitting

# In[113]:


df6=df5.copy()
df6 = pd.get_dummies(df6, drop_first=True)
df6 = df6.astype(float)
scaler = StandardScaler()
numerical_cols = df6.select_dtypes(include=['float64', 'int64']).columns
df6[numerical_cols] = scaler.fit_transform(df6[numerical_cols])

X = df6.drop('ET-mm', axis=1)  # Replace 'ET-mm' with your actual target column name
y = df6['ET-mm']

# Split the original data into training and test sets
# Ensure the last 15 rows are used for testing
X_train = X.iloc[:-15]
X_test = X.iloc[-15:]
y_train = y.iloc[:-15]
y_test = y.iloc[-15:]

# Further split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define and compile the model (use your actual model definition here)
model_9 = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.3),
    BatchNormalization(),
    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.3),
    BatchNormalization(),
    Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dense(1)
])

# Compile the model with a lower learning rate
model_9.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error', metrics=['mae'])

# Train the model and save the history
history = model_9.fit(X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_val, y_val),  # Pass validation data
                    verbose=1)

# Evaluate the model
y_pred = model_9.predict(X_test).flatten()

# Compute Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

# Print training, validation, and test metrics
train_loss, train_mae = model_9.evaluate(X_train, y_train, verbose=0)
val_loss, val_mae = model_9.evaluate(X_val, y_val, verbose=0)
test_loss, test_mae = model_9.evaluate(X_test, y_test)

print(f"Training Mean Absolute Error: {train_mae}")
print(f"Validation Mean Absolute Error: {val_mae}")
print(f"Test Mean Absolute Error: {test_mae}")

# Plot training and validation loss
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 6))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()


# Predict on the test set
y_pred = model_9.predict(X_test).flatten()

# Compute Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Compute Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Compute R² Score
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")

# Compute training accuracy
train_loss, train_mae = model_9.evaluate(X_train, y_train, verbose=0)
print(f"Training Mean Absolute Error: {train_mae}")

# Compute validation accuracy
val_loss, val_mae = model_9.evaluate(X_val, y_val, verbose=0)
print(f"Validation Mean Absolute Error: {val_mae}")

# Compute test accuracy
test_loss, test_mae = model_9.evaluate(X_test, y_test)
print(f"Test Mean Absolute Error: {test_mae}")

# Since your model is using MAE, convert it to accuracy
train_accuracy = 1 - train_mae
val_accuracy = 1 - val_mae
test_accuracy = 1 - test_mae

print(f"Training Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")
print(f"Test Accuracy: {test_accuracy}")


# Using Dropout and batchnormalization to avoid over fitting. And using minmax scaler instead of standardscaler.

# In[114]:


df6=df5.copy()
df6 = pd.get_dummies(df6, drop_first=True)
df6 = df6.astype(float)
scaler = MinMaxScaler()
numerical_cols = df6.select_dtypes(include=['float64', 'int64']).columns
df6[numerical_cols] = scaler.fit_transform(df6[numerical_cols])

X = df6.drop('ET-mm', axis=1)  # Replace 'ET-mm' with your actual target column name
y = df6['ET-mm']

# Split the original data into training and test sets
# Ensure the last 15 rows are used for testing
X_train = X.iloc[:-15]
X_test = X.iloc[-15:]
y_train = y.iloc[:-15]
y_test = y.iloc[-15:]

# Further split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define and compile the model (use your actual model definition here)
model_9 = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    BatchNormalization(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)
])
model_9.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model and save the history
history = model_9.fit(X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_val, y_val),  # Pass validation data
                    verbose=1)

# Evaluate the model
y_pred = model_9.predict(X_test).flatten()

# Compute Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

# Print training, validation, and test metrics
train_loss, train_mae = model_9.evaluate(X_train, y_train, verbose=0)
val_loss, val_mae = model_9.evaluate(X_val, y_val, verbose=0)
test_loss, test_mae = model_9.evaluate(X_test, y_test)

print(f"Training Mean Absolute Error: {train_mae}")
print(f"Validation Mean Absolute Error: {val_mae}")
print(f"Test Mean Absolute Error: {test_mae}")

# Plot training and validation loss
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 6))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()


# Predict on the test set
y_pred = model_9.predict(X_test).flatten()

# Compute Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Compute Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Compute R² Score
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")

# Compute training accuracy
train_loss, train_mae = model_9.evaluate(X_train, y_train, verbose=0)
print(f"Training Mean Absolute Error: {train_mae}")

# Compute validation accuracy
val_loss, val_mae = model_9.evaluate(X_val, y_val, verbose=0)
print(f"Validation Mean Absolute Error: {val_mae}")

# Compute test accuracy
test_loss, test_mae = model_9.evaluate(X_test, y_test)
print(f"Test Mean Absolute Error: {test_mae}")

# Since your model is using MAE, convert it to accuracy
train_accuracy = 1 - train_mae
val_accuracy = 1 - val_mae
test_accuracy = 1 - test_mae

print(f"Training Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")
print(f"Test Accuracy: {test_accuracy}")


# Using Dropout and batchnormalization to avoid over fitting. And using robustscaler instead of standardscaler.

# In[115]:


df6=df5.copy()
df6 = pd.get_dummies(df6, drop_first=True)
df6 = df6.astype(float)
scaler = RobustScaler()
numerical_cols = df6.select_dtypes(include=['float64', 'int64']).columns
df6[numerical_cols] = scaler.fit_transform(df6[numerical_cols])

X = df6.drop('ET-mm', axis=1)  # Replace 'ET-mm' with your actual target column name
y = df6['ET-mm']

# Split the original data into training and test sets
# Ensure the last 15 rows are used for testing
X_train = X.iloc[:-15]
X_test = X.iloc[-15:]
y_train = y.iloc[:-15]
y_test = y.iloc[-15:]

# Further split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define and compile the model (use your actual model definition here)
model_9 = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    BatchNormalization(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)
])
model_9.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model and save the history
history = model_9.fit(X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_val, y_val),  # Pass validation data
                    verbose=1)

# Evaluate the model
y_pred = model_9.predict(X_test).flatten()

# Compute Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

# Print training, validation, and test metrics
train_loss, train_mae = model_9.evaluate(X_train, y_train, verbose=0)
val_loss, val_mae = model_9.evaluate(X_val, y_val, verbose=0)
test_loss, test_mae = model_9.evaluate(X_test, y_test)

print(f"Training Mean Absolute Error: {train_mae}")
print(f"Validation Mean Absolute Error: {val_mae}")
print(f"Test Mean Absolute Error: {test_mae}")

# Plot training and validation loss
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 6))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()


# Predict on the test set
y_pred = model_9.predict(X_test).flatten()

# Compute Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Compute Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Compute R² Score
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")

# Compute training accuracy
train_loss, train_mae = model_9.evaluate(X_train, y_train, verbose=0)
print(f"Training Mean Absolute Error: {train_mae}")

# Compute validation accuracy
val_loss, val_mae = model_9.evaluate(X_val, y_val, verbose=0)
print(f"Validation Mean Absolute Error: {val_mae}")

# Compute test accuracy
test_loss, test_mae = model_9.evaluate(X_test, y_test)
print(f"Test Mean Absolute Error: {test_mae}")

# Since your model is using MAE, convert it to accuracy
train_accuracy = 1 - train_mae
val_accuracy = 1 - val_mae
test_accuracy = 1 - test_mae

print(f"Training Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")
print(f"Test Accuracy: {test_accuracy}")


# Using Dropout and batchnormalization to avoid over fitting. And using maxabscaler instead of standardscaler.

# In[116]:


df6=df5.copy()
df6 = pd.get_dummies(df6, drop_first=True)
df6 = df6.astype(float)
scaler = MaxAbsScaler()
numerical_cols = df6.select_dtypes(include=['float64', 'int64']).columns
df6[numerical_cols] = scaler.fit_transform(df6[numerical_cols])

X = df6.drop('ET-mm', axis=1)  # Replace 'ET-mm' with your actual target column name
y = df6['ET-mm']

# Split the original data into training and test sets
# Ensure the last 15 rows are used for testing
X_train = X.iloc[:-15]
X_test = X.iloc[-15:]
y_train = y.iloc[:-15]
y_test = y.iloc[-15:]

# Further split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define and compile the model (use your actual model definition here)
model_9 = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    BatchNormalization(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)
])
model_9.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model and save the history
history = model_9.fit(X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_val, y_val),  # Pass validation data
                    verbose=1)

# Evaluate the model
y_pred = model_9.predict(X_test).flatten()

# Compute Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

# Print training, validation, and test metrics
train_loss, train_mae = model_9.evaluate(X_train, y_train, verbose=0)
val_loss, val_mae = model_9.evaluate(X_val, y_val, verbose=0)
test_loss, test_mae = model_9.evaluate(X_test, y_test)

print(f"Training Mean Absolute Error: {train_mae}")
print(f"Validation Mean Absolute Error: {val_mae}")
print(f"Test Mean Absolute Error: {test_mae}")

# Plot training and validation loss
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 6))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()


# Predict on the test set
y_pred = model_9.predict(X_test).flatten()

# Compute Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Compute Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Compute R² Score
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")

# Compute training accuracy
train_loss, train_mae = model_9.evaluate(X_train, y_train, verbose=0)
print(f"Training Mean Absolute Error: {train_mae}")

# Compute validation accuracy
val_loss, val_mae = model_9.evaluate(X_val, y_val, verbose=0)
print(f"Validation Mean Absolute Error: {val_mae}")

# Compute test accuracy
test_loss, test_mae = model_9.evaluate(X_test, y_test)
print(f"Test Mean Absolute Error: {test_mae}")

# Since your model is using MAE, convert it to accuracy
train_accuracy = 1 - train_mae
val_accuracy = 1 - val_mae
test_accuracy = 1 - test_mae

print(f"Training Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")
print(f"Test Accuracy: {test_accuracy}")


# In[117]:


# Assuming df5['ET-mm'] is the original target data before scaling
scaler_y = MaxAbsScaler()
scaler_y.fit(df5[['ET-mm']])  # Fit scaler on the original target data

# Inverse transform the scaled y_test and y_pred
y_test_original = scaler_y.inverse_transform(y_test.values.reshape(-1, 1)).flatten()
y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

# Display the original and predicted values side by side
comparison_df = pd.DataFrame({
    'Original': y_test_original,
    'Predicted': y_pred_original
})

print(comparison_df)
plt.figure(figsize=(10, 6))
plt.plot(y_test_original, label='Original Values', marker='o')
plt.plot(y_pred_original, label='Predicted Values', marker='x')
plt.xlabel('Test Data Index')
plt.ylabel('ET-mm')
plt.title('Comparison of Original and Predicted Values')
plt.legend()
plt.grid(True)
plt.show()


# Doing some hyperparameter tuning

# In[118]:


df6=df5.copy()
df6 = pd.get_dummies(df6, drop_first=True)
df6 = df6.astype(float)
scaler = MaxAbsScaler()
numerical_cols = df6.select_dtypes(include=['float64', 'int64']).columns
df6[numerical_cols] = scaler.fit_transform(df6[numerical_cols])

X = df6.drop('ET-mm', axis=1)  # Replace 'ET-mm' with your actual target column name
y = df6['ET-mm']

# Split the original data into training and test sets
# Ensure the last 15 rows are used for testing
X_train = X.iloc[:-15]
X_test = X.iloc[-15:]
y_train = y.iloc[:-15]
y_test = y.iloc[-15:]

# Further split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Learning rate scheduler function
def lr_schedule(epoch, lr):
    if epoch > 75:
        lr = lr * 0.5
    return lr

# Define and compile the model
model_9 = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.001)),
    Dropout(0.2),
    BatchNormalization(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.2),
    BatchNormalization(),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
model_9.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Define early stopping and learning rate scheduler
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = LearningRateScheduler(lr_schedule)

# Train the model with callbacks
history = model_9.fit(X_train, y_train,
                      epochs=200,
                      batch_size=32,
                      validation_data=(X_val, y_val),
                      verbose=1,
                      callbacks=[early_stopping, lr_scheduler])

# Evaluate the model
y_pred = model_9.predict(X_test).flatten()

# Compute Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")

# Print training, validation, and test metrics
train_loss, train_mae = model_9.evaluate(X_train, y_train, verbose=0)
val_loss, val_mae = model_9.evaluate(X_val, y_val, verbose=0)
test_loss, test_mae = model_9.evaluate(X_test, y_test)

print(f"Training Mean Absolute Error: {train_mae}")
print(f"Validation Mean Absolute Error: {val_mae}")
print(f"Test Mean Absolute Error: {test_mae}")

# Plot training and validation loss
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 6))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# Since your model is using MAE, convert it to accuracy
train_accuracy = 1 - train_mae
val_accuracy = 1 - val_mae
test_accuracy = 1 - test_mae

print(f"Training Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

##############################
# Assuming df5['ET-mm'] is the original target data before scaling
scaler_y = MaxAbsScaler()
scaler_y.fit(df5[['ET-mm']])  # Fit scaler on the original target data

# Inverse transform the scaled y_test and y_pred
y_test_original = scaler_y.inverse_transform(y_test.values.reshape(-1, 1)).flatten()
y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

# Display the original and predicted values side by side
comparison_df = pd.DataFrame({
    'Original': y_test_original,
    'Predicted': y_pred_original
})

print(comparison_df)
plt.figure(figsize=(10, 6))
plt.plot(y_test_original, label='Original Values', marker='o')
plt.plot(y_pred_original, label='Predicted Values', marker='x')
plt.xlabel('Test Data Index')
plt.ylabel('ET-mm')
plt.title('Comparison of Original and Predicted Values')
plt.legend()
plt.grid(True)
plt.show()


# Further working on accuracy of test data which is real data

# In[119]:


df6 = df5.copy()
df6 = pd.get_dummies(df6, drop_first=True)
df6 = df6.astype(float)
scaler = MaxAbsScaler()
numerical_cols = df6.select_dtypes(include=['float64', 'int64']).columns
df6[numerical_cols] = scaler.fit_transform(df6[numerical_cols])

X = df6.drop('ET-mm', axis=1)  # Replace 'ET-mm' with your actual target column name
y = df6['ET-mm']

# Split the original data into training and test sets
# Ensure the last 15 rows are used for testing
X_train = X.iloc[:-15]
X_test = X.iloc[-15:]
y_train = y.iloc[:-15]
y_test = y.iloc[-15:]

# Further split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Updated learning rate schedule
def lr_schedule(epoch, lr):
    if epoch > 50:
        lr = lr * 0.5
    return lr

# Define and compile the model
model_10 = Sequential([
    Dense(256, input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.0005)),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),
    BatchNormalization(),
    Dense(128, kernel_regularizer=l2(0.0005)),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),
    BatchNormalization(),
    Dense(64, kernel_regularizer=l2(0.0005)),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)
])
model_10.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

# Early stopping and learning rate scheduler
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = LearningRateScheduler(lr_schedule)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

# Train the model with callbacks
history = model_10.fit(X_train, y_train,
                      epochs=200,
                      batch_size=32,
                      validation_data=(X_val, y_val),
                      verbose=1,
                      callbacks=[early_stopping, lr_scheduler, reduce_lr])

# Evaluate the model
y_pred = model_10.predict(X_test).flatten()

# Compute the evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")

# Print training, validation, and test metrics
train_loss, train_mae = model_10.evaluate(X_train, y_train, verbose=0)
val_loss, val_mae = model_10.evaluate(X_val, y_val, verbose=0)
test_loss, test_mae = model_10.evaluate(X_test, y_test)

print(f"Training Mean Absolute Error: {train_mae}")
print(f"Validation Mean Absolute Error: {val_mae}")
print(f"Test Mean Absolute Error: {test_mae}")

# Plot training and validation loss
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 6))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# Since your model is using MAE, convert it to accuracy
train_accuracy = 1 - train_mae
val_accuracy = 1 - val_mae
test_accuracy = 1 - test_mae

print(f"Training Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

# Inverse transform the scaled y_test and y_pred
y_test_original = scaler_y.inverse_transform(y_test.values.reshape(-1, 1)).flatten()
y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

# Display the original and predicted values side by side
comparison_df = pd.DataFrame({
    'Original': y_test_original,
    'Predicted': y_pred_original
})

print(comparison_df)
plt.figure(figsize=(10, 6))
plt.plot(y_test_original, label='Original Values', marker='o')
plt.plot(y_pred_original, label='Predicted Values', marker='x')
plt.xlabel('Test Data Index')
plt.ylabel('ET-mm')
plt.title('Comparison of Original and Predicted Values')
plt.legend()
plt.grid(True)
plt.show()


# In[120]:


df6 = df5.copy()
df6 = pd.get_dummies(df6, drop_first=True)
df6 = df6.astype(float)
scaler = MaxAbsScaler()
numerical_cols = df6.select_dtypes(include=['float64', 'int64']).columns
df6[numerical_cols] = scaler.fit_transform(df6[numerical_cols])

X = df6.drop('ET-mm', axis=1)  # Replace 'ET-mm' with your actual target column name
y = df6['ET-mm']

# Split the original data into training and test sets
# Ensure the last 15 rows are used for testing
X_train = X.iloc[:-15]
X_test = X.iloc[-15:]
y_train = y.iloc[:-15]
y_test = y.iloc[-15:]

# Further split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Updated learning rate schedule
def lr_schedule(epoch, lr):
    if epoch > 50:
        lr = lr * 0.5
    return lr

# Define and compile a simpler model
model_11 = Sequential([
    Dense(128, input_shape=(X_train.shape[1],)),
    ReLU(),
    Dropout(0.3),
    Dense(64),
    ReLU(),
    Dropout(0.3),
    Dense(32),
    ReLU(),
    Dense(1)
])
model_11.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

# Define early stopping and learning rate scheduler
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Train the model with callbacks
history = model_11.fit(X_train, y_train,
                      epochs=200,
                      batch_size=32,
                      validation_data=(X_val, y_val),
                      verbose=1,
                      callbacks=[early_stopping, reduce_lr])

# Evaluate the model
y_pred = model_11.predict(X_test).flatten()

# Compute evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")

# Print training, validation, and test metrics
train_loss, train_mae = model_11.evaluate(X_train, y_train, verbose=0)
val_loss, val_mae = model_11.evaluate(X_val, y_val, verbose=0)
test_loss, test_mae = model_11.evaluate(X_test, y_test)

print(f"Training Mean Absolute Error: {train_mae}")
print(f"Validation Mean Absolute Error: {val_mae}")
print(f"Test Mean Absolute Error: {test_mae}")

# Plot training and validation loss
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 6))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# Since your model is using MAE, convert it to accuracy
train_accuracy = 1 - train_mae
val_accuracy = 1 - val_mae
test_accuracy = 1 - test_mae

print(f"Training Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

# Inverse transform the scaled y_test and y_pred
y_test_original = scaler_y.inverse_transform(y_test.values.reshape(-1, 1)).flatten()
y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

# Display the original and predicted values side by side
comparison_df = pd.DataFrame({
    'Original': y_test_original,
    'Predicted': y_pred_original
})

print(comparison_df)
plt.figure(figsize=(10, 6))
plt.plot(y_test_original, label='Original Values', marker='o')
plt.plot(y_pred_original, label='Predicted Values', marker='x')
plt.xlabel('Test Data Index')
plt.ylabel('ET-mm')
plt.title('Comparison of Original and Predicted Values')
plt.legend()
plt.grid(True)
plt.show()


# In[121]:


# Load and preprocess data
df6 = df5.copy()
df6 = pd.get_dummies(df6, drop_first=True)
df6 = df6.astype(float)

# Scale features
scaler = MaxAbsScaler()
numerical_cols = df6.select_dtypes(include=['float64', 'int64']).columns
df6[numerical_cols] = scaler.fit_transform(df6[numerical_cols])

# Split features and target
X = df6.drop('ET-mm', axis=1)
y = df6['ET-mm']

# Split the data into training and test sets
X_train = X.iloc[:-15]
X_test = X.iloc[-15:]
y_train = y.iloc[:-15]
y_test = y.iloc[-15:]

# Further split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define and compile a more complex model
model_11 = Sequential([
    Dense(256, input_shape=(X_train.shape[1],), kernel_regularizer='l2'),
    ReLU(),
    Dropout(0.3),
    Dense(128, kernel_regularizer='l2'),
    ReLU(),
    Dropout(0.3),
    Dense(64, kernel_regularizer='l2'),
    ReLU(),
    Dense(32, kernel_regularizer='l2'),
    ReLU(),
    Dense(1)
])
model_11.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

# Define early stopping and learning rate scheduler
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)

# Train the model with callbacks
history = model_11.fit(X_train, y_train,
                      epochs=300,  # Increased epochs
                      batch_size=64,  # Changed batch size
                      validation_data=(X_val, y_val),
                      verbose=1,
                      callbacks=[early_stopping, reduce_lr])

# Evaluate the model
y_pred = model_11.predict(X_test).flatten()

# Compute evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")

# Print training, validation, and test metrics
train_loss, train_mae = model_11.evaluate(X_train, y_train, verbose=0)
val_loss, val_mae = model_11.evaluate(X_val, y_val, verbose=0)
test_loss, test_mae = model_11.evaluate(X_test, y_test)

print(f"Training Mean Absolute Error: {train_mae}")
print(f"Validation Mean Absolute Error: {val_mae}")
print(f"Test Mean Absolute Error: {test_mae}")

# Plot training and validation loss
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 6))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# Convert MAE to accuracy
train_accuracy = 1 - train_mae
val_accuracy = 1 - val_mae
test_accuracy = 1 - test_mae

print(f"Training Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

# Inverse transform the scaled y_test and y_pred
scaler_y = MaxAbsScaler()
scaler_y.fit(df5[['ET-mm']])  # Fit scaler on the original target data

y_test_original = scaler_y.inverse_transform(y_test.values.reshape(-1, 1)).flatten()
y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

# Display the original and predicted values side by side
comparison_df = pd.DataFrame({
    'Original': y_test_original,
    'Predicted': y_pred_original
})

print(comparison_df)
plt.figure(figsize=(10, 6))
plt.plot(y_test_original, label='Original Values', marker='o')
plt.plot(y_pred_original, label='Predicted Values', marker='x')
plt.xlabel('Test Data Index')
plt.ylabel('ET-mm')
plt.title('Comparison of Original and Predicted Values')
plt.legend()
plt.grid(True)
plt.show()


# Final modeling for print duration estimation in minutes

# In[122]:


df6 = df5.copy()
df6 = pd.get_dummies(df6, drop_first=True)
df6 = df6.astype(float)
scaler = MaxAbsScaler()
numerical_cols = df6.select_dtypes(include=['float64', 'int64']).columns
df6[numerical_cols] = scaler.fit_transform(df6[numerical_cols])

X = df6.drop('ET-mm', axis=1)
y = df6['ET-mm']

# Split the original data into training and test sets
X_train = X.iloc[:-15]
X_test = X.iloc[-15:]
y_train = y.iloc[:-15]
y_test = y.iloc[-15:]

# Further split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define a more complex model
model_11 = Sequential([
    Dense(256, input_shape=(X_train.shape[1],), kernel_regularizer='l2'),
    BatchNormalization(),
    LeakyReLU(),
    Dropout(0.4),
    Dense(128, kernel_regularizer='l2'),
    BatchNormalization(),
    LeakyReLU(),
    Dropout(0.4),
    Dense(64, kernel_regularizer='l2'),
    BatchNormalization(),
    LeakyReLU(),
    Dense(32, kernel_regularizer='l2'),
    BatchNormalization(),
    LeakyReLU(),
    Dropout(0.4),
    Dense(4, kernel_regularizer='l2'),
    BatchNormalization(),
    LeakyReLU(),
    Dense(1)
])
model_11.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

# Define early stopping, learning rate scheduler, and model checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
lr_scheduler = LearningRateScheduler(lambda epoch: 0.001 * 0.5**(epoch // 50))
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

# Train the model with callbacks
history = model_11.fit(X_train, y_train,
                      epochs=300,  # Increased epochs
                      batch_size=32,  # Adjusted batch size
                      validation_data=(X_val, y_val),
                      verbose=1,
                      callbacks=[early_stopping, reduce_lr, lr_scheduler, model_checkpoint])

# Evaluate the model
y_pred = model_11.predict(X_test).flatten()

# Compute evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")

# Print training, validation, and test metrics
train_loss, train_mae = model_11.evaluate(X_train, y_train, verbose=0)
val_loss, val_mae = model_11.evaluate(X_val, y_val, verbose=0)
test_loss, test_mae = model_11.evaluate(X_test, y_test)

print(f"Training Mean Absolute Error: {train_mae}")
print(f"Validation Mean Absolute Error: {val_mae}")
print(f"Test Mean Absolute Error: {test_mae}")

# Plot training and validation loss
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 6))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# Convert MAE to accuracy
train_accuracy = 1 - train_mae
val_accuracy = 1 - val_mae
test_accuracy = 1 - test_mae

print(f"Training Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

# Inverse transform the scaled y_test and y_pred
scaler_y = MaxAbsScaler()
scaler_y.fit(df5[['ET-mm']])  # Fit scaler on the original target data

y_test_original = scaler_y.inverse_transform(y_test.values.reshape(-1, 1)).flatten()
y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

# Display the original and predicted values side by side
comparison_df = pd.DataFrame({
    'Original': y_test_original,
    'Predicted': y_pred_original
})

print(comparison_df)
plt.figure(figsize=(10, 6))
plt.plot(y_test_original, label='Original Values', marker='o')
plt.plot(y_pred_original, label='Predicted Values', marker='x')
plt.xlabel('Test Data Index')
plt.ylabel('ET-mm')
plt.title('Comparison of Original and Predicted Values')
plt.legend()
plt.grid(True)
plt.show()


# In[123]:


# Fit a linear model to the original vs. predicted values
model = LinearRegression()
model.fit(y_pred_original.reshape(-1, 1), y_test_original)

# Predict the adjusted values
a, b = model.coef_[0], model.intercept_
y_pred_adjusted = a * y_pred_original + b

# Display results
comparison_df_linear = pd.DataFrame({
    'Original': y_test_original,
    'Predicted': y_pred_original,
    'Adjusted Predicted': y_pred_adjusted
})

a_rounded = round(float(a), 3)
b_rounded = round(float(b), 3)
print(f"final_predicted_print_duration_in_minutes =\n\t\t {a_rounded} * predicted_print_duration_in_minutes + {b_rounded}")
print(comparison_df_linear)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(y_test_original, label='Original Values', marker='o')
plt.plot(y_pred_original, label='Original Predicted Values', marker='x', linestyle='--')
plt.plot(y_pred_adjusted, label='Adjusted Predicted Values', marker='x', linestyle='-')
plt.xlabel('Test Data Index')
plt.ylabel('ET-mm')
plt.title('Comparison of Original and Adjusted Predicted Values')
plt.legend()
plt.grid(True)
plt.show()


# In[124]:


print(f"final_predicted_print_duration_in_minutes =\n\t\t {a_rounded} * predicted_print_duration_in_minutes + {b_rounded}")


# In[125]:


type(y_pred_adjusted)


# In[126]:


import numpy as np

def predict_energy_consumption_in_min_nparray(test_durations_in_min):
    """
    Predict the energy consumption and CO2 emissions based on the test_durations_in_min.

    Parameters:
    - test_durations_in_min: A list or numpy array of print durations in minutes.

    Returns:
    - A list of dictionaries with energy consumption and CO2 emissions values.
    """

    # Ensure test_durations_in_min is a numpy array
    test_durations_in_min = np.asarray(test_durations_in_min)

    # Convert minutes to seconds
    test_durations_in_sec = test_durations_in_min * 60

    # Reshape to a 2D array as the model expects
    test_duration_array = test_durations_in_sec.reshape(-1, 1)

    # Predict the energy consumption
    predicted_energy_Wh = model_linear.predict(test_duration_array)

    # Ensure predicted_energy_Wh is 2D and reshape if necessary
    if predicted_energy_Wh.ndim == 1:
        predicted_energy_Wh = predicted_energy_Wh.reshape(-1, 1)

    results = []
    for energy_Wh in predicted_energy_Wh:
        energy_Wh = energy_Wh[0]  # Extract scalar value
        energy_in_kWh = energy_Wh / 1000
        energy_in_J = energy_Wh * 3.6e3
        energy_in_kJ = energy_Wh * 3.6
        carbon_emissions_in_grams = energy_in_kWh * 300

        results.append({
            'Energy_Wh': round(energy_Wh, 3),
            'Energy_kWh': round(energy_in_kWh, 3),
            'Energy_J': round(energy_in_J, 3),
            'Energy_kJ': round(energy_in_kJ, 3),
            'CO2_emissions_grams': round(carbon_emissions_in_grams, 3)
        })

    return results

# Example usage with a numpy array of test durations
test_durations = y_pred_adjusted  # Example durations in minutes
results = predict_energy_consumption_in_min_nparray(test_durations)

# Print results
for idx, result in enumerate(results):
    print(f"Test Duration {round(float(test_durations[idx]),3)} minutes:")
    print(f"  Predicted Energy Consumption (Wh): {result['Energy_Wh']}")
    print(f"  Predicted Energy Consumption (kWh): {result['Energy_kWh']}")
    print(f"  Predicted Energy Consumption (Joules): {result['Energy_J']}")
    print(f"  Predicted Energy Consumption (Kilo Joules): {result['Energy_kJ']}")
    print(f"  Predicted CO2 Emissions (grams): {result['CO2_emissions_grams']}")
    print()


# In[126]:





# In[126]:





# In[126]:





# #Explainable AI for last model in validation of real data#

# Without using parallel processing

# In[127]:


# # Set the number of rows to consider for SHAP initialization

# # n = 10  # You can change this to use a different number of rows
# # Initialize the SHAP explainer using the new model_11
# # explainer = shap.KernelExplainer(model_11.predict, X_train[:n])

# # To explain all the rows in the data
# explainer = shap.KernelExplainer(model_11.predict, X_train)

# # Compute SHAP values for the entire test set
# shap_values = explainer.shap_values(X_test)

# # Create a DataFrame with the same columns as your data for visualization
# X_test_sample = pd.DataFrame(X_test, columns=df6.drop('ET-mm', axis=1).columns)

# # Iterate through all rows in the test set for SHAP value computation
# for row in range(X_test_sample.shape[0]):
#     # Extract SHAP values for the specific row
#     single_shap_values = shap_values[row]

#     # Extract base value and convert it to a numpy scalar if needed
#     base_value = explainer.expected_value
#     if isinstance(base_value, list) or isinstance(base_value, np.ndarray):
#         base_value = base_value[0]  # Use the first element if it's a list or array
#     if tf.is_tensor(base_value):
#         base_value = base_value.numpy()  # Convert tensor to numpy array

#     # Ensure that we are selecting a single explanation
#     if single_shap_values.ndim > 1:
#         single_shap_values = single_shap_values[:, 0]  # Select the first column if necessary

#     ###################
#     # Plot the waterfall chart for the specific row
#     shap.initjs()
#     explanation = shap.Explanation(values=single_shap_values, base_values=base_value, data=X_test_sample.iloc[row, :])
#     shap.plots.waterfall(explanation, max_display=10)

#     # Optionally, save the plot
#     plt.savefig(f'waterfall_plot_row_{row}.png')

#     ###########################
#     # Optionally, plot a force plot for the specific row
#     shap.force_plot(base_value, single_shap_values, X_test_sample.iloc[row, :])

#     # Optionally, save the force plot
#     shap.save_html(f'force_plot_row_{row}.html', shap.force_plot(base_value, single_shap_values, X_test_sample.iloc[row, :]))

# # Optionally, plot a summary plot for all SHAP values
# shap.summary_plot(shap_values, X_test_sample)


# By using parallel processing

# In[128]:


# Set the number of rows to consider for SHAP initialization

# n = 10  # You can change this to use a different number of rows
# Initialize the SHAP explainer using the new model_11
# explainer = shap.KernelExplainer(model_11.predict, X_train[:n])

# To explain all the rows in the data
# Initialize the SHAP explainer using KernelExplainer
explainer = shap.KernelExplainer(model_11.predict, X_train)

# Function to compute SHAP values for a batch
def compute_shap_values_batch(explainer, X_test_batch):
    return explainer.shap_values(X_test_batch)

# Define batch size and number of jobs for parallel processing
batch_size = 100  # Adjust based on available resources
n_jobs = -1  # Use all available CPU cores

# Initialize an empty list to store SHAP values
shap_values = []

# Iterate over the test set in batches and compute SHAP values in parallel
for i in range(0, X_test.shape[0], batch_size):
    # Create the batch
    X_test_batch = X_test[i:i+batch_size]

    # Parallel computation of SHAP values for the batch
    batch_shap_values = Parallel(n_jobs=n_jobs)(delayed(compute_shap_values_batch)(explainer, X_test_batch.iloc[[j]]) for j in range(X_test_batch.shape[0]))

    # Flatten the list of lists (as Parallel returns a list of lists)
    batch_shap_values_flattened = np.concatenate(batch_shap_values, axis=0)  # Ensure the shape is correct

    # Extend the main shap_values list with the results from the current batch
    shap_values.append(batch_shap_values_flattened)

# Concatenate all the SHAP values into a single array
shap_values = np.concatenate(shap_values, axis=0)

# Create a DataFrame with the same columns as your data for visualization
X_test_sample = pd.DataFrame(X_test, columns=df6.drop('ET-mm', axis=1).columns)

# Iterate through all rows in the test set for SHAP value computation
for row in range(X_test_sample.shape[0]):
    # Extract SHAP values for the specific row
    single_shap_values = shap_values[row]

    # Extract base value and convert it to a numpy scalar if needed
    base_value = explainer.expected_value
    if isinstance(base_value, list) or isinstance(base_value, np.ndarray):
        base_value = base_value[0]  # Use the first element if it's a list or array
    if tf.is_tensor(base_value):
        base_value = base_value.numpy()  # Convert tensor to numpy array

    # Ensure that we are selecting a single explanation
    if single_shap_values.ndim > 1:
        single_shap_values = single_shap_values[:, 0]  # Select the first column if necessary

    ###################
    # Plot the waterfall chart for the specific row
    shap.initjs()
    explanation = shap.Explanation(values=single_shap_values, base_values=base_value, data=X_test_sample.iloc[row, :])
    shap.plots.waterfall(explanation, max_display=10)

    # Optionally, save the plot
    plt.savefig(f'waterfall_plot_row_{row}.png')

    ###########################
    # Optionally, plot a force plot for the specific row
    shap.force_plot(base_value, single_shap_values, X_test_sample.iloc[row, :])

    # Optionally, save the force plot
    shap.save_html(f'force_plot_row_{row}.html', shap.force_plot(base_value, single_shap_values, X_test_sample.iloc[row, :]))

# Plot a summary plot for all SHAP values
shap.summary_plot(shap_values, X_test_sample)


# In[128]:





# In[128]:





# In[128]:





# In[128]:





# #Covert the notebook to html file#

# In[129]:


# !jupyter nbconvert --to html /content/drive/MyDrive/\>5GB/data_science/manali_master_thesis/parameter_analysis/Cura_Parameter_Analysis_v0.05.ipynb
# !jupyter nbconvert --to html --ClearOutputPreprocessor.enabled=True --output new_filename.html /content/drive/MyDrive/>5GB/data_science/manali_master_thesis/parameter_analysis/Cura_Parameter_Analysis_v0.05.ipynb
# !jupyter nbconvert --to html --ClearOutputPreprocessor.enabled=True /content/drive/MyDrive/\>5GB/data_science/manali_master_thesis/parameter_analysis/Cura_Parameter_Analysis_v0.06.ipynb


# In[132]:


# prompt: save the output also to html file
# !jupyter nbconvert --to html --execute --output Cura_Parameter_Analysis_with_output_v0.06.html /content/drive/MyDrive/\>5GB/data_science/manali_master_thesis/parameter_analysis/Cura_Parameter_Analysis_v0.06.ipynb
get_ipython().system('jupyter nbconvert --to html --ClearOutputPreprocessor.enabled=True --output Cura_Parameter_Analysis_v0.06.html /content/drive/MyDrive/\\>5GB/data_science/manali_master_thesis/parameter_analysis/Cura_Parameter_Analysis_v0.06.ipynb')


# In[130]:





# #Total time taken to execute the notebook#

# In[131]:


# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the elapsed time
print(f"Time taken to complete the notebook: {elapsed_time:.2f} seconds")


# Time consumed on different platforms of colab pro:
# 
# TPU - 1069 seconds

# In[131]:




