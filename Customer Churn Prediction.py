# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV

# Load dataset
data = pd.read_csv(r'C:\Users\venka\Downloads\data sets excel\WA_Fn-UseC_-Telco-Customer-Churn.csv')
data.head()

# Check for missing values
print(data.isnull().sum())

# Fill missing 'TotalCharges' with 0 (if any)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'].fillna(0, inplace=True)

# Drop irrelevant columns
data.drop(['customerID'], axis=1, inplace=True)

# Label Encoding for 'Churn' column
data['Churn'] = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Convert other categorical variables
categorical_features = data.select_dtypes(include=['object']).columns
data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

# Split the data into features and target variable
X = data.drop('Churn', axis=1)
y = data['Churn']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))




param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid search with cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters
print("Best parameters:", grid_search.best_params_)

# Final model
final_model = grid_search.best_estimator_
y_pred_final = final_model.predict(X_test)

# Evaluation of tuned model
print("Final Accuracy:", accuracy_score(y_test, y_pred_final))
print("Classification Report:\n", classification_report(y_test, y_pred_final))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_final))

# Plot feature importance
importances = final_model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
plt.title("Top 10 Features Affecting Churn")
plt.show()

#By analyzing the feature importances, we can understand the factors driving churn. This model may highlight attributes such as "Contract Type," "Monthly Charges," or "Tenure" as important, which can inform targeted strategies like discounts for high-risk customers or incentives for long-term contracts.