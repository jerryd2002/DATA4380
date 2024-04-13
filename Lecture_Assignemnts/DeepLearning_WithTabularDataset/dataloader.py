#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm

def load_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    return df

def compare_features_binary(df):
    unique_counts = df.nunique()
    selected_columns = unique_counts[unique_counts > 4].index
    df_selected = df[selected_columns]

    for col in df_selected.select_dtypes(include=['number']).columns:
        sns.histplot(df[col], kde=True)
        plt.title(f'Histogram and Density Plot of {col}')
        plt.show()

    for col in df_selected.select_dtypes(include=['number']).columns:
        sns.boxplot(x=df[col])
        plt.title(f'Box Plot of {col}')
        plt.show()

    sns.pairplot(df_selected.select_dtypes(include=['number']), kind='reg')
    plt.suptitle('Pairwise Scatterplots of Numerical Variables with Regression Lines', y=1.02)
    plt.show()

    unique_counts_2 = df.nunique()
    selected_columns_2 = unique_counts[unique_counts < 4].index
    df_selected_2 = df[selected_columns_2]

    for col in df_selected_2.select_dtypes(include=['number']).columns:
        sns.countplot(x=df[col])
        plt.title(f'Frequency Plot of {col}')
        plt.show()

    correlation_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix Heatmap')
    plt.show()

def detect_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (column < lower_bound) | (column > upper_bound)

def process_data(df):
    label_encoder = LabelEncoder()
    df['Heart Disease'] = label_encoder.fit_transform(df['Heart Disease'])

    df = df[~detect_outliers(df['BP'])]
    df = df[~detect_outliers(df['Cholesterol'])]
    df = df[~detect_outliers(df['Max HR'])]
    df = df[~detect_outliers(df['ST depression'])]

    X = df.drop(columns=['Heart Disease'])
    y = df['Heart Disease']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix ')
    plt.show()

    y_probs = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = roc_auc_score(y_test, y_probs)
    print("ROC AUC:", roc_auc)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    
def logistic_regression_model(X_train, X_test, y_train, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_predict = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_predict)
    print("Logistic Regression Accuracy:", accuracy)

    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_predict), annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Logistic Regression')
    plt.show()

    y_probs = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = roc_auc_score(y_test, y_probs)
    print("Logistic Regression ROC AUC:", roc_auc)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - Logistic Regression')
    plt.legend()
    plt.show()
    

def random_forest_model(X_train, X_test, y_train, y_test):
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    predict = rfc.predict(X_test)
    score = accuracy_score(y_test, predict)
    print("Random Forest Accuracy:", score)

    y_probs = rfc.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = roc_auc_score(y_test, y_probs)
    print("Random Forest ROC AUC:", roc_auc)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Guessing')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - Random Forest')
    plt.legend()
    plt.show()

    cm = confusion_matrix(y_test, predict)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Random Forest')
    plt.show()


# In[ ]:




