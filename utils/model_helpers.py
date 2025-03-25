# Model loading & predictions 

import numpy as np
import pandas as pd
import streamlit as st

from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LinearRegression

from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE

# utils/model_helpers.py

import pandas as pd

# Load this dataset for EDA (exploratory data analysis).

# Potentially re-use it in both training and prediction stages.

def load_dataset(path="data/diabetes.csv"):
    column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                    'Insulin', 'BMI', 'PedigreeFunction', 'Age', 'Outcome']
    data = pd.read_csv(path, names=column_names)
    return data

#Task 5B, 5C, 5D: Imputation + Linear Regression Prediction


def handle_missing_and_impute(data):
    # Step 1: Replace 0s with NaN for specific features
    features_with_zero_invalid = ['BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Glucose']
    data_copy = data.copy()
    for col in features_with_zero_invalid:
        data_copy[col].replace(0, np.nan, inplace=True)
    
    # Step 2: Impute medians for some features
    for col in ['BloodPressure', 'BMI', 'SkinThickness', 'Insulin']:
        median_val = data[col].median()
        data[col].replace(0, median_val, inplace=True)

    # Step 3: Use linear regression to predict missing 'Glucose'
    metric = "Glucose"
    known = data[data[metric] != 0]
    unknown = data[data[metric] == 0]

    if not unknown.empty:
        X_known = known.drop(columns=[metric])
        y_known = known[metric]
        X_unknown = unknown.drop(columns=[metric])

        model = LinearRegression()
        model.fit(X_known, y_known)
        predicted_values = model.predict(X_unknown)
        data.loc[data[metric] == 0, metric] = predicted_values

    return data

#Task 6A: Handle Skewness
#This is data transformation
def handle_skewness(data):
    skewed_features = ['SkinThickness', 'Insulin', 'BMI', 'PedigreeFunction', 'Age']
    data = data.copy()
    for feature in skewed_features:
        data[feature] = np.log(data[feature] + 1)  # add 1 to avoid log(0)
    return data

#Task 6C: Feature Scaling

def scale_features(data):
    scaler = StandardScaler()
    data_scaled = data.copy()
    feature_array = data_scaled.to_numpy()
    feature_array[:, :8] = scaler.fit_transform(feature_array[:, :8])
    return feature_array, scaler

#Task 7: Dataset Splitting

def split_dataset_from_raw(X, Y):
    X_train, X_assess, Y_train, Y_assess = train_test_split(X, Y, test_size=0.4, random_state=10)
    X_val, X_test, Y_val, Y_test = train_test_split(X_assess, Y_assess, test_size=0.5, random_state=10)
    return X_train, X_val, X_test, Y_train, Y_val, Y_test

def split_dataset(scaled_data):
    # Select features (0: Pregnancies, 1: Glucose, 5: BMI, 7: Age)
    X = scaled_data[:, [0, 1, 5, 7]]
    
    # Outcome column (index 8)
    Y_raw = scaled_data[:, 8]
    Y = pd.get_dummies(Y_raw)  # One-hot encoding
    
    # Split into train and assess (40% held out)
    X_train, X_assess, Y_train, Y_assess = train_test_split(X, Y, test_size=0.4, random_state=10)
    
    # Split assess into validation and test (50-50 of assess)
    X_val, X_test, Y_val, Y_test = train_test_split(X_assess, Y_assess, test_size=0.5, random_state=10)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test

#Task 8: Creating the Neural Network Model

def build_model(input_dim):
    model = Sequential()
    model.add(Dense(12, input_dim=input_dim, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(2, activation='softmax'))  # 2 output classes
    return model

#Task 9A, 9B: Compile & Train
#Note that we are applying Class Weighting in Keras
#There is an imbalance (500 vs. 268) in the dataset, meaning that the model is likely biased toward predicting 0 (No Diabetes).
#Class weighting techniquw will give more importance to the under-represented 1 class (Diabetes).

def compile_and_train_model(model, X_train, Y_train, X_val, Y_val, epochs=40, batch_size=32):
    # Convert one-hot encoded Y_train to single label
    y_labels = np.argmax(Y_train.to_numpy(), axis=1)
    
    # Compute class weights
    weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_labels), y=y_labels)
    class_weights = dict(enumerate(weights))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        validation_data=(X_val, Y_val),
        batch_size=batch_size,
        verbose=1,
        class_weight=class_weights
    )
    return model, history

#Task 9C: Evaluate
def evaluate_model(model, X_test, Y_test):
    scores = model.evaluate(X_test, Y_test, verbose=0)
    return scores[1]  # accuracy

#Task 9D & 9E: Save Model & Weights
def save_model(model, model_path="diabetes_model"):
    # Save model architecture
    model_json = model.to_json()
    with open(f"{model_path}.json", "w") as json_file:
        json_file.write(model_json) 
    
    # Save weights
    model.save_weights(f"{model_path}.weights.h5")

#Task 12: Classification Report
#Use F1-Score and AUC for Evaluation

def get_classification_report(Y_test, Y_pred_probs, target_names=["No-Diabetes", "Diabetes"]):
    actuals = np.argmax(Y_test.to_numpy().T, axis=0)
    predicted = np.argmax(Y_pred_probs, axis=1)
    
    report = classification_report(actuals, predicted, target_names=target_names, output_dict=True)
    auc_score = roc_auc_score(actuals, Y_pred_probs[:,1])

    return report, auc_score

# Resampling with SMOTE or Undersampling
def apply_smote(X, Y):
    y_labels = np.argmax(Y.to_numpy(), axis=1)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y_labels)
    
    # Convert y_resampled back to one-hot
    Y_resampled = pd.get_dummies(y_resampled)
    return X_resampled, Y_resampled

import streamlit as st
from keras.models import model_from_json

@st.cache_resource
def load_model():
    with open("model.json", "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights("model.weights.h5")
    return model

@st.cache_resource
def get_shap_explainer(model, background):
    import shap
    return shap.KernelExplainer(model.predict, background)

