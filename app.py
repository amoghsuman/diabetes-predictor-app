# app.py (Final Inference Version with SHAP Fix)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import shap
import streamlit.components.v1 as components

from utils.plots import (
    generate_pairplot,
    plot_correlation_matrix,
    plot_feature_distributions
)

from utils.model_helpers import (
    load_dataset,
    handle_missing_and_impute,
    handle_skewness,
    scale_features,
    load_model
)

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Title and intro
st.title("🧠 Diabetes Prediction using Keras")
st.markdown("This app predicts the likelihood of diabetes based on health parameters.")

# Load and clean data
data = load_dataset()

# Dataset overview
st.subheader("📊 Dataset Preview")
st.markdown("Here's a glimpse of the dataset used to train the model."
"Each row represents a patient, and each column contains a health-related feature.")
st.dataframe(data.head())
st.write(f"Shape of dataset: {data.shape}")
st.write(f"Columns: {list(data.columns)}")

# EDA
st.subheader("🔍 Data Exploration")
st.markdown("Before building a model, it's important to check if the data is clean and complete. "
"This section helps us identify missing values and understand the range and average of each feature.")

st.markdown("#### Missing Values (if any)")
missing_values = data.isnull().sum().sort_values(ascending=False)
st.dataframe(missing_values.head())

st.markdown("#### Statistical Summary")
st.dataframe(data.describe())

st.subheader("📉 Pairplot Visualization")
st.markdown("This shows how different features relate to each other."
"Each dot represents a patient, and the colors indicate whether they had diabetes or not.")

plot_path = "images/pairplot.png"
generate_pairplot(data, output_path=plot_path)
st.image(plot_path, caption="Seaborn Pairplot of Features Colored by Outcome", use_column_width=True)

st.subheader("🧠 Feature Correlation Matrix")
st.markdown("This heatmap highlights which health features are most related to diabetes. "
"Darker colors mean stronger relationships.")

corr_plot_path = "images/corr_heatmap.png"
plot_correlation_matrix(data, corr_plot_path)
st.image(corr_plot_path, caption="Correlation matrix (features with ≥0.2 correlation)", use_column_width=True)

# Data Cleaning + Scaling
st.subheader("🧼 Data Preprocessing")
st.markdown("We clean the data by replacing missing or zero values, and transform it so that the model understands it better."
"This step improves accuracy.")

data = handle_missing_and_impute(data)
data = handle_skewness(data)
st.success("Zero values handled and skewed features log-transformed.")

st.subheader("📈 Feature Distributions")
st.markdown("These charts show how each feature is distributed among diabetic and non-diabetic patients.")

dist_plot_path = "images/distributions.png"
plot_feature_distributions(data, dist_plot_path)
st.image(dist_plot_path, caption="Feature Distributions by Outcome", use_column_width=True)

st.subheader("⚙️ Feature Scaling")
st.markdown("We adjust the scale of all features so they are comparable. "
"This helps the model learn more effectively.")

scaled_data, scaler = scale_features(data)
st.success("Feature scaling completed using StandardScaler.")

# Load pre-trained model
model = load_model()
st.success("Pre-trained model loaded successfully.")

# Evaluation
st.subheader("✅ Model Evaluation on Sample Data")
st.markdown("We test the model's performance on unseen data to see how well it predicts diabetes. "
"The numbers here give us confidence in its predictions.")

X_all = scaled_data[:, [0, 1, 5, 7]]
Y_all = pd.get_dummies(scaled_data[:, 8])

Y_pred_probs = model.predict(X_all)
Y_pred = np.argmax(Y_pred_probs, axis=1)
Y_true = np.argmax(Y_all.values, axis=1)

accuracy = np.mean(Y_pred == Y_true)
st.write(f"🧪 **Accuracy on Sample Data:** `{accuracy * 100:.2f}%`")

st.subheader("📊 Confusion Matrix")
st.markdown("This table compares the model's predictions with actual outcomes. "
"It tells us how many patients were correctly or incorrectly classified.")

cm = confusion_matrix(Y_true, Y_pred)
fig, ax = plt.subplots(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=["No Diabetes", "Diabetes"], yticklabels=["No Diabetes", "Diabetes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
st.pyplot(fig)

st.subheader("📋 Classification Report")
st.markdown("This report summarizes how well the model distinguishes between diabetic and non-diabetic patients using precision, recall, and F1 score.")

report = classification_report(Y_true, Y_pred, target_names=["No-Diabetes", "Diabetes"], output_dict=True)
report_df = pd.DataFrame(report).T
st.dataframe(report_df.style.format(precision=2))

auc = roc_auc_score(Y_true, Y_pred_probs[:, 1])
st.markdown(f"🔵 **ROC AUC Score:** `{auc:.4f}`")

# Real-Time Prediction
st.sidebar.title("🧪 Predict Diabetes Risk")
st.sidebar.markdown("Enter your health details below and click the button to get your predicted risk of diabetes.")

st.sidebar.markdown("Enter the following health parameters:")

with st.sidebar.form("user_input_form"):
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=79)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
    age = st.number_input("Age", min_value=1, max_value=120, value=33)

    submitted = st.form_submit_button("✅ Predict Diabetes")

if submitted:
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]])
    input_data[:, [3, 4, 5, 6, 7]] = np.log(input_data[:, [3, 4, 5, 6, 7]] + 1)
    input_scaled = scaler.transform(input_data)
    input_selected = input_scaled[:, [0, 1, 5, 7]]

    prediction = model.predict(input_selected)
    prob_class_1 = prediction[0][1]

    st.markdown("### 🧾 Prediction Result")
    st.write(f"**🧮 Probability of having diabetes:** `{prob_class_1*100:.2f}%`")

    if prob_class_1 > 0.5:
        st.error("⚠️ High risk of diabetes. Please consult a healthcare professional.")
    else:
        st.success("🎉 Low risk of diabetes. Keep up the healthy lifestyle!")

    # SHAP Explainability using Waterfall Plot
    import shap
    import matplotlib.pyplot as plt

    st.subheader("🧠 Why did the model predict this?")
    st.markdown("This section explains how much each health feature contributed to your prediction. "
    "Think of it as a reasoning behind the model’s decision.")

    # Prepare background
    background = X_all[:100]
    #explainer = shap.KernelExplainer(model.predict, background)
    from utils.model_helpers import get_shap_explainer
    explainer = get_shap_explainer(model, background)

    shap_values = explainer.shap_values(input_selected)

    # Class 1 = Diabetes
    shap_vals = shap_values[0][:, 1]
    base_val = explainer.expected_value[1]
    features = input_selected[0]

    # Create Explanation
    exp = shap.Explanation(
        values=shap_vals,
        base_values=base_val,
        data=features,
        feature_names=["Pregnancies", "Glucose", "BMI", "Age"]
    )

    # Plot and display directly
    st.markdown("#### 🔍 Feature Impact (Waterfall Plot)")
    fig = plt.figure()
    shap.plots.waterfall(exp, show=True)
    st.pyplot(fig)

    # SHAP Bar Plot
    st.markdown("#### 📊 Feature Importance (SHAP Bar Plot)")

    fig_bar = plt.figure()
    shap.plots.bar(exp, show=True)
    st.pyplot(fig_bar)

