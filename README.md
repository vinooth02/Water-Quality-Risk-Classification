# Water Quality Risk Classification

## 📌 Project Overview

Water Quality Risk Classification is a machine learning-based application that predicts the risk level of a water sample based on different water-quality parameters.

The project uses a Decision Tree classification model and provides an interactive Streamlit web application where users can enter water-quality values and receive a prediction.

> ⚠️ This is an educational prototype and is not a substitute for laboratory water-quality testing.

---

## 🎯 Objectives

- Analyze important water-quality parameters.
- Train a machine learning classification model.
- Predict whether a water sample has lower or higher risk.
- Provide an easy-to-use web interface using Streamlit.
- Display the prediction probability.
- Warn users when input values are outside the typical training-data range.

---

## 🧪 Input Parameters

The application uses the following parameters:

- pH
- Hardness
- Total Dissolved Solids (TDS)
- Chlorine
- Sulfate
- Conductivity
- Organic Carbon
- Trihalomethanes
- Turbidity

---

## 🤖 Machine Learning Model

The project uses:

**Decision Tree Classifier**

The trained model is saved as:

```text
model.joblib
