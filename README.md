# Heart Health Predictor 

## Project Overview
This project aims to create a machine learning model that predicts the likelihood of heart disease in patients based on health indicators like cholesterol levels, age, and blood pressure to be implemented into a web app where the user will provide several medical and lifestyle-related inputs in the form of text fields and the output will be a prediction indicating the likelihood of heart disease.

---

## Dataset
- **Source:** Heart Disease Prediction dataset from kaggle (https://www.kaggle.com/datasets/rishidamarla/heart-disease-prediction/data)
- **Description:** Contains labeled patient data, including health metrics like age, cholesterol, and blood pressure.
- **Target Variable:** Heart disease presence (binary classification: 1 = disease, 0 = no disease)

## Preprocessing
1. **Label Encoding:** Converted target labels to binary values.
2. **Feature Scaling:** Standardized features to improve model accuracy.
3. **Class Balancing:** Addressed using SMOTE to prevent model bias toward the majority class.

## Model
- **Chosen Model:** Logistic Regression
- **Frameworks:** scikit-learn for model building
- **Train/Test Split:** 80/20 split

## Evaluation Metrics
- **Confusion Matrix**
- **Precision & Recall**
- **F1 Score**
- **ROC-AUC Curve**

---

## Contributers
Anastasiia Nemyrovska, Shirley Ding, Michelle Sayeh, Lucie Shi

## Acknowledgments
Thanks to the MAIS 202 instructors and TPMs for their support in this project.

