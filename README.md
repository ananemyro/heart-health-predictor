# Heart Health Predictor 

## Project Overview
This project aims to create a machine learning model that predicts the likelihood of heart disease in patients based on health indicators like cholesterol levels, age, and blood pressure to be implemented into a web app where the user will provide several medical and lifestyle-related inputs in the form of text fields and the output will be a prediction indicating the likelihood of heart disease.

---

## Dataset
- **Source:** Heart Disease Prediction [dataset](https://www.kaggle.com/datasets/rashadrmammadov/heart-disease-prediction?resource=download&select=heart_disease_dataset.csv) from kaggle.
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
- Confusion Matrix
- Precision & Recall
- F1 Score
- ROC-AUC Curve

---
## Getting Started

### Prerequisites
- Python 3.7+
- Required packages: scikit-learn, pandas, numpy, seaborn, matplotlib, imbalanced-learn

### Files in the Repository
- Heart_Disease_Prediction.csv: The dataset containing patient data with health indicators used for training and testing the model.
- ml.py: The main script for data preprocessing, model training, evaluation, and generating visualizations.
- Project Deliverable 1.pdf: Document detailing the initial project proposal.

---

## Contributors
Anastasiia Nemyrovska, Shirley Ding, Michelle Sayeh, Lucie Shi

## Acknowledgments
Thanks to the MAIS 202 instructors and TPMs for their support in this project.

