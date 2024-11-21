import numpy as np
import joblib

# Load the model and scaler
logreg = joblib.load('logistic_model.joblib')
scaler = joblib.load('scaler.joblib')

# New data (replace with actual data you want to predict)
new_data = np.array([[65, 'Male', 220, 140, 80, 'No', 'No', 2, 'Yes', 'Yes', 'No', 5, 120, 'No', 'Typical Angina']])

# Convert new_data to DataFrame and preprocess (e.g., encode categorical variables)
import pandas as pd
new_data_df = pd.DataFrame(new_data, columns=['Age', 'Gender', 'Cholesterol', 'Blood Pressure', 'Heart Rate',
                                              'Smoking', 'Alcohol Intake', 'Exercise Hours', 'Family History',
                                              'Diabetes', 'Obesity', 'Stress Level', 'Blood Sugar',
                                              'Exercise Induced Angina', 'Chest Pain Type'])

categorical_columns = ['Gender', 'Smoking', 'Alcohol Intake', 'Family History', 'Chest Pain Type', 'Diabetes', 
                       'Obesity', 'Exercise Induced Angina']
new_data_df = pd.get_dummies(new_data_df, columns=categorical_columns, drop_first=True)

# Ensure the new data matches the model's feature order
new_data_df = new_data_df.reindex(columns=scaler.feature_names_in_, fill_value=0)

# Scale the input data
new_data_scaled = scaler.transform(new_data_df)

# Predict
prediction = logreg.predict(new_data_scaled)
probability = logreg.predict_proba(new_data_scaled)[:, 1]


# Output results
print("Predicted Heart Disease (0=Absence, 1=Presence):", prediction[0])
print("Probability of Heart Disease:", probability[0])