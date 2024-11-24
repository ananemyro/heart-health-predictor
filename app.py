from flask import Flask, request, render_template
import joblib
import pandas as pd

from health_suggestions import health_suggestions

app = Flask(__name__)

# Load the model and scaler
logreg = joblib.load('logistic_model.joblib')
scaler = joblib.load('scaler.joblib')


@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/assess')
def assess_page():
    return render_template('assess.html')

@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/results', methods=['POST'])
def results():
    # Get user inputs from the form
    inputs = request.form
    features = {
        'Age': [int(inputs['age'])],
        'Gender': [inputs['gender']],
        'Cholesterol': [int(inputs['cholesterol'])],
        'Blood Pressure': [int(inputs['blood_pressure'])],
        'Heart Rate': [int(inputs['heart_rate'])],
        'Smoking': [inputs['smoking']],
        'Alcohol Intake': [inputs['alcohol']],
        'Exercise Hours': [int(inputs['exercise'])],
        'Family History': [inputs['family_history']],
        'Diabetes': [inputs['diabetes']],
        'Obesity': [inputs['obesity']],
        'Stress Level': [int(inputs['stress_level'])],
        'Blood Sugar': [int(inputs['blood_sugar'])],
        'Exercise Induced Angina': [inputs['angina']],
        'Chest Pain Type': [inputs['chest_pain']],
    }

    new_data_df = pd.DataFrame(features, columns=['Age', 'Gender', 'Cholesterol', 'Blood Pressure', 'Heart Rate',
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
    prediction = logreg.predict(new_data_scaled)[0]
    probability = logreg.predict_proba(new_data_scaled)[0][1]

    # Convert probability to percentage
    probability_percentage = round(probability * 100)

    # Generate suggestions
    suggestions = health_suggestions(probability, features)

    # Determine the result message
    result_message = "Yes" if prediction == 1 else "No"
    description = (
        "you are at risk of heart disease." if prediction == 1
        else "you are not at risk of heart disease."
    )

    # Pass the data to the template
    return render_template(
        'results.html',
        result_message=result_message,
        description=description,
        probability=probability_percentage,
        suggestions=suggestions
    )

    # Return results to the user
    '''
    return f"""
        <h1>Results</h1>
        <p>Predicted Heart Disease (0=Absence, 1=Presence): <strong>{prediction}</strong></p>
        <p>Probability of Heart Disease: <strong>{probability}</strong></p>
        <a href="/">Back to Assessment</a>
    """
    '''

if __name__ == '__main__':
    app.run(debug=True)
