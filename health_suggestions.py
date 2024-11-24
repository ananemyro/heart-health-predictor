def health_suggestions(probability, features):
    suggestions = []

    # Cholesterol Threshold
    cholesterol_threshold = 200  # Default threshold
    if features['Age'][0] < 19:
        cholesterol_threshold = 170  # Lower threshold for individuals under 19
    if features['Cholesterol'][0] > cholesterol_threshold:
        suggestions.append({
            'issue': 'High Cholesterol',
            'suggestion': (
                "Your cholesterol is elevated based on your age or family history. High cholesterol can cause arterial plaque "
                "buildup, increasing the risk of heart disease and stroke. Adopt a heart-healthy diet with more fruits, "
                "vegetables, and whole grains, and consult your doctor for further evaluation."
            ),
            'further_reading': (
                "For more info on managing cholesterol, visit the American Heart Association's website: "
                "https://www.heart.org/en/health-topics/cholesterol"
            )
        })

    # Blood Pressure Threshold
    bp_threshold = 130  # Default threshold
    if features['Age'][0] < 40 or features['Diabetes'][0] == 'Yes' or features['Obesity'][0] == 'Yes':
        bp_threshold = 120  # Stricter threshold for younger individuals, diabetes, or obesity
    if features['Blood Pressure'][0] > bp_threshold:
        suggestions.append({
            'issue': 'High Blood Pressure',
            'suggestion': (
                "Your blood pressure is higher than the personalized recommendation. Hypertension strains your heart and "
                "blood vessels, increasing the risk of stroke and kidney damage. Reduce sodium, limit alcohol, exercise regularly, "
                "and consult your doctor if the issue persists."
            ),
            'further_reading': (
                "For more info on managing blood pressure, visit the CDC's website: "
                "https://www.cdc.gov/bloodpressure/"
            )
        })

    # Heart Rate Threshold
    hr_threshold = 100  # Default threshold
    if features['Stress Level'][0] >= 7 or features['Exercise Hours'][0] < 2:
        hr_threshold = 95  # Stricter threshold for high stress or low exercise
    if features['Heart Rate'][0] > hr_threshold:
        suggestions.append({
            'issue': 'High Heart Rate',
            'suggestion': (
                "Your heart rate is higher than it should be. This could be due to stress, dehydration, or underlying "
                "conditions. Manage stress with relaxation techniques, stay hydrated, and consult your doctor if the elevated "
                "rate persists."
            ),
            'further_reading': (
                "For more info on managing heart rate, visit the Mayo Clinic's website: "
                "https://www.mayoclinichealthsystem.org/hometown-health/speaking-of-health/know-your-heart-health-numbers"
            )
        })

    # Blood Sugar Threshold
    sugar_threshold = 126  # Default threshold
    if features['Obesity'][0] == 'Yes' or features['Family History'][0] == 'Yes':
        sugar_threshold = 110  # Lower threshold for obesity or family history
    if features['Blood Sugar'][0] > sugar_threshold:
        suggestions.append({
            'issue': 'High Blood Sugar',
            'suggestion': (
                "Your blood sugar is above the personalized threshold, indicating a potential risk of diabetes. Elevated blood "
                "sugar can harm your blood vessels and nerves over time. Monitor your levels regularly, adopt a low-glycemic diet, "
                "and consult a healthcare provider."
            ),
            'further_reading': (
                "For more info on managing blood sugar, visit the American Diabetes Association's website: "
                "https://www.diabetes.org/"
            )
        })

    # Exercise Hours
    exercise_threshold = 2  # Default threshold
    if features['Age'][0] >= 60:
        exercise_threshold = 3  # Higher threshold for older adults
    if features['Exercise Hours'][0] < exercise_threshold:
        suggestions.append({
            'issue': 'Low Physical Activity',
            'suggestion': (
                "Your reported physical activity is below the personalized recommendation. Regular exercise strengthens the heart "
                "and reduces heart disease risk. Aim for at least 150 minutes of moderate exercise weekly, increasing gradually as "
                "needed."
            ),
            'further_reading': (
                "For more info on physical activity guidelines, visit the WHO's website: "
                "https://www.who.int/news-room/fact-sheets/detail/physical-activity"
            )
        })

    # Obesity
    if features['Obesity'][0] == 'Yes':
        suggestions.append({
            'issue': 'Obesity',
            'suggestion': (
                "Obesity is a major risk factor for heart disease and diabetes. Excess weight strains your heart and increases "
                "blood pressure. Consult a healthcare provider for a personalized plan involving dietary changes, exercise, and "
                "behavioral support."
            ),
            'further_reading': (
                "For more info on managing obesity, visit the CDC's website: "
                "https://www.cdc.gov/obesity/"
            )
        })

    # Chest Pain
    if features['Chest Pain Type'][0] != '':
        suggestions.append({
            'issue': 'Chest Pain',
            'suggestion': (
                "Chest pain should never be ignored. It could indicate a serious condition such as angina or a heart attack. "
                "Seek immediate medical attention to determine the cause. Early diagnosis can be life-saving."
            ),
            'further_reading': (
                "For more info on chest pain, visit the American Heart Association's website: "
                "https://www.heart.org/en/health-topics/heart-attack/warning-signs-of-a-heart-attack"
            )
        })

    # Return suggestions
    if probability > 0.5:
        return suggestions
    else:
        return [{"issue": "Low Risk", "suggestion": "Maintain a healthy lifestyle to keep your risk low."}]
