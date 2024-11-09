import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve
from imblearn.over_sampling import SMOTE

# Load the dataset from csv with pandas
data = pd.read_csv('Heart_Disease_Prediction.csv')

# Label the Heart Disease column
# Turn text labels to numerical labels
# Presence will be 1
# Absence will be 0
data['Heart Disease'] = data['Heart Disease'].map({'Presence': 1, 'Absence': 0})

# Features (X) and target (y) variable
X = data.drop(columns=['Heart Disease'])  # every column except Heart Disease is a feature
y = data['Heart Disease']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train with Logistic Regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Predict on test data
y_pred = logreg.predict(X_test)
y_pred_proba = logreg.predict_proba(X_test)[:, 1]

# EVALUATION METRICS
c_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Calculate TN and FP for specificity
TN = c_matrix[0, 0]
FP = c_matrix[0, 1]
specificity = TN / (TN + FP)

f1 = f1_score(y_test, y_pred)

print("Confusion matrix: ", c_matrix)
print("Precision: ", precision)
print("Recall: ", recall)
print("Specificity: ", specificity)
print("F1 Score: ", f1)

# Graph 1: Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(c_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Graph 2: Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
plt.figure(figsize=(6, 5))
plt.plot(recall_vals, precision_vals, marker='.', label='Logistic Regression')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

# Graph 3: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, marker='.', label='Logistic Regression')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
