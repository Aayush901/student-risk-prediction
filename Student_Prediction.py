import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# STEP 2: 
# Creating dataset
data = {
    'Attendance': ['High','Low','Medium','High','Low','Medium','High','Medium','Low','High'],
    'Internal_Marks': ['Good','Poor','Average','Good','Poor','Average','Good','Good','Poor','Average'],
    'Assignment_Submission': ['Yes','No','Yes','Yes','No','Yes','Yes','Yes','No','Yes'],
    'Result': ['Low Risk','High Risk','Medium Risk','Low Risk','High Risk',
               'Medium Risk','Low Risk','Low Risk','High Risk','Medium Risk']
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

# STEP 3: Proper Encoding (FIXED)

le_att = LabelEncoder()
le_marks = LabelEncoder()
le_assign = LabelEncoder()
le_result = LabelEncoder()

df['Attendance'] = le_att.fit_transform(df['Attendance'])
df['Internal_Marks'] = le_marks.fit_transform(df['Internal_Marks'])
df['Assignment_Submission'] = le_assign.fit_transform(df['Assignment_Submission'])
df['Result'] = le_result.fit_transform(df['Result'])

print("\nEncoded Dataset:")
print(df)

# STEP 4: Train-Test Split

X = df.drop('Result', axis=1)   # Input features
y = df['Result']               # Output label

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
# STEP 5: Train Naive Bayes Model

model = GaussianNB()
model.fit(X_train, y_train)

print("\nNaive Bayes model trained successfully!")
# STEP 6: Prediction and Evaluation

y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# STEP 7: Predict for a new student (CORRECT WAY)

new_student = pd.DataFrame({
    'Attendance': ['Low'],
    'Internal_Marks': ['Poor'],
    'Assignment_Submission': ['No']
})

# ONLY transform (NO fit_transform)
new_student['Attendance'] = le_att.transform(new_student['Attendance'])
new_student['Internal_Marks'] = le_marks.transform(new_student['Internal_Marks'])
new_student['Assignment_Submission'] = le_assign.transform(new_student['Assignment_Submission'])

prediction = model.predict(new_student)

print("\nPrediction for new student:")

if prediction[0] == le_result.transform(['High Risk'])[0]:
    print("Student is at HIGH RISK 🚨")
elif prediction[0] == le_result.transform(['Low Risk'])[0]:
    print("Student is at LOW RISK 😎")
else:
    print("Student is at MEDIUM RISK ⚠️")
