from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# ---------- DATA ----------
data = {
    'Attendance': ['High','Low','Medium','High','Low','Medium','High','Medium','Low','High'],
    'Internal_Marks': ['Good','Poor','Average','Good','Poor','Average','Good','Good','Poor','Average'],
    'Assignment_Submission': ['Yes','No','Yes','Yes','No','Yes','Yes','Yes','No','Yes'],
    'Result': ['Low Risk','High Risk','Medium Risk','Low Risk','High Risk',
               'Medium Risk','Low Risk','Low Risk','High Risk','Medium Risk']
}

df = pd.DataFrame(data)

le_att = LabelEncoder()
le_marks = LabelEncoder()
le_assign = LabelEncoder()
le_result = LabelEncoder()

df['Attendance'] = le_att.fit_transform(df['Attendance'])
df['Internal_Marks'] = le_marks.fit_transform(df['Internal_Marks'])
df['Assignment_Submission'] = le_assign.fit_transform(df['Assignment_Submission'])
df['Result'] = le_result.fit_transform(df['Result'])

X = df[['Attendance', 'Internal_Marks', 'Assignment_Submission']]
y = df['Result']

model = GaussianNB()
model.fit(X, y)

# ---------- ROUTES ----------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    att = le_att.transform([data["attendance"]])[0]
    marks = le_marks.transform([data["internal_marks"]])[0]
    assign = le_assign.transform([data["assignment"]])[0]

    prediction = model.predict([[att, marks, assign]])[0]

    if prediction == 0:
        result = "HIGH RISK"
    elif prediction == 1:
        result = "LOW RISK"
    else:
        result = "MEDIUM RISK"

    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)
