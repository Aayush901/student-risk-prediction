from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Load your pre-trained model
MODEL_PATH = 'student_risk_model.pkl'
if os.path.exists(MODEL_PATH):
    model = pickle.load(open(MODEL_PATH, 'rb'))
else:
    model = None  # For testing without model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Safely get form values
    att = request.form.get('attendance', None)
    marks = request.form.get('marks', None)
    assign = request.form.get('assignment', None)

    if None in [att, marks, assign]:
        return "Form data missing! Check input fields.", 400

    # Convert categorical to numeric (example mapping)
    att_map = {'High': 2, 'Medium': 1, 'Low': 0}
    marks_map = {'Good': 2, 'Average': 1, 'Poor': 0}
    assign_map = {'Yes': 1, 'No': 0}

    input_features = [
        att_map.get(att, 0),
        marks_map.get(marks, 0),
        assign_map.get(assign, 0)
    ]

    # Make prediction
    if model:
        prediction = model.predict([input_features])[0]
    else:
        prediction = "High Risk"  # Dummy for testing

    return render_template('index.html', prediction_text=f"Student Risk: {prediction}")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))  # Render requires PORT variable
    app.run(host='0.0.0.0', port=port, debug=True)
