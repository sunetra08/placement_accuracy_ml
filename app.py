from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the pickled model
with open('placement_prediction.pkl', 'rb') as file:
    logistic = pickle.load(file)


@app.route('/')
def home():
    return render_template('placement.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    # Get user input from the form
    gender = request.form['gender']
    ssc_p = float(request.form['ssc_p'])
    hsc_p = float(request.form['hsc_p'])
    degree_p = float(request.form['degree_p'])
    mba_p = float(request.form['mba_p'])
    etest_p = float(request.form['etest_p'])

    # Create a new data point using the user input
    new_data = [[gender, ssc_p, hsc_p, degree_p, mba_p, etest_p]]

    # Predict the status for the new data point using the loaded model
    prediction = logistic.predict(new_data)

    return render_template('placement.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
