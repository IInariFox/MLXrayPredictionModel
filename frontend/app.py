# frontend/app.py

from flask import Flask, render_template, request
import requests

app = Flask(__name__)

# Replace with the URL where your FastAPI is running
API_URL = "http://localhost:8000/predict"

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            files = {'file': (file.filename, file.stream, file.mimetype)}
            response = requests.post(API_URL, files=files)
            if response.status_code == 200:
                data = response.json()
                prediction = f"{data['prediction']} (Confidence: {data['confidence']:.2f})"
            else:
                prediction = "Error in prediction"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

