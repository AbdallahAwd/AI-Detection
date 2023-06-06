from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load and preprocess the data
# Replace 'your_dataset.csv' with the actual file path
data = pd.read_csv('dataset.csv')
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['label']

# Train the model
classifier = LogisticRegression()
classifier.fit(X, y)

# Define the prediction endpoint


@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    text_vectorized = vectorizer.transform([text])
    prediction = classifier.predict(text_vectorized)[0]
    if prediction == 'AI':
        score = classifier.predict_proba(text_vectorized)[0][0]
    else:
        score = 1 - classifier.predict_proba(text_vectorized)[0][1]
    response = [
        {
            'label': prediction,
            'score': round(float(score), 4)
        }
    ]
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
