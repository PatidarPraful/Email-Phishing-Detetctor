import re
import pandas as pd
from flask import Flask, request, jsonify, render_template
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Load the pre-trained ensemble classifier
ensemble_classifier = RandomForestClassifier()  # Assuming you have trained and saved the classifier previously

# Load the extracted features CSV file
df = pd.read_csv("extracted_features1.csv")

# Handle missing values by imputing with the median value of each column
imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Split the dataset into features (X) and the target variable (y)
X = df_imputed.drop(columns=['target'])
y = df_imputed['target']

# Train the ensemble classifier
ensemble_classifier.fit(X, y)

# Function to extract features from the text
def extract_features(text):
    features = {}

    # Features related to body text
    features['body_forms_body_html'] = bool(re.search(r'<form>', text, re.IGNORECASE))
    features['body_noCharacters'] = len(text)
    features['body_noDistinct Words'] = len(set(text.split()))
    function_words = ["and", "but", "if", "or", "because", "when", "what", "who", "which", "how", "where", "whom",
                      "whose", "whether", "why", "that", "since", "until", "after", "before", "although", "though",
                      "even", "as", "if", "once", "while"]
    features['body_noFunctionWords'] = sum(1 for word in text.split() if word.lower() in function_words)
    features['body_noWords'] = len(text.split())
    features['body_richness'] = features['body_noWords'] / features['body_noDistinct Words'] if features[
                                                                                                    'body_noDistinct Words'] > 0 else 0
    features['body_suspension'] = bool(re.search(r'\bsuspension\b', text, re.IGNORECASE))
    features['body_verifyYourAccount'] = bool(re.search(r'\bverify your account\b', text, re.IGNORECASE))

    # Features related to sender's address
    sender_domain = re.search(r'From:.*<(\S+)>', text)
    if sender_domain:
        sender_domain = sender_domain.group(1)
        features['send_noCharacters'] = len(sender_domain)
        features['send_noWords'] = len(sender_domain.split())

    # Initialize modal_domain
    modal_domain = None
    if sender_domain:
        modal_domain = urlparse(sender_domain).netloc

    # Features related to URLs in the text
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    if urls:
        features['url_noLinks'] = len(urls)
        if modal_domain:
            features['url_noExtLinks'] = sum(1 for url in urls if urlparse(url).netloc != modal_domain)
            features['url_noIntLinks'] = sum(1 for url in urls if urlparse(url).netloc == modal_domain)
            features['url_noDomains'] = len(set(urlparse(url).netloc for url in urls))
            features['url_noImgLinks'] = sum(
                1 for url in urls if re.search(r'\b(\.jpg|\.jpeg|\.png|\.gif|\.bmp|\.tiff)\b', url, re.IGNORECASE))
            features['url_noIpAddress'] = sum(
                1 for url in urls if re.match(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', urlparse(url).netloc))
            features['url_nonModalHereLinks'] = sum(1 for url in urls if
                                                    re.search(r'\bhere\b', url, re.IGNORECASE) and urlparse(
                                                        url).netloc != modal_domain)
            features['url_atSymbol'] = sum(1 for url in urls if '@' in url)
            features['url_ipAddress'] = sum(
                1 for url in urls if re.match(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', urlparse(url).netloc))
            features['url_linkText'] = sum(
                1 for url in urls if re.search(r'\b(click|here|login|update terms)\b', url, re.IGNORECASE))
            features['url_max_NoPeriods'] = max(url.count('.') for url in urls)
            features['url_ports'] = sum(1 for url in urls if ':' in urlparse(url).netloc)

    return features

# Function to predict phishing based on extracted features
def predict_phishing(text):
    features = extract_features(text)
    # Create a DataFrame with the same columns as X
    features_df = pd.DataFrame(columns=X.columns, data=[features])
    prediction = ensemble_classifier.predict(features_df)[0]
    return prediction

# @app.route('/')
# def index():
#     return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    # prediction = None
    # if request.method == 'POST':
    #     email_text = request.form['email_text']
    #     prediction = predict_phishing(email_text)
    # return render_template('index.html', prediction=prediction)
    return "hello world"

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.json['email_text']
    prediction = predict_phishing(email_text)
    return jsonify({'prediction': prediction})

# if __name__ == '__main__':
#     app.run(debug=True)
