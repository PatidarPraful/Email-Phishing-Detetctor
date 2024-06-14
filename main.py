

import pandas as pd
import csv
import re
from urllib.parse import urlparse


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


# Function to read the dataset and extract features
def process_dataset(dataset_path):
    dataset = pd.read_csv(dataset_path)
    extracted_data = []
    for text, label in zip(dataset['text'], dataset['target']):
        features = extract_features(text)
        features['target'] = label  # Include the label in the features
        extracted_data.append(features)
    return extracted_data

# Function to save extracted features to a new CSV
def save_features_to_csv(features, output_path):
    # Get the field names from the first feature dictionary
    fieldnames = features[0].keys()

    # Update the field names to include all features present in any data point
    all_fieldnames = set()
    for feat in features:
        all_fieldnames.update(feat.keys())

    # Open the CSV file for writing
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=all_fieldnames)
        writer.writeheader()

        # Write the features to the CSV file
        writer.writerows(features)



# Main function
def main():
    # Path to the labeled dataset CSV file
    dataset_path = "spam_assassin.csv"
    # Path for saving the extracted features CSV file
    output_path = "extracted_features1.csv"

    # Process the dataset and extract features
    extracted_features = process_dataset(dataset_path)

    # Save extracted features to CSV
    save_features_to_csv(extracted_features, output_path)


if __name__ == "__main__":
    main()
