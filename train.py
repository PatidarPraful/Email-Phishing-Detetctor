# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score
#
# # Load the extracted features CSV file
# df = pd.read_csv("extracted_features.csv")
#
# # Split the dataset into features (X) and the target variable (y)
# X = df.drop(columns=['target'])
# y = df['target']
#
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Define the pipeline for preprocessing and training
# pipeline = Pipeline([
#     ('imputer', SimpleImputer(strategy='median')),  # Use median to impute missing values
#     ('classifier', GaussianNB())  # Initialize the classifier
# ])
#
# # Train the pipeline (including imputation and classifier)
# pipeline.fit(X_train, y_train)
#
# # Make predictions on the testing set
# pred = pipeline.predict(X_test)
#
# # Evaluate the performance of the classifier
# accuracy = accuracy_score(y_test, pred)
# print("Classifier Accuracy:", accuracy)

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.impute import SimpleImputer
#
# # Load the extracted features CSV file
# df = pd.read_csv("extracted_features1.csv")
#
# # Handle missing values by imputing with the median value of each column
# imputer = SimpleImputer(strategy='median')
# df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
#
# # Split the dataset into features (X) and the target variable (y)
# X = df_imputed.drop(columns=['target'])
# y = df_imputed['target']
#
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Initialize classifiers
# nb_classifier = GaussianNB()
# svm_classifier = SVC()
# rf_classifier = RandomForestClassifier()
#
# # Train classifiers
# nb_classifier.fit(X_train, y_train)
# svm_classifier.fit(X_train, y_train)
# rf_classifier.fit(X_train, y_train)
#
# # Make predictions on the testing set using each classifier
# nb_pred = nb_classifier.predict(X_test)
# svm_pred = svm_classifier.predict(X_test)
# rf_pred = rf_classifier.predict(X_test)
#
# # Ensemble prediction: Take the majority vote
# ensemble_pred = []
# for nb, svm, rf in zip(nb_pred, svm_pred, rf_pred):
#     majority_vote = sum([nb, svm, rf]) >= 2  # Majority vote
#     ensemble_pred.append(majority_vote)
#
# # Evaluate the performance of the ensemble classifier
# ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
# print("Ensemble Classifier Accuracy:", ensemble_accuracy)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Load the extracted features CSV file
df = pd.read_csv("extracted_features1.csv")

# Handle missing values by imputing with the median value of each column
imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Split the dataset into features (X) and the target variable (y)
X = df_imputed.drop(columns=['target'])
y = df_imputed['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers
nb_classifier = GaussianNB()
svm_classifier = SVC()
rf_classifier = RandomForestClassifier()

# Train classifiers
nb_classifier.fit(X_train, y_train)
svm_classifier.fit(X_train, y_train)
rf_classifier.fit(X_train, y_train)

# Make predictions on the testing set using each classifier
nb_pred = nb_classifier.predict(X_test)
svm_pred = svm_classifier.predict(X_test)
rf_pred = rf_classifier.predict(X_test)

# Ensemble prediction: Take the majority vote
ensemble_pred = []
for nb, svm, rf in zip(nb_pred, svm_pred, rf_pred):
    majority_vote = sum([nb, svm, rf]) >= 2  # Majority vote
    ensemble_pred.append(majority_vote)

# Evaluation metrics
ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
precision = precision_score(y_test, ensemble_pred)
recall = recall_score(y_test, ensemble_pred)
f1 = f1_score(y_test, ensemble_pred)
conf_matrix = confusion_matrix(y_test, ensemble_pred)

# Individual classifier metrics
nb_accuracy = accuracy_score(y_test, nb_pred)
nb_precision = precision_score(y_test, nb_pred)
nb_recall = recall_score(y_test, nb_pred)
nb_f1 = f1_score(y_test, nb_pred)

svm_accuracy = accuracy_score(y_test, svm_pred)
svm_precision = precision_score(y_test, svm_pred)
svm_recall = recall_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred)

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

# Print individual classifier metrics
print("\nIndividual Classifier Metrics:")
print("\nNaive Bayes Classifier:")
print("Accuracy:", nb_accuracy)
print("Precision:", nb_precision)
print("Recall:", nb_recall)
print("F1-score:", nb_f1)

print("\nSupport Vector Machine Classifier:")
print("Accuracy:", svm_accuracy)
print("Precision:", svm_precision)
print("Recall:", svm_recall)
print("F1-score:", svm_f1)

print("\nRandom Forest Classifier:")
print("Accuracy:", rf_accuracy)
print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("F1-score:", rf_f1)


print("\nEnsemble Classifier Metrics:")
print("Accuracy:", ensemble_accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Confusion Matrix:\n", conf_matrix)

# Accuracy values for individual classifiers
classifier_names = ['Naive Bayes', 'SVM', 'Random Forest', 'Ensemble']
accuracy_values = [nb_accuracy, svm_accuracy, rf_accuracy, ensemble_accuracy]

# Plotting the bar graph
# plt.figure(figsize=(10, 6))
# plt.bar(classifier_names, accuracy_values, color=['blue', 'orange', 'green', 'red'])
# plt.xlabel('Classifier')
# plt.ylabel('Accuracy')
# plt.title('Accuracy of Different Classifiers')
# plt.ylim(0, 1)  # Set the y-axis limit to better visualize accuracy values
# plt.show()