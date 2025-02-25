# Spam Detector

This project implements a spam detection system using various machine learning algorithms, including Gaussian Naive Bayes, Support Vector Classifier (SVC), and a Voting Classifier. After experimenting with multiple models, it was found that the Multinomial Naive Bayes (MultinomialNB) algorithm yielded the best results for this task.

## Overview

Spam detection is crucial for filtering unwanted emails and messages. This project aims to classify messages as either spam or not spam using different classifiers and comparing their performance.

## Algorithms Used

1. **Gaussian Naive Bayes**: A probabilistic classifier based on Bayes' theorem, assuming independence among predictors.
2. **Support Vector Classifier (SVC)**: A powerful classifier that works well for both linear and non-linear data by finding the optimal hyperplane.
3. **Voting Classifier**: Combines multiple classifiers to improve the overall accuracy. In this project, it aggregates the predictions from the Gaussian Naive Bayes and SVC classifiers.

### Best Performing Model

After evaluating the performance of the classifiers, it was determined that the **Multinomial Naive Bayes** classifier provided the highest accuracy for spam detection. This model is particularly effective for text classification tasks due to its ability to handle discrete features.

## Installation

To run this project, you need to have the following Python libraries installed:

- `scikit-learn`
- `numpy`
- `pandas`

You can install the required libraries using pip:

```bash
pip install scikit-learn numpy pandas
Usage
Load your dataset containing labeled messages (spam and not spam).
Preprocess the text data (e.g., tokenization, vectorization).
Train the classifiers using the training set.
Evaluate the models using the testing set and compare their performance.
Example Code
Here’s a basic example of how to implement the classifiers:

Copy
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Load dataset
data = pd.read_csv('spam_data.csv')
X = data['text']  # Features
y = data['label']  # Labels

# Preprocessing steps go here...

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers
gnb = GaussianNB()
svc = SVC(probability=True)
mnb = MultinomialNB()

# Voting Classifier
voting_clf = VotingClassifier(estimators=[('gnb', gnb), ('svc', svc)], voting='soft')

# Fit the model
voting_clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = voting_clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Check MultinomialNB performance
mnb.fit(X_train, y_train)
mnb_pred = mnb.predict(X_test)
print("MultinomialNB Accuracy:", accuracy_score(y_test, mnb_pred))
Conclusion
This spam detector project showcases the effectiveness of various classifiers in identifying spam messages. The results indicate that the Multinomial Naive Bayes classifier is the most reliable for this task. Further improvements can be made by exploring additional preprocessing techniques and tuning the model parameters.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
Scikit-learn Documentation for comprehensive guides on machine learning algorithms.