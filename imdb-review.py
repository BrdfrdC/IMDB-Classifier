import time
import numpy as np
import pandas as pd

from nltk.corpus import stopwords

from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

# Import Data
start = time.time()
print("Creating datasets...")
    # Get only the columns we need (review_detail and rating)
df = pd.read_json("sample.json").drop(columns=["review_id", "reviewer", "movie", "review_summary", "review_date", "spoiler_tag", "helpful"])
    # Remove null rating rows
df = df[df["rating"].notna()]

dfOne = df[df.rating == 1.0]
dfFive = df[df.rating == 5.0]
dfEight = df[df.rating == 8.0]
dfSeven = df[df.rating == 7.0]
dfTen = df[df.rating == 10.0]

dfBinary = pd.concat([dfOne, dfTen])
dfTertiary =pd.concat([dfBinary, dfFive])
dfMid = pd.concat([dfSeven, dfEight])

print("Distribution of ratings:\n", df.groupby("rating").nunique())

# Preprocessing
print("Preprocessing...")
    # Remove stop words and punctuation, make all words lowercase
stop = stopwords.words("english")
X = df["review_detail"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)])).str.lower().str.replace(r'[^\w\s]+', '', regex = True)
XBinary = dfBinary["review_detail"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)])).str.lower().str.replace(r'[^\w\s]+', '', regex = True)
XTertiary = dfTertiary["review_detail"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)])).str.lower().str.replace(r'[^\w\s]+', '', regex = True)
XMid = dfMid["review_detail"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)])).str.lower().str.replace(r'[^\w\s]+', '', regex = True)

    # Vectorize words and weigh them based on tf-idf
vectorizer = TfidfVectorizer(max_features=10000)
vectorizer.fit(X)
vectorizer.fit(XBinary)
vectorizer.fit(XTertiary)
vectorizer.fit(XMid)

X = vectorizer.transform(X)
XBinary = vectorizer.transform(XBinary)
XTertiary = vectorizer.transform(XTertiary)
XMid = vectorizer.transform(XMid)

y = df["rating"]
yBinary = dfBinary["rating"]
yTertiary = dfTertiary["rating"]
yMid = dfMid["rating"]

# Creating Training and Testing Sets
print("Splitting datasets...")
    # Spliting data
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = 0.2)
XBTrain, XBTest, yBTrain, yBTest = train_test_split(XBinary, yBinary, test_size = 0.2)
XTTrain, XTTest, yTTrain, yTTest = train_test_split(XTertiary, yTertiary, test_size = 0.2)
XMTrain, XMTest, yMTrain, yMTest = train_test_split(XMid, yMid, test_size = 0.2)

print("Normalizing Data...")
    # Scaling and Normalizing Data
scaler = MaxAbsScaler()
XTrain = scaler.fit_transform(XTrain)
XTest = scaler.fit_transform(XTest)
XBTrain = scaler.fit_transform(XBTrain)
XBTest = scaler.fit_transform(XBTest)
XTTrain = scaler.fit_transform(XTTrain)
XTTest = scaler.fit_transform(XTTest)
XMTrain = scaler.fit_transform(XMTrain)
XMTest = scaler.fit_transform(XMTest)

# Model Creation

    # Training Model
print("Training...")

svc = LinearSVC(C = 0.1)
clf = BaggingClassifier(estimator = svc, n_estimators = 11)
clf.fit(XTrain, yTrain)
svcB = LinearSVC(C = 0.1)
clfB = BaggingClassifier(estimator = svc, n_estimators = 3)
clfB.fit(XBTrain, yBTrain)
svcT = LinearSVC(C = 0.1)
clfT = BaggingClassifier(estimator = svc, n_estimators = 4)
clfT.fit(XTTrain, yTTrain)
svcM = LinearSVC(C = 0.1)
clfM = BaggingClassifier(estimator = svc, n_estimators = 3)
clfM.fit(XMTrain, yMTrain)

    # Testing Model
print("Testing...")

predictions = clf.predict(XTest)
predictionsB = clfB.predict(XBTest)
predictionsT = clfT.predict(XTTest)
predictionsM = clfM.predict(XMTest)

print("#### ALL RATINGS #####")
print(classification_report(yTest, predictions))
print(confusion_matrix(yTest, predictions))
sns.heatmap(confusion_matrix(yTest, predictions), annot=True)
plt.show()
print(" ")
print("#### ONE AND TEN #####")
print(classification_report(yBTest, predictionsB))
print(confusion_matrix(yBTest, predictionsB))
sns.heatmap(confusion_matrix(yBTest, predictionsB), annot=True)
plt.show()
print(" ")
print("# ONE, FIVE, AND TEN #")
print(classification_report(yTTest, predictionsT))
print(confusion_matrix(yTTest, predictionsT))
sns.heatmap(confusion_matrix(yTTest, predictionsT), annot=True)
plt.show()
print(" ")
print("### SEVEN AND EIGHT ###")
print(classification_report(yMTest, predictionsM))
print(confusion_matrix(yMTest, predictionsM))
sns.heatmap(confusion_matrix(yMTest, predictionsM), annot=True)
plt.show()
print(" ")
print("Time taken: %s seconds" % (time.time() - start))