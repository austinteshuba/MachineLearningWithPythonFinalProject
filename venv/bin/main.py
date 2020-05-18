# The purpose of this assignment is to use
# empirical data about a series of borrowers
# to build a model that will accurately predict
# if a borrower will default on a new loan.
# This is a part of the Machine Learning with Python course by IBM
# While they did provide the dataset,
# The analysis, processing, and analysis of data below is completed on my own.

import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import matplotlib.ticker as ticker
from sklearn import preprocessing
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

# Step One: Load the Data
path = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv"

df = pd.read_csv(path)
print(df.head())
print(df["effective_date"].value_counts())

# Step Two: Pre-Processing

# Convert dates to date-time objects
df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
print(df.head())

# Let's convert this to day of the week loan is acquired
# This is more helpful than arbitrary due dates
df['dayofweek'] = df['effective_date'].dt.dayofweek

# We can visualize the relationship between day of week and loan default
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()

# Looks like there is a decent correlation if the loan was taken out on a weekend,
# Add this to the feature set
df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
print(df.head())

# Change gender to numeric, binary values where male is 0 and female is 1.
df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
print(df.head())

# One-hot encoding

# Let's look at education
print(df.groupby(['education'])['loan_status'].value_counts(normalize=True))
print(df["education"].value_counts())

Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
# It's reasonable to assume that if you have a master's degree, you have a bachelors
# While dropping the column is a good intuition, since
# Having a master's is not correlated to paying off or collections,
# it's actually because there's only two samples of masters degrees.
Feature["Bechalor"][Feature['Master or Above'] == 1] = 1
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()
# Looks like the reassignment worked

# Get our feature sets
X = Feature
y = df['loan_status'].values

# Step 3: Normalize and train-test-split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

X_train= preprocessing.StandardScaler().fit(X_train).transform(X_train.astype(float))
X_test = preprocessing.StandardScaler().fit(X_test).transform(X_test.astype(float))

# Step 4: KNN

max_K = 10  # We are only going to check the accuracy up to k=10 (this is a high val considering n=346)

meanAccuracyKnn = np.zeros((max_K - 1))
meanStdDev = np.zeros((max_K - 1))
for k in range(1, max_K):
    neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    yhatKnn = neigh.predict(X_test)
    meanAccuracyKnn[k - 1] = metrics.accuracy_score(y_test, yhatKnn)  # only looking at out-of-sample
    # This is the Root Mean Square Error. Shows the range of error in the graph
    meanStdDev[k - 1] = np.std(yhatKnn == y_test) / np.sqrt(yhatKnn.shape[0])

plt.plot(range(1, max_K), meanAccuracyKnn, 'g')
plt.fill_between(range(1, max_K), meanAccuracyKnn - 1 * meanStdDev, meanAccuracyKnn + 1 * meanStdDev, alpha=0.1)
plt.ylabel("Accuracy")
plt.xlabel("Number of Neighbours (i.e. K)")
plt.show()

print("The best k value is ", meanAccuracyKnn.argmax() + 1)

# Based on this, we can create a final model.

neigh = KNeighborsClassifier(n_neighbors = meanAccuracyKnn.argmax()+1).fit(X_train, y_train)
yhatKnn = neigh.predict(X_test)
accuracyKnn = metrics.accuracy_score(yhatKnn, y_test)
print(accuracyKnn)

# Step 5: Let's look at Decision Tree
# I just manually selected max-depth after some trial and error
tree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
tree.fit(X_train, y_train)
yhat_decisionTree = tree.predict(X_test)
accuracy_decisionTree = metrics.accuracy_score(y_test, yhat_decisionTree)
print(accuracy_decisionTree)

# Step 6: Let's look at SVM (a.k.a. SVC)

# We are going to try the different kernels and see how they perform
# let rbf be -1, sigmoid be 0, 1 and onwards will be degrees
def SVC(degree, X_train, X_test, y_train, y_test):
    if degree == -1:
        clf = svm.SVC(kernel='rbf')
    elif degree == 0:
        clf = svm.SVC(kernel="sigmoid")
    else:
        clf = svm.SVC(kernel="poly", degree = degree)
    clf.fit(X_train, y_train)
    yhatsvc = clf.predict(X_test)
    return (metrics.accuracy_score(y_test, yhatsvc), clf, yhatsvc)


# Let's find the best kernal
maxDegree = 10
meanAccuracySVC = np.zeros((maxDegree + 1))
for k in range(-1, maxDegree - 2):
    meanAccuracySVC[k + 1] = SVC(k, X_train, X_test, y_train, y_test)[0]

plt.plot(range(-1, maxDegree), meanAccuracySVC, 'g')
plt.ylabel("Accuracy")
plt.xlabel("Degree of Kernel (-1->rbf, 0-> sigmoid)")
plt.show()
print("Best degree = ", meanAccuracySVC.argmax() - 1)

# Based on that, we can create the final model
accuracy_svc, clf, yhat_svc = SVC(meanAccuracySVC.argmax()-1, X_train, X_test, y_train, y_test)
print(accuracy_svc)

# Step 7: Logistic Regression
LR = LogisticRegression(C=0.01, solver="liblinear").fit(X_train, y_train)
yhatLR = LR.predict(X_test)
yhatLR_prob = LR.predict_proba(X_test)
accuracyLR = metrics.accuracy_score(yhatLR, y_test)
print(accuracyLR)


# Step 8: Evaluation against test-set

path = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv"
test_df = pd.read_csv(path)

# Test set pre-proccessing.
# The test set has to be in the same format as the training data

test_df['dayofweek'] = pd.to_datetime(test_df['effective_date']).dt.dayofweek

test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])
test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x > 3) else 0)

test_df['Gender'].replace(to_replace=['male', 'female'], value=[0, 1], inplace=True)

feature_df = test_df[['Principal', 'terms', 'age', 'Gender', 'weekend']]

feature_df = pd.concat([feature_df, pd.get_dummies(test_df['education'])], axis=1)
feature_df["Bechalor"][feature_df['Master or Above'] == 1] = 1

feature_df.drop(['Master or Above'], axis=1, inplace=True)

feature_df = preprocessing.StandardScaler().fit(feature_df).transform(feature_df.astype(float))

y = test_df["loan_status"]

# Looks like everything matches the above.
# Create the results array
evaluation_df = pd.DataFrame("NA", index=[0,1,2,3], columns=["Algorithm", "Jaccard", "F1-score","LogLoss"])
evaluation_df["Algorithm"] = ["KNN", "Decision Tree", "SVM", "LogisticRegression"]
evaluation_df.head()

# Populate KNN
from sklearn.metrics import f1_score
yhat_KNN_finalTest = neigh.predict(feature_df)

evaluation_df["Jaccard"][0] = metrics.accuracy_score(y, yhat_KNN_finalTest)
evaluation_df["F1-score"][0] = f1_score(y, yhat_KNN_finalTest, labels=["PAIDOFF", "COLLECTION"], average="weighted")


# Populate Decision Tree
yhat_DecisionTree_finalTest = tree.predict(feature_df)

evaluation_df["Jaccard"][1] = metrics.accuracy_score(y, yhat_DecisionTree_finalTest)
evaluation_df["F1-score"][1] = f1_score(y, yhat_DecisionTree_finalTest, labels=["PAIDOFF", "COLLECTION"], average="weighted")

# Populate SVM
yhat_SVM_finalTest = clf.predict(feature_df)
evaluation_df["Jaccard"][2] = metrics.accuracy_score(y, yhat_SVM_finalTest)
evaluation_df["F1-score"][2] = f1_score(y, yhat_SVM_finalTest, labels=["PAIDOFF", "COLLECTION"], average="weighted")

# Populate Logistic Regression
yhat_LR_finalTest = LR.predict(feature_df)
yhat_LR_prob = LR.predict_proba(feature_df)
evaluation_df["Jaccard"][3] = metrics.accuracy_score(y, yhat_LR_finalTest)
evaluation_df["F1-score"][3] = f1_score(y, yhat_LR_finalTest, labels=["PAIDOFF", "COLLECTION"], average="weighted")
evaluation_df["LogLoss"][3] = log_loss(y, yhat_LR_prob)

print(evaluation_df)

# So I would at this point try to optimize further, but I don't know
# How much more accurate I can make this.
# This is for two major reasons:
# 1) Limited sample size. There's only a few hundred samples, so there's not much to train a model with
# 2) No financial data. Practically, loan default is a financial phenomenon. While
# These empiracal facts can be part of a risk profile, accuracy will severly suffer
# without the inclusion of any financial data. After all, there are unemployed university students,
# and millionaire high school drop outs. Clearly, education is not the primary factor here for default.

# It's not a great model, but it's the best you're going to get!
# Not so bad considering we only have qualitative data about each borrower.







