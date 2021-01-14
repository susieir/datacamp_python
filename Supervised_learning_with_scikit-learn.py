"""Supervised learning with Scikit-learn"""

""" Chapter 1 - Classification"""

# Supervised learning

## What is machine learning
# The art and science of:
#  - Giving computers the ability to learn to make decisions from data
#  - Without being explicitly programmed!

# E.g. learning to predict whether an email is spam or not - predicting class label - spam or not
# E.g. clustering Wikipedia entries into different categories

# Supervised learning - uses labeled data
# Unsupervised learning - uses unlabeled data

# Unsupervised learning
#  - Uncovering hidden patterns from unlabelled data
#  - Example - grouping customers into different categories (clustering)

# Reinforcement learning
# Software agents interact with an environment
#  - Learn how to optimise their behaviour
#  - Given a system of rewards and punishments
#  - Draws inspiration from behavioural psychology
# Applications:
#  - Economics
#  - Genetics
#  - Game playing

# Supervised learning
# Predictor variables / features and a target variable
# Data commonly represented in a table structure
# Aim: predict the target variable, given the predictor variables
#  - Classification - target variable consists of categories
#  - Regression - target variable is continuous
# E.g. Predictor variables = plant measurements
# E.g. Target variable = plant species

# Features = predictor variables = independent variables
# Target variable = dependent variable = response variable

# Supervised learning
# Automate time-consuming or expensive manual tasks
#  - E.g. Doctors diagnosis
# Make predictions about the future
#  - E.g. Will a customer click on an add or not?
# Need labelled data
#  - Historical data - with labels
#  - Experiments to get data - e.g. A/B testing
#  - Crowd-sourced labelling data, e.g. Recaptcha for text recognition

# Supervised learning in Python
# This course uses scikit-learn/sklearn
# Integrates well with the SciPy stack, incl. numpy
# Other libraries: TensorFlow, keras

# Exploratory data analysis
# Features: petal length, petal width, sepal length, sepal width
# Target variable: species - one of: versicolor, virginica, setosa

from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

iris = datasets.load_iris()
print(type(iris))  # Bunch similar to dict - contains key value pairs

print(iris.keys())  # Feature names, desc, target names, data (contains values, features), target - target data

print(type(iris.data))
print(type(iris.target))

print(iris.data.shape)  # 150 rows (samples), 4 cols (features)

# target variable
# 0 for setosa
# 1 for versicolor
# 2 for virginica

print(iris.target_names)

# Exploratory data analysis (EDA)
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)

print(df.head())

_ = pd.plotting.scatter_matrix(df, c=y,  # Color, allows for color by species
                               figsize= [8,8],  # Specifies size of figure
                               s=150,  # Marker size
                               marker='D')  # Marker shape
#plt.show()
# Diag - hist of features
# Off-diag - scatters of col feature vs. row feature, colored by target variable

# The classification challenge
# Already labelled data = Training data

# k-nearest neighbours (kNN)
# Basic idea: predict the label of a data point by
#  - Looking at the 'k' closest labelled data points
#  - Taking a majority vote
# Creates a set of decision boundaries

# Scikit-learn fit and predict
# All machine learning models implemented as python classes
#  - They implement the algorithms for learning and predicting
#  - Store all the information that is learned from the data
# Training a model on the data = 'fitting' a model to the data
#  - .fit() method
# To predict the labels of new data: .predict() method

# Using scikit-learn to fit a classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(iris['data'], iris['target'])  # Apply fit to classifier, args: features, labels (target)
# Returns classifier, and modifies it to fit to the data

# Important notes:
# Data must be a numpy array or pandas df
# Features must take on continuous values
# Requires there are no missing values
# Each col must be a feature, each row an observation
# Target must be single col with same number of obs as feature data

iris['data'].shape  # 150 observations of four features
iris['target'].shape  # 150 labels

# Predicting on unlabelled data
X_new = np.array([[5.6, 2.8, 3.9, 1.1],
                 [5.7, 2.6, 3.8, 1.3],
                 [4.7, 3.2, 1.3, 0.2]])

prediction = knn.predict(X_new)  # Predict method on classifier and parse data
# X_new must be numpy array with features in cols and observations in rows
print(X_new.shape)  # 3 rows, 4 cols

print('Prediction: {}'.format(prediction))

# Measuring model performance
# In classification, accuracy is a commonly used metric
# Accuracy = fraction of correct predictions
# Which data do we use to compute accuracy?
# How well will model perform on new data?

# Could measure accuracy on data used to fit the classifier
# NOT indicative of ability to generalise
# Split data into training set and test set
# Fit/train classifier on the training set
# Make predictions on test set
# Compare predictions with the known labels

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =  # Unpacked into four variables
    train_test_split(X, # Feature data
                     y, # Targets or labels
                     test_size=0.3,  # What prop of orig data used for test set
                     random_state=21, # Seed for random number generator, allows for replication
                     stratify=y)

# Default 75% training, 25% test data - good rule of thumb
# Best practice to perform split so it reflects labels on your data - stratify = y (list containing labels)
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(\"Test set predictions:\\n {}".format(y_pred))

knn.score(X_test, y_test)  # 95% - good for out of the box model

# Model complexity
# Larger k = smoother decision boundary = less complex model
# Smaller k = more complex model = can lead to overfitting

""" Chapter 2 - Introduction to regression"""
# Target - continuous variable

# Boston housing data
boston = pd.read_csv('boston.csv')

# Drop target for feature arrays
X = boston.drop('MEDV', axis=1).values  # Using values, returns numpy array that we will use
y = boston['MEDV'].values

# Predicting house value from a single feature
X_rooms = X[:,5]
type(X_rooms), type(y)  # Both are numpy arrays
# Need to get them into the right shape

y = y.reshape(-1, 1)  # Keeps the first dimension, adds another dimension of size 1 to x
X_rooms = X_rooms.reshape(-1, 1)

# Plotting house value vs. number of rooms
plt.scatter(X_rooms, y)
plt.ylabel('Value of house /1000 ($)')
plt.xlabel('Number of rooms')
plt.show()

# Fitting a regression model
import numpy as np
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_rooms, y)
prediction_space = np.linspace(min(X_rooms),
                               max(X_rooms)).reshape(-1, 1)

plt.scatter(X_rooms, y, color='blue')
plt.plot(prediction_space, reg.predict(prediction_space),
         color='black', linewidth=3)
plt.show()

# The basics of linear regression
# for a line y = ax + b
# Chooses the line that minimises the error function
# Residual - difference between a data point and the line
# Minimising residual - large pos would cancel out large negative resid
# Therefore - minimise the square of the residuals
# Ordinary least squares - minimizes the sum of squares of the residuals
# Same as minimising the mean square errors of the predictions on the training set

# Linear regression with higher dimensions
# Must specify a coefficient for each feature and the variable b
# Scikit learn API works the same way
#  - Pass two arrays - features and target

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3, random_state=42)
reg_all = LinearRegression()  # Instantiate regressor
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)

# R squared - default scoring model for linear regression
# Quantifies the amount of variance in the target variable that is predicted from the feature variables

reg_all.score(X_test, y_test)

# You'd rarely use linear regression out of the box like this
# You'd usually regularise first

# Cross-validation
# Model performance is dependent on way the data is split
# May not be representative of the model's ability to generalise unseen data
# Solution: cross-validation!

# Cross-validation basics
# Split the data into 5 groups (folds)
# Hold the first fold as the test set, then fit out the remaining four folds, predict and compute
# Then hold the second fold as test set and repeat
# And so on...

# As a result, get 5 values of R-squared
# 5 folds = 5-fold Cross-validation (CV)
# 10 folds = 10-fold CV
# k folds = k-fold CV
# More folds = more computationally expensive

# Cross-validation in scikit-learn
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

reg = LinearRegression()  # Instantiate our model
cv_results = cross_val_score(reg,  # Regressor
                             X, y, # Feature and target data
                             cv=5)  # Number of folds
# Returns an array of cross-validation scores
# Score reported is R-squared

# Regularised regression
# Linear regression minimises a loss function
# It chooses a coefficient for each feature variable
# Large coefficients can lead to overfitting
# Alter loss function so penalises for large coefficients - regularization

# Ridge regression
# Loss function = OLS loss function + squared value of each coefficient multiplied by some constant (alpha)
# Models are penalised for coefficients with a large magnitude - positive and negative
# Alpha - parameter we need to choose in order to fit and predict
# Can select the alpha for which the model performs best
# Picking alpha here is similar to picking k in k-NN
# Hyperparameter tuning
# Alpha controls model complexity
## - Alpha = 0, we get back OLS (can lead to overfitting)
## - Alpha = large, can lead to model too simple, underfitting data

from sklearn.linear_model import Ridge
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3, random_state=42)
ridge = Ridge(alpha=0.1, normalize=True)  # Ensures all variables are on the same scale
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
ridge.score(X_test, y_test)

# Lasso regression
# Loss function = OLS loss function + absolute value of each coefficient multiplied by some constant (alpha)

from sklearn.linear_model import Lasso
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3, random_state=42)
lasso = Lasso(alpha=0.1, normalise=True)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
lasso.score(X_test, y_test)

# Lasso regression for feature selection
# Can be used to select important features of a dataset
# Shrinks the coefficient of less important features to be exactly 0

from sklearn.linear_model import Lasso
names = boston.drop('MEDV', axis=1).columns  # Feature names
lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(X, y).coef_  # Extracts coef attribute
_ = plt.plot(range(len(names)), lasso_coef)
_ = plt.xticks(range(len(names)), names, rotation=60)
_ = plt.ylabel('Coefficients')
plt.show()

# Most important predictor is number of rooms

# Ridge should be first choice for regression models
# Lasso = L1 regularisation - regularisation term is L1 norm of the coefficients
# L2 regularisation - L2 norm - Ridge regression

""" Chapter 3 - How good is your model?"""
# Classification metrics - measuring model performance with accuracy:
#  - Fraction of correctly classified samples
#  - Not always a useful metric

# Class imbalance example - emails
# Spam classification - 99% real, 1% spam
# Can build a classifier that predicts ALL emails as real - would be 99% accurate!
# Fails at its original purpose
# Class imbalance - higher proportion of real emails than spam
# More nuanced metrics required to assess model performance

# Diagnosing classification problems
# Confusion matrix
"""
                    Predicted: spam email | Predicted: real email
                    ----------------------------------------------
Actual:spam email  | True positive        | False negative
Actual: real email | False positive       | True negative  
"""
# Usually class of interest - positive class
# Accuracy = (tp + tn) / (tp + fn + fp + fn)

# Metrics from confusion matrix
# Precision = share of true positives = tp / (tp + fp) (also positive predictive value - ppv)
# Recall = tp / (tp + fn) = sensitivity / hit rate / true positive rate
# F1score = 2 x ((precision * recall) / (precision + recall)) = harmonic mean of precision and recall
# High-precision - not many real emails predicted as spam
# High-recall - predicted most spam emails correctly

# Confusion matrix in scikit-learn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
knn = KNeighborsClassifier(n_neighbors=8)
X_train, X_test, y_train ,y_test = train_test_split(X, y,
                                                    test_size=0.4, random_state=42)
knn.fit(X_train, Y_train)
y_pred = knn.predict(X_test)

print(confusion_matrix(y_test, y_pred))  # Prints confusion matrix

print(classification_report(y_test, y_pred))  # Prints precision, recall, f1score, support
# First argument - true label
# Second argument - prediction

# Logistic regression and the ROC curve
# Logistic regression - used in classification problems, not regression problems

# Logistic regression for binary classification
# Logistic regression outputs probabilities
# If p > 0.5:
    # The data is labelled as 1
# If p < 0.5:
    # The data is labelled as 0
# Produces a linear decision boundary

# Logistic regression in scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

# Probability thresholds
# By default, logistic regression threshold = 0.5
# Not specific to logistic regression
#  - kNN classifiers also have thresholds

# What happens if we vary the threshold
# If p=0, model predicts 1 for all data
#  - True pos rate = False pos rate = 1
# If p=1, model predicts 0 for all data
#  - True and false pos rates are 0
# If vary p, get a series of different true and false pos rates
# Set of points when trying all possible p's is called Receiver Operating Characteristic curve (ROC curve)

# Plotting the ROC curve
from sklearn.metrics import roc_curve
y_pred_prob = logreg.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label='Logistic regression')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Logistic Regression ROC curve')
plt.show()

# Need both the prediction on the test set and the probability that logreg model outputs before using threshold
logreg.predict_proba(X_test)[:,1]  # Returns a 2-col array, probs for respective target values
# Choose second column, probabilities of predicted labels being 1

# Precision-recall curve

# Area under the ROC curve (AUC)
# Larger area under the ROC curve = better model
# Maximises the true positive rate and minimises the false positive rate

# AUC in scikit-learn
from sklearn.metrics import roc_auc_score
logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.4, random_state=42)
logreg.fit(X_train, y_train)
y_pred_prob = logreg.predict_proba(X_test)[:,1]
roc_auc_score(y_test, y_pred_prob)

# AUC using cross-validation - alt method
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc')
print(cv_scores)

# Hyperparameter tuning
# Linear regression - choosing parameters
# Ridge/Lasso regression - choosing alpha
# k-Nearest Neighbors - choosing n_neighbors
# Parameters like alpha and k (chosen before using a model) - hyperparameters
# Hyperparameters cannot be explicitly learned by fitting the model

# Choosing the correct hyperparameter
# Try a bunch of different values
# Fit all of them separately
# See how well each performs
# Choose the best performing one
# Hyperparameter tuning
# Essential to use cross-validation - using train, test, fit alone - overfit the parameter to the test set
# Even after cross-validation - want to keep separate a test set to evaluate performance on a dataset it has not seen

# Grid search cross-validation
# Choose parameter(s)
# Create a grid for each possible value of parameter(s)
# Perform k-fold cross validation for each possible combination of parameters
# Then choose combination of hyperparameters that perform the best
# Called a Grid-search

# GridSearchCV in scikit-larn
from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors' : np.arange(1, 50)} # Dict, keys are hyperparam names, values list of possible values
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, # Model
                      param_grid, # Grid to tune over
                      cv=5)  # Number of folds
# Returns a grid search object that can be fitted to the data
knn_cv.fit(X, y)  # Grid search object can be fitted to the data
knn_cv.best_params_  # Returns best parameters
knn_cv.best_score_  # Returns best scores - mean CV score over that fold

# Hyperparameter tuning with GridSearchCV - logistic regression
# Logreg - regularisation parameter, C
# C controls the inverse of regularisation strength
# Large C -> overfit model
# Small C -> underfit model

# Hyperparameter tuning with RandomizedSearchCV
# GridSearchCV can be computationally expensive - especially if dealing with multiple params over a large hyperparam space
# RandomizedSearchCV - not all hyperparam values are tried out
# A fixed number of settings is sampled from specified probability distributions

# Hold-out set for final evaluation
# Hold-out set reasoning
#  - How well can the model perform on never before seen data
#  - Using ALL data for cross-validation is not ideal
#  - Split data into training and hold-out set at the beginning
#  - Perform grid search cross-validation on training set
#  - Choose best hpyerparameters and evaluate on hold-out set

# Elastic net regularization
# Penalty term is a linear combination of the L1 and L2 penalties:
# a*L1 + b*L2
# In scikit-learn this term is represented by the 'l1_ratio' parameter
# An 'l1_ratio' of 1 corresponds to an L1 penalty, and anything lower is a combination of L1 and L2


""" Chapter 4 - Preprocessing and pipelines"""
# Pre-processing data
# Dealing with categorical features
# Scikit-learn API will not acecpt categorical features by default
# Need to encode categorical features numerically
# Need to convert to 'dummy variables', one for each category
#  - 0 - observation was NOT that category
#  - 1 - observation was that category

# E.g. Origin: US, Europe, Asia
# Create binary features for each of the origins: origin_Asia, origin_Europe, origin_US
# Each row will have 1 in just one of the three columns and 0 in the other two
# Do not really need 3 features, only 2 - delete origin_Asia, otherwise carrying duplicate data

# Dealing with categorical features in Python
# scikit-learn: OneHotEncoder()
# pandas: get_dummies()

# E.g. Automobile dataset
# target variable - mpg
# categorical feature - origin

import pandas as pd
df = pd.read_csv('auto.csv')
df_origin = pd.get_dummies(df)  # Will create three columns, need to drop origin_Asia
print(df_origin.head())
df_origin = df_origin.drop('origin_Asia', axis=1)

# Linear regression with dummy varialbes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=3, random_state=42)
ridge = Ridge(alpha=0.5, normalize=True).fit(X_train, y_train)
ridge.score(X_test, y_test)
# can also use drop_first=True argument

# Handling missing data
# If no value for a given feature in a particular row
# Dropping missing data

df.insulin.replace(0, np.nan, inplace=True)  # Replaces 0s with NaNs
df.triceps.replace(0, np.nan, inplace=True)
df.bmi.replace(0, np.nan, inplace=True)

# Dropping missing data
df = df.dropna()
df.shape

# Imputing missing data
# Making an educated guess about missing values
# E.g. using the mean of the non-missing entries

from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN',  # Missing values represented as NaN
              strategy='mean',  # Replaces with mean
              axis=0)  # Impute along columns (1 would mean rows)
imp.fit(X)  # Fit to data
X = imp.transform(X)  # Transform data
# Imputers are known as transformers

# Imputing within a pipeline
# Allows you to transform data and run model at same time

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
logreg = LogisticRegression()  # Instantiate model
# List of steps in tuples
steps = [('imputation', imp),  # Name for step and estimator
         ('logistic_regression', logreg)]
# Pass list to pipeline constructor
pipeline = Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
pipeline.score(X_test, y_test)

# Each step but the last much be a transformer
# Last step must be an estimator

# Centering and scaling
# Many models use some form of distance to inform them
# Features on larger scales can unduly influence the model
# Example - k-NN uses distance explicitly when making predictions
# Want features to be on a similar scale
# Normalizing (or scaling and centering)

# Ways to normalise your data
# Standarization - substract the mean and divide by variance
#  - All features are centered around zero and have variance one
# Can subtract by the minimum and divide by the range
#  - Minimum zero and max one
# Can also normalise so that the data ranges from -1 to +1

# Scaling in scikit-learn
from sklearn.preprocessing import scale
X_scaled = scale(X)
# Can also put scaler in a pipeline object

# Scaling in a pipeline
from sklearn.preprocessing import StandardScaler
steps = [('scaler', StandardScaler()),
          ('knn', KNeighborsClassifier())]
pipeline = Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=21)
knn_scaled = pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
accuracy_score(y_test, y_pred)

knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)
knn_unscaled.score(X_test, y_test)

# CV and scaling in a pipeline
steps = [('scaler', StandardScaler()),
         (('knn', KNeighborsClassifier()))]
pipeline = Pipeline(steps)
parameters = {knn__n_neighbors: np.arange(1, 50)}  # Specify hyperparameter space with dict
# keys are pipeline step name, followed by __, followed by hyperparameter name
# value is a list or array of values to try for hyperparam
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
cv = GridSearchCV(pipeline, param_grid=parameters)  # Instantiating GridSearchCV object
cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
print(cv.best_params_)
print(cv.score(X_test, y_test))
print(classification_report(y_test, y_pred))
