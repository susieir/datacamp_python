""" Case Study - budgeting with machine learning in Python"""

""" Chapter 1 - Exploring the raw data"""

# A: Introducing the challenge
# DrivenData - online challenges

# Budgets for schools are huge, complex and not-standardised
#   - Hundreds of hours each year are spent manually labelling
# Goal: Build a machine learning algorithm that can automate the process

# Data
#  - Line item: e.g. "Algebra books for 8th grade students"
#  - Labels: "textbooks", "math", "middle school" --> target variable
# Supervised learning problem
# Want to be able to suggest labels for unlabelled lines

# Over 100 target variables
# Classification problem
# Want to create label suggestions and confidence interval
# Human-loop machine learning system

# Predictions will be probabilities for each label

# B: Exploring the data
# Need a probability for each possible value in each column

# Load and preview data
import pandas as pd
sample_df = pd.read_csv('sample_data.csv')
sample_df.head()

sample_df.info()
sample_df.describe()

# C: Looking at the datatypes

# Encode labels as categories
# ML algorithms work on numbers, not strings
#  - Need a numeric representation of these strings
# Strings can be slow compared to numbers
# Pandas - category dtype encodes categorical data numerically
#  - Can speed up code
# astype(category)

sample_df.label = sample_df.label.astype('category')
# Converts into numeric representations of categories
# To see this, use get_dummies

# Dummary variable encoding
dummies = pd.get_dummies(sample_df[['label']], prefix_sep='_')
# Separates the labels into columns / binary indicator representation

# Lambda functions
# Alternative to def syntax
# Useful for simple, oneline functions
square = lambda x: x * x

square(2)

# Encode labels as categories
# In the sample df we have only one relevant column
# In the budget data, there are multiple cols that need to be made categorical

categorize_label = lambda x: x.astype('category')
sample_df.label = sample_df[['label']].apply(categorize_label, axis=0)
sample_df.info()

# D: How do we measure success?
# Accuracy can be misleading when classes are imbalanced
# Metric used in this function: log loss
#  - It is a loss function
#  - It measures error
#  - Want to minimise the error

# logloss = true_label * log(p) + (1-true_label) * log(1-p)
# better to be less confident than confident and wrong
# Provides a steep penalty for predictions that are both wrong and confident

# Computing log loss with NumPy

import numpy as np


def compute_log_loss(predicted, actual, eps=1e-14):
    """ Computes the logarithmic loss between predicted and
    actual when these are 1D arrays.

    :param predicted: The predicted probabilities as floats between 0-1
    :param actual: The actual binary labels. Either 0 or 1.
    :param eps (optional): log(0) is inf, so we need to offset our predicted values slightly by eps from 0 or 1
    """
    predicted = np.clip(predicted, eps, 1 - eps)   # Clip - sets a max and min value for the elements in an array
    loss = -1 * np.mean(actual * np.log(predicted)
                        + (1 - actual)
                        * np.log(1 - predicted))

    return loss

""" Chapter 2 - Creating a simple first model """

# A: It's time to build a model
# Always a good approach to start with a v. simple model
# Gives a sense of how challenging a problem is
# Many more things can go wrong in complex models
# How much signal can we pull out using basic methods?

# Train basic model on numeric data only
#  - Want to go from raw data to predictions quickly
# Multi-class logistic regression
# - treats each label column independently
# - train classifier on each label separately and use those to predict
# - Format predictions and save to csv
# - Compute log-loss score

# Splitting the multi-class dataset
# Recall - Train-test split
#  - Will not work here
#  - Some labels appear in only small fraction of dataset
#  - may end up with labels in test set that never appear in training set

# Solution - StratifiedShuffleSplit
# - Only works for a single target variable, but we have many
# - Utility function proivded - multilabel_train_test_split() - ensures all classes represented in train and test

# Splitting the data
data_to_train = df[NUMERIC_COLUMNS].fillna(-1000)  # Varible provided - list of column names for numeric cols
# -1000 -> want algorithm to respond to NaN's differently to 0
labels_to_use = pd.get_dummies(df[LABELS]) # Takes categories and produces a binary indicator for targets
X_train, X_test, y_train, y_test = multilabel_train_test_split(data_to_train,
                                                               labels_to_use,
                                                               size=0.2,
                                                               seed=123)

# Training the model
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier  # Lets us treat each col of y independently
clf = OneVsOneClassifier(LogisticRegression()) # Fits a sep classifier for each of the cols
clf.fit(X_train, y_train)

# C - Making predictions
# Predicting on holdout data
holdout = pd.read_csv('HoldoutData.csv', index_col=0)
holdout = holdout[NUMERIC_COLUMNS].fillna(-1000)  # Select just numeric columns and replace NaNs
predictions = clf.predict_proba(holdout)   # Predicts probabilities for each label
# If .predict() was used - output would be 0 or 1
# Log loss penalises for being confident and wrong
# As a result, there would be a worse performance compared to .predict_proba()

# Submitting your predictions as a csv
# Submission - df with column headers and row with probabilities for each column
# All formatting can be done with the pandas to_csv function
# Cols have orig column name separated from value by two '_' (some already contained '_')

# Prediction - array of values, needs to be converted to a df

prediction_df = pd.DataFrame(columns=pd.get_dummies(df[LABEELS],
                                                    prefix_sep='_').columns,  # Separates orig col names from col values
                             index=holdout.index,
                             data=predictions)

prediction_df.to_csv('predictions.csv')
score = score_submission(pred_path='predictions.csv')  # Score_submission - provided

# C: A very brief introduction to NLP
# Natual Language Processing
# Can be text, documents, speech
# First step - tokenization
#  - Splitting a long string into segments
#  - Store segments as list
# E.g. "Natural language processing" split into three tokens: ['Natural','language','processing']

# Tokens and token patterns
# Tokenize on whitespace - split everytime space, tab or return
# Could tokenize on whitespace and punctuation

# Bag of words representation
# Count the number of times a particular token appears
# "Bag of words"
#  - Count the number of times a word was pulled out of a bag
#  - This approach discards information about word order
#  - E.g. "red, not blue" is the same as "blue, not red"

# N-grams - more sophisticated
# Col for each token - 1-gram
# Col for every ordered pair of two words - 2-gram

# D: Representing text numerically
# Bag-of-words
#  - Simple way to represent text in machine learning
#  - Discards information about grammar and word order
#  - Computes frequency of occurrence

# Sckiit-learn tools for bag-of-words
# CountVectorizer()
#  - Tokenizes all strings
#  - Builds a "vocabulary"
#  - Counts the occurrences of each token in the vocab

# Using CountVectorizer() on col of main dataset

from sklearn.feature_extraction.text import CountVectorizer
TOKENS_BASIC = '\\\\S+(?=\\\\s+'  # Regex - does split on white space
df.Program_Description.fillna('', inplace=True)  # Makes sure desc does not have any NaN values, replace with empty str
vec_basic = CountVectorizer(token_pattern=TOKENS_BASIC)  # Create CV object, pass in pattern
vec_basic.fit(df.Program_Description)
msg = 'There are {} tokens in Program_Description if tokens are any non-whitespace'
print(msg.format(len(vec_basic.get_feature_names())))

# Vocab = all tokens appearing in this dataset

""" Chapter 3 - Improving your model"""

# A: Pipelines, feature and text pre-processing
# The pipeline workflow
# Repeatable way to go from raw data to trained model
# Pipeline object takes sequential list of steps
#  - Output of one step is input into next step
# Each step is a tuple with two elements
#  - Name: string
#  - Transform: obj implementing .fit() and .transform()
# Flexible - a step can itself be another pipeline

# Instantiate simple pipeline with one step
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

pl = Pipeline([
    ('clf', OneVsOneClassifier(LogisticRegression()))
])

# Train and test with sample numeric data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    sample_df[['numeric']],
    pd.get_dummies(sample_df['label']),
    random_state=2
)
pl.fit(X_train, y_train)

accuracy = pl.score(X_test, y_test)
print('accuracy on numeric data, no nans: ',accuracy)

# Adding an imputer to deal with NaN values
from sklearn.preprocessing import Imputer
X_train, X_test, y_train, y_test = train_test_split(
    sample_df[['numeric', 'with_missing']],
    pd.get_dummies(sample_df['label']),
    random_state=2
)

pl = Pipeline([
    ('imp', Imputer()),  # Default is to fill with mean of value
    ('clf', OneVsOneClassifier())
])

pl.fit(X_train, y_train)
accuracy = pl.score(X_test, y_test)
print('accuracy on all numeric, incl nans:', accuracy)

# B: Text features and feature unions
# Preprocessing text features
from sklearn.feature_extraction.text import CountVectorizer
X_train, X_test, y_train, y_test = train_test_split(sample_df['text'],
                                                    pd.get_dummies(sample_df['label']), random_state=2)

pl = Pipeline([
    ('vec', CountVectorizer()),
    ('clf', OneVsOneClassifier(LogisticRegression()))
])

pl.fit(X_train, y_train)

accuracy = pl.score(X_test, y_test)
print('accuracy on sample data: ', accuracy)

# Preprocessing multiple dtypes
# Want to use all available features in one pipeline
# Problem
#  - Pipeline steps for numeric and text preprocessing can't follow each other
#  - e.g. Output of CountVectorizer can't be input to Imputer

# Need to separately operate on the text and numeric columns
# Solution
#  - FunctionTransformer()
#  - FeatureUnion()

# FunctionTransformer
# Turns a python function into an object that a scikit-learn pipeline can understand
# Need to write two functions for pipeline preprocessing
#  - Take entire DF, return numeric columns
#  - Take entire DF, return text columns
# Can then preprocess numeric and text data in separate pipelines

X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric', 'with_missing', 'text']],
                                                    pd.get_dummies(sample_df['label']), random_state=2)
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion
# Create two function transformer objects
get_text_data = FunctionTransformer(lambda x: x['text'], validate=False)
get_numeric_data = FunctionTransformer(lambda x: x[['numeric', 'with_missing']], validate=False)
# Allow us to set up separate pipelines that work on these columns only
# Validate - doesn't need to check NaNs or validate

# FeatureUnion object puts two sets of features together as a single array that will be the input to the classifier

# Create separate text and numeric pipelines
numeric_pipeline = Pipeline([
    ('selector', get_numeric_data),
    ('imputer', Imputer())
])

text_pipeline = Pipeline([
    ('selector', get_text_data),
    ('vectorizer', CountVectorizer())
])

# Create overall pipeline - feature union concats, then classifier
pl = Pipeline([
    ('union', FeatureUnion([
        ('numeric', numeric_pipeline),
        ('text', text_pipeline)
    ])),
    ('clf', OneVsOneClassifier(LogisticRegression()))
    ])

# Can call fit() and transform() on this pipeline

# C: Choosing a classification model

# Main dataset - lots of text
LABELS = ['Function', 'Use', 'Sharing', 'Reporting',
          'Student_Type', 'Position_Type', 'Object_Type', 'Pre_K', 'Operating_Status']
NON_LABELS = [c for c in df.columns if c not in LABELS]
len(NON_LABELS) - len(NUMERIC_COLUMNS)
# 14 text columns - want to combine into a single function

# Using pipeline with the main dataset

import numpy as np
import pandas as pd
df = pd.read_csv('TrainingSetSample.csv', index_col=0)
dummy_labels = pd.get_dummies(df[LABELS])
X_train, X_test, y_train, y_test = multilabel_train_test_split(
    df[NON_LABELS], dummy_labels, 0.2
)

get_text_data = FunctionTransformer(combine_text_columns, validate=False)
get_numeric_data = FunctionTransformer(lambda x: x[NUMERIC_COLUMNS], validate=False)

pl = Pipeline([
    ('union', FeatureUnion([
        ('numeric_features', Pipeline([
            ('selector', get_numeric_data),
            ('imputer', Imputer())
        ])),
        ('text_features', Pipeline([
            ('selector', get_text_data),
            ('vectorizer', CountVectorizer())
        ]))
    ])
     ), ('clf', OneVsOneClassifier(LogisticRegression()))
])

plt.fit(X_train, y_train)

# Flexibility of model step
# Can be tried quickly with pipelines
#  - Pipeline preprocessing steps unchanged
#  - Edit the model step in your pipeline
# - Random Forest, Naive Bayes, K-NN

# E.g. Random forest alternative option
from sklearn.ensemble import RandomForestClassifier

pl = Pipeline([
    ('union', FeatureUnion([
        ('numeric_features', Pipeline([
            ('selector', get_numeric_data),
            ('imputer', Imputer())
        ])),
        ('text_features', Pipeline([
            ('selector', get_text_data),
            ('vectorizer', CountVectorizer())
        ]))
    ])
     ), ('clf', OneVsOneClassifier(RandomForestClassifier()))
])

""" Chapter 4 - Learning from the experts"""

# A: Pre-processing
# Text preprocessing

# NLP tricks for text data:
#  - Tokenize on punctuation to avoid hypthens, underscores etc.
#  - Include unigrams and bi-grams in the model to capture information involving multiple tokens, e.g. middle school

# N-grams and tokenization
vec = CountVectorizer(token_pattern=TOKENS_ALPHNUMERIC,
                      ngram_range=(1, 2))
# - Simple changes to CountVectorizer
#  - Alphanumeric tokenization
#  - ngram_range=(1,2) # 1 and 2 gram vectorisation

# B: A stats trick
# Interaction terms
# E.g
#  - English teacher for 2nd grade
#  - 2nd grade - budget for English teacher
# Interaction terms mathematically describe when tokens appear together

from sklearn.preprocessing import PolynomialFeatures

interaction = PolynomialFeatures(degree=2,
                                 interaction_only=True,  # Don't need to multiply by itself
                                 include_bias=False)
# Larger degrees - quickly becomes computationally expensive

interaction.fit_transform(x)
# Adds an interaction column

# Bias allows you to to model situation where when x=0, y<>0

# Sparse interaction features
# CountVectorizer - returns sparse matrix
# PolynomialFeatures - does not support sparse matrices
# We have provided SparseInteractions to work for this problem

SparseInteractions(degree=2).fit_transform(x).toarray()

# C: The Winning Model
# Hashing trick
# Adding new features may cause enormous increase in array size
# Hashing is a way of increasing memory efficiency
# Hashing is a way of increasing memory efficiency
# Takes a token and outputs a hashvalue
# Hash function limits possible outputs, fixing array size
# Some cols will have multiple tokens that map to them
# Should be very little effect on model accuracy

# Want to make array of features as small as possible
#  - Dimensionality reduction
#  - Particularly useful on large datasets

# Implementing the hashing trick in scikit-learn
from sklearn.feature_extraction.text import HashingVectorizer
vec = HashingVectorizer(norm=None,
                        non_negative=True,
                        token_pattern=TOKENS_ALPHANUMERIC,
                        ngram_range=(1,2))

# Always better to see how far you can get with simpler methods
