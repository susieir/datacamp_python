""" Machine learning with tree based models in python """

""" Chapter 1 - Classification and Regression tree (CART)"""

# Decision-Tree for Classification
# Sequence of is-else questions about individual features
# Objective: infer class labels

# Able to capture non-linear relationships between features and labels
# Don't require scaling (e.g. standardisation)

# Breast cancer dataset in 2D
# Tree learns a set of if-else questions with one feature and one split point
# max number of branches separating top from extreme end = maximum depth

# Classification in scikit-learn
# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
# Import train_test_split
from sklearn.model_selection import train_test_split
# Import accuracy_score
from sklearn.metrics import accuracy_score
# Split dataset into 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    stratify=y,  # Train/test set to have same prop of labels
                                                    random_state=1)
# Instantiate dt
dt = DecisionTreeClassifier(max_depth=2, random_state=1)
# Fit dt to the training set
dt.fit(X_train, y_train)

# Predict test set labels
y_pred = dt.predict(X_test)
# Evaluate test-set accuracy
accuracy_score(y_test, y_pred)

# Decision region - region in the feature space where all instances are assigned to one class label
# Decision boundary - surface separating different decision regions
# Decision tree results in rectangular regions in the decision space

# Classification tree learning
# Building blocks of a decisionn-tree
# Decision-tree: data structure consisting of a hierachy of nodes
# Node - question or prediction
# Three kinds of nodes:
#  - Root: no parent node, question giving rise to two children nodes
#  - Internal node: one parent node, question giving rise to two children nodes
#  - Leaf: one parent node, no children nodes --> prediction

# Each leaf - one class label predominant

# Information Gain (IG)
# Nodes of class train grown recursively
# The obtention of an internal node or leaf depends on state of predecessors
# At each node, the tree as the question about one feature (f) at a split point (sp)
# To pick feature and split point - maximises information gain
# Every node maximises information gain after each split
# Parent node impurity minus weighted average impurity of left and right nodes
# Impurity can be measured by:
#  - Gini index
#  - entropy

# Classification-Tree learning
# Nodes are grown recursively - based on the states of its predecessors
# At each node, split the data based on:
#  - feature f and split-spoint sp to maximise IG(node)
#  - If IG(node)=0, decleare the node a leaf (for unconstrained trees)

# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
# Import train_test_split
from sklearn.model_selection import train_test_split
# Import accuracy_score
from sklearn.metrics import accuracy_score
# Split dataset into 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    stratify=y,  # Train/test set to have same prop of labels
                                                    random_state=1)
# Instantiate dt, set 'criterion' to 'gini'
dt = DecisionTreeClassifier(criterion='gini', random_state=1)
# Fit dt to the training set
dt.fit(X_train, y_train)

# Predict test set labels
y_pred = dt.predict(X_test)
# Evaluate test-set accuracy
accuracy_score(y_test, y_pred)

# Generally - entropy and Gini index lead to the same results. Gini is slightly faster to compute
# Gini - default criterion used in DecisionTreeClassifier model of scikit-learn

# Decision tree for regression
# Auto-mpg dataset
# 6 features
# Continuous target variable - mpg
# Predict mpg consumption given 6 features

# mpg / displ (displacement) - negative, non-linear relationship

# Regression-Tree in scikit-learn
# Import DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
# Import train_test_split
from sklearn.model_selection import train_test_split
# Import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as MSE
# Split in to 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=3)
# Instantiate a DecisionTreeRegressor 'dt'
dt = DecisionTreeRegressor(max_depth=4,
                           min_samples_leaf=0.1,  # Impose a stopping condition, each leaf must contain at least 10% training data
                           random_state=3)

# Fit 'dt' to the training-set
dt.fit(X_train, y_train)
# Predict test-set labels
y_pred = dt.predict(X_test)

# Compute test-set MSE
mse_dt = MSE(y_test, y_pred)
# Compute test-set RMSE
rmse_dt = mse_dt ** (1/2)
# Print rmse_dt
print(rmse_dt)

# Information criterion for regression-tree
# Impurity(node) = Mean Square Error(node)
# Impurity of the node is the mean square error of the targets within that node
# Regression tree tries to find splits that reduces the leafs where in each leaf the target values are on average,
# the closest possible to the mean value of the labels in that particular leaf

# Prediction
# Prediction - average of target variables contained in that leaf

""" Chapter 2 - The Bias-Variance Trade-off"""

# Generalization error
# Supervised learning - under the hood
# Make an assumption there's a link f between features and labels
# y=f(x) - f is unknown, to be determined

# Goals of supervised learning
# Find a model f-hat that best approximates f
# f-hat can be Logistic Regression, Decision Tree, Neural Network
# Want to discard as much noise as possible
# End goal - f-hat should achieve a low predictive error on unseen datasets

# Difficulties in approximating f
# Overfitting - f-hat(x) fits the training set noise
# Underfitting - f-hat is not flexible enough to approximate f

# Overfitting
# When a model is overfit, it's predictive power on unseen datasets is pretty low
# Low training set error, high test set error

# Underfitting
# Training set error approx equal to test set error
# However both errors are relatively high
# Relationship not sophisticated enough to capture complexity between features and labels

# Generalisation error
# Generalisation error of f-hat: does f-hat generalise well on unseen data?
# It can be decomposed into three terms:
#  - bias squared
#  - variance
#  - irreducible error - error contribution of noise

# Bias
# Bias - error term that tells you on average how much f-hat and f are different
# High bias models - lead to underfitting

# Variance
# Variance tells you how much f-hat is inconsistent over different training sets
# High variance models - lead to overfitting

# Model complexity
# Model complexity - sets the flexibility of f-hat
# Example - maximum tree depth, minimum samples per leaf

# Best model complexity - minimises generalisation error
# As model complexity increases bias decreases while variance increases
# Irreducible error is constant
# Therefore minimising generalisation error is a trade-off between bias and variance
# The bias-variance trade-off

# Diagnosing bias and variance problems

# Estimating the generalisation error
# Cannot be done directly because:
#  - f is unknown
#  - usually you only have one dataset
#  - noise is unpredictable

# Solution
#  - Split the data into training and test sets
#  - fit f-hat to the training set
#  - evaluate the error of f-hat on the unseen test set
#  - generalisation error of f-hat approx. equals test set error of f-hat

# Better model evaluation with cross-validation
# Test set must not be touched until we are confident about f-hat's performance
# Evaluating f-hat on training set - biased estimate, f-hat has already seen all training points
# Solution - Cross Validation (CV):
#  - K-fold CV
#  - Hold-out CV
# CV error is the mean of the K obtained errors

# Diagnose variance problems
# If f-hat suffers from high-variance: CV error of f-hat > training set error of f-hat
# f-hat is said to have overfit the training set. To remedy overfitting:
#  - Decrease model complexity
#  - Gather more data

# Diagnose bias problems
# If f-hat suffers from high bias: CV error of f-hat is approx. equal to training set error of f-hat
# But both are greater than the desired error
# f-hat is said to have underfit the training set. To remedy underfitting:
#  - Increase model complexity
#  - Gather more relevant features

# K-fold CV in sklearn on the Auto Dataset
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import cross_val_score

# Set seed for reproducibility
SEED = 123
# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=SEED)

# Instantiate DecisionTree regressor and assign it to 'dt'
dt = DecisionTreeRegressor(max_depth=4,
                           min_samples_leaf=0.14,
                           random_state=SEED)

# Evaluate the list of MSE obtained by 10-fold CV
# Set n_jobs to -1 in order to exploit all CPU cores in computation
MSE_CV = - cross_val_score(dt, X_train, y_train, cv=10,
                           scoring='neg_mean_squared_error',  # CV score does not allow direct comp of MSE
                           n_jobs=-1)  # Exploit all available CPUs

# Fit 'dt' to training set
dt.fit(X_train, y_train)
# Predict labels of training set
y_predict_train = dt.predict(X_train)
# Predict labels of test set
y_predict_test = dt.predict(X_test)

# CV MSE
print('CV MSE: {:.2f}'.format(MSE_CV.mean))
# Training set MSE
print('Train MSE: {:.2f}'.format(MSE(y_train, y_predict_train)))
# Test set MSE
print('Test MSE: {:.2f}'.format(MSE(y_test, y_predict_test)))

# Ensemble learning
# Advantages of CARTs
#  - Simple to understand
#  - Simple to interpret
#  - Easy to use
# Flexibility - ability to describe non-linear relationships
# Preprocessing - no need to standardise or normalise features

# Limiations of CARTs
# Classification - can only produce orthogonal decision boundaries
# Sensitive to small variations in training set
# High variance - unconstrained CARTs may overfit the training set
# Solution - ensemble learning

# Ensemble learning
# Train different models on the same dataset
# Let each model make its predictions
# Meta-model - aggregates the predictions of individual models
# Final prediction - more robust and less prone to errors
# Best results - models are skillful in different ways

# Ensemble learning in practice - voting classifier
# Binary classification task
# N classifiers make predictions: P1, P2, .., PN with P either 0 or 1
# Meta-model prediction: hard-voting

# Hard voting
# The most votes gets it!

# Voting classifier in sklearn
# Import functions to compute accuracy and split data
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Import models, including VotingClassifier meta-model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import VotingClassifier

# Set seed for reproducibility
SEED = 1

# Split data into 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=SEED)

# Instantiate individual classifiers
lr = LogisticRegression(random_state=SEED)
knn = KNN()
dt = DecisionTreeClassifier(random_state=SEED)
# Define a list called classifier that contains the tuples (classifier_name, classifier)
classifiers = [('Logistic Regression', lr),
               ('K Nearest Neighbors', knn),
               ('Classification Tree', dt)]
# Iterate over the defined list of tuples containing the classifiers
for clf_name, clf in classifiers:
    # fit clf to the training set
    clf.fit(X_train, y_train)

    # Predict the labels of the test set
    y_pred = clf.predict(X_test)

    # Evaluate the accuracy of clf on the test set
    print('{:s} : {:.3f}'.format(clf_name, accuracy_score(y_test, y_pred)))

# Instantiate VotingClassifier 'vc'
vc = VotingClassifier(estimators=classifiers)

# Fit 'vc' to the training set and predict test set labels
vc.fit(X_train, y_train)
y_pred = vc.predict(X_test)

# Evaluate the test-set accuracy of 'vc'
print('Voting Classifier: {.3f}'.format(accuracy_score(y_test, y_pred)))

""" Chapter 3 - Bagging and Random Forests"""
# Bootstrap aggregation - bagging - an ensemble method

# Voting classifier - fit to same training set using different alogrithms, predicting using majority voting
# Bagging - one algorithm - different models trained on different subsets of the data
# Reduces variance of individual models in the ensemble
# Bootstrap - sampling with replacement
# Bagging - n different bootstrap samples from the training set
# In classification - the final prediction is obtained by majority voting
# 'BaggingClassifier' in scikit-learn
# In regression - the final prediction is obtained by averaging
# 'BaggingRegressor' in scikit-learn

# Import models and utility functions
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Set seed for reproducibility
SEED = 1

# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    stratify=y,
                                                    random_state=SEED)

# Instantiate a classification tree 'dt'
dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.16, random_state=SEED)
# Instantiate a bagging classifier 'bc'
bc = BaggingClassifier(base_estimator=dt, n_estimators=300, n_jobs=-1)  # 300 classification trees, -1 = all cores used
# Fit 'bc' to the training set
bc.fit(X_train, y_train)
# Predict test set labels
y_pred = bc.predict(X_test)

# Evaluate and print test-set accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy of Bagging Classifier: {:.3f}'.format(accuracy))

# Out of bag evaluation
# Some instances may be sampled several times for one model
# Other instances may not be sampled at all

# Out of bag (OOB) instances
# On average, 63% of the training instances are sampled
# The remaining 37% constitute the OOB instances
# Can be used to estimate the performance of the ensemble, without the need for cross validation
# This is OOB evaluation

# OOB evaluation
# For each bootstrap model, it is evaluated on the OOB instances
# The OOB score is then calcluated on an average of the OOB scores for each bootstrap model

# OOB evaluation in sklearn
# Import models and split utility function
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Set seed for reproducibility
SEED = 1

# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    stratify=y,
                                                    random_state=SEED)

# Instantiate a classification tree 'dt'
dt = DecisionTreeClassifier(max_depth=4,
                            min_samples_leaf=0.16,
                            random_state=SEED)

# Instantiate a bagging classifier 'bc'; set oob_score = True
bc = BaggingClassifier(base_estimator=dt, n_estimators=300,
                       oob_score=True, n_jobs=-1)  # Evalutes the OOB accuracy of bc after training
# OOB_score - accuracy score for classifiers, R-squared for regressors

# Fit 'bc' to the training set
bc.fit(X_train, y_train)

# Predict the test set labels
y_pred = bc.predict(X_test)

# Evaluate test set accuracy
test_accuracy = accuracy_score(y_test, y_pred)
# Extract the OOB accuracy from 'bc'
oob_accuracy = bc.oob_score_

# Print test set accuracy
print('Test set accuracy: {:.3f}'.format(test_accuracy))
# Print OOB accuracy
print('OOB accuracy: {:.3f}'.format(oob_accuracy))

# Random forests

# Bagging
# Base estimator - can be any model, e.g. Decision Tree, Logistic Regression, Neural Network
# Each estimator is trained on a distinct bootstrap sample of the training set
# Estimators use all features for training and evaluation

# Further diversity with random forests
# Base estimator - decision tree
# Each estimator is trained on a different bootstrap sample having the same size as the training set
# RF introduces further randomisation than bagging in the training of individual trees
# d features are sampled at each node without replacement (d<total features)

# Random forests - training
# Each tree forming ensemble trained on a different bootstrap sample from the training set
# In addition, when a tree is trained, at each node only d features are sampled from all features without replacement
# The node is then split using the sampled feature that maximises information gain
# In sklearn, d defaults to the square root of the number of features
# E.g. If 100 features, 10 are sampled at each node
# Once trained, preductions can be made on new instances
# When a new instance is fed to the different base estimators, each outputs a prediction
# The predictions are then collected by the RandomForests metamodel and a final prediction is made
# For classification
#  - Prediction made by majority voting - RandomForestClassifier in scikit learn
# For regression
#  - Aggregates predictions through averaging - RandomForestRegressor in scikit learn

# In general random forests achieve a lower variance than individual trees

# Random forests regressor in sklearn
# Basic imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
# Set seed for reproducibility
SEED = 1

# Split dataset into 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=SEED)

# Instantiate a random forests regressor 'rf' with 400 estimators
rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=0.12,
                           random_state=SEED)

# Fit 'rf' to the training set
rf.fit(X_train, y_train)
# Predict the test set labels 'y_pred'
y_pred = rf.predict(X_test)

# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred)**(0.5)

# Print the test set RMSE
print('Test set RMSE of rf: {.2f}'format(rmse_test))

# Feature importance
# Tree-based methods: enable measuring the importance of each feature in prediction
# In sklearn:
#  - How much the tree nodes use a particular feature (weighted average) to reduce impurity
#  - Expressed as a percentage
#  - Accessed using feature_importance_

# Feature importance in sklearn
import pandas as pd
import matplotlib.pyplot as plt

# Create a pd.Series of features importances
importances_rf = pd.Series(rf.feature_importances_, index=X.columns)

# Sort importances_rf
sorted_importances_rf = importances_rf.sort_values()

# Make a horizontal bar plot
sorted_importances_rf.plot(kind='barh', color='lightgreen'); plt.show()

""" Chapter 4 - Boosting"""

# AdaBoost
# Boosting - ensemble method in which many predictors  trained and each predictor learns from errors of its predecessor
# Combines many weak learners to form a strong learner
# Weak learner - model doing slightly better than random guessing
# E.g. Decision stump - CART with max depth 1

# Boosting
# Train an ensemble of predictors sequentially
# Each predictor tries to correct its predecessor
# Most popular boosting methods:
#  - AdaBoost
#  - Gradient Boosting

# Adaboost - Adaptive Boosting
# Each prediction pays more attention to the instances wrongly predicted by its predecessor
# Achieved by changing the weights of training instances
# Each predictor - coefficient alpha - determines contribution in the final prediction
# Alpha depends on the predictor's training error
# N predictors in total

# Learning rate (eta) - between 0 and 1
# Used to shrink the coefficient alpha of a trained predictor
# Trade-off between eta and the number of estimators
# Smaller eta should be compensated by a greater number of estimators

# AdaBoost - prediction
# Classification
#  - Each predictor predicts the label of the new instance
#  - Weighted majority voting
#  - In sklearn AdaBoostClassifier
# Regression
#  - Weighted average
#  - In sklearn AdaBoostRegressor

# Predictors don't need to be CARTs, but are mostly used

# AdaBoost Classification in sklearn (Breast Cancer dataset)
# Import models and utility functions
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Set seed for reproducibility
SEED = 1

# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    stratify=y,
                                                    random_state=SEED)

# Instantiate a classification-tree 'dt'
dt = DecisionTreeClassifier(max_depth=1, random_state=SEED)

# Instantiate an AdaBoostClassifier 'adb_clf'
adb_clf = AdaBoostClassifier(base_estimator=dt, n_estimators=100)

# Fit adb_clf to the training set
adb_clf.fit(X_train, y_train)

# Predict the test set probabilities of positive class
y_pred_proba = adb_clf.predict_proba(X_test)[:,1]

# Evaluate test roc_auc score
adb_clf_roc_auc_score = roc_auc_score(y_test, y_pred_proba)

# Print adb_clf_roc_auc score
print('ROC AUC score: {:.2f}'.format(adb_clf_roc_auc_score))

# Gradient Boosting (GB)
# Gradient Boosted Trees - sequential correction of predecessor's errors
# Does not tweak the weights of the training instances
# Fit each predictor is trained using its predecessor's residual errors as labels
# Gradient Boosted Trees - CART is used as a base learner
# Important parameter - shrinkage - prediction of each tree is shrinked after multiplication by a learning rate, eta (0 to 1)
# Similar to AdaBoost - trade-off between Eta and the number of estimators
# Decreasing learning rate, needs to be compensated by increasing the number of estimators

# Prediction
# Regression - y_pred = y1 + eta*r1 + eta*r2 + ... + eta*rn
#  - In sklearn GradientBoostingRegressor
# Similar alg used for classification - GradientBoostingClassifier (not discussed here)

# Gradient boosting in sklearn (auto dataset)
# Import models and utility functions
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

# Set seed for reproducibility
SEED = 1

# Split dataset into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=SEED)

# Instantiate a GradientBoostRegressor 'gbt'
gbt = GradientBoostingRegressor(n_estimators=300, max_depth=1, random_state=SEED)

# Fit 'gbt' to the training set
gbt.fit(X_train, y_train)

# Predict the test set labels
y_pred = gbt.predict(X_test)

# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred) ** 0.5
# Print the test set RMSE
print('Test set RMSE: {:.2f}'.format(rmse_test))

# Stochastic gradient boosting (SGB)
# GB involves an exhaustive search procedure
# Each CART is trained to find the best split points and features
# May lead to CARTs using the same split points and maybe the same features

# To mitigate - stochastic GB
# Each tree is trained on a random subset of rows of the training data
# The sampled instances (40-80% of the training set) are sampled without replacement
# Features are sampled (without replacement) when choosing split points
# Result - further ensemble diversity
# Effect - adding further variance to the ensemble of trees

# Stochastic gradient boosting in sklearn (auto dataset)
# Import models and utility functions
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

# Set seed for reproducibility
SEED = 1

# Split dataset into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=SEED)

# Instantiate a stochastic GradientBoostingRegressor 'sgbt'
sgbt = GradientBoostingRegressor(max_depth=1,
                                 subsample=0.8,  # Each tree samples 80% data for training
                                 max_features=0.2,  # Each tree uses 20% of avail features to fit
                                 n_estimators=300,
                                 random_state=SEED)

# Fit 'sgbt' to the training set
sgbt.fit(X_train, y_train)

# Predict the test set labels
y_pred = sgbt.predict(X_test)

# Evaluate test set RMSE
rmse_test = MSE(y_test, y_pred) ** 0.5

# Print 'rmse_test'
print('Test set RMSE: {:.2f}'.format(rmse_test))

""" Chapter 5 - Model Tuning"""

# Tuning a CARTs hyperparameters
# Machine learning model:
#  - Parameters - learned from data, e.g. split-point of a node
#  - Hyperparameters - not learned from data, set prior to training, e.g. max_depth, min_samples_leaf

# What is hyperparameter tuning?
# Problem - search for a set of optimal hyperparameters for a learning algorithm
# Solution - find a set of optimal hyperparameters that results in an optimal model
# Optimal model - yields an optimal score
# Score - in sklearn defaults to accuracy (classification) and R-squared (regression)
# Cross validation is used to estimate the generalization performance

# Why tune hyperparameters?
# In sklearn, a model's default hyperparameters are not optimal for all problems
# Hyperparameters should be tuned to obtain the best model performance

# Approaches to hyperparameter tuning
#  - Grid Search
#  - Random Search
#  - Bayesian optimisation
#  - Genetic algorithms
# ...

# Grid search cross-validation
# Manually set a grid of discrete hyperparameter values
# Set a metric for scoring model performance
# Search exhaustively through the grid
# For each set of hyperparameters, evaluate each model's CV score
# The optimal hyperparameters are those of the model achieving the best CV score
# Suffers from the curse of dimensionality!

# Grid-search cross-validation example
# Hyperparameter grids:
#  - Max_depth: {2, 3, 4}
#  - Min_samples_leaf = {0.05, 0.1}
# Hyperparameter space: {(2,0.05), (2,0.1), (3,0.05), ...}
# CV_scores = {score(2,0.05), ...}
# Optimal hyperparameters - set of hyperparameters corresponding to the best CV score

# Inspecting the hyperparameters of a CART in scikit learn
# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# Set seed to 1 for reproducibility
SEED = 1

# Instantiate a decision tree classifier 'dt'
dt = DecisionTreeClassifier(random_state=SEED)

# Print out 'dt's hyperparameters
print(dt.get_params())  # Prints dict, where keys are hyperparams names
# Max_features - number of features to consider when looking for the best split
# When float - interpreted as percentage

# Import GridSearchCV
from sklearn.model_selection import GridSearchCV
# Define the grid of hyperparameters 'params_dt'
params_dt = {
    'max_depth': [3, 4, 5, 6],
    'min_samples_leaf': [0.04, 0.06, 0.08],
    'max_features': [0.2, 0.4, 0.6, 0.8]
}

# Instantiate a 10-fold CV grid search object 'grid_dt'
grid_dt = GridSearchCV(estimator=dt,
                       param_grid=params_dt,
                       scoring='accuracy',
                       cv=10,
                       n_jobs=-1)
# Fit 'grid_dt' to the training data
grid_dt.fit(X_train, y_train)

# Extract best hyperparams from 'grid_dt'
best_hyperparams = grid_dt.best_params_
print('Best hyperparameters:\n', best_hyperparams)

# Extract best CV score from 'grid_dt'
best_CV_score = grid_dt.best_score_
print('Best CV accuracy'.format(best_CV_score))

# Extract best model from 'grid_dt'
best_model = grid_dt.best_estimator_  # Model fitted on the whole training set, reset param of GridCV set to true by default

# Evaluate test set accuracy
test_acc = best_model.score(X_test, y_test)

# Print test set accuracy
print("Test set accuracy of best model: {:.3f}".format(test_acc))

# Tuning an RF's hyperparameters

# Random forests hyperparameters
# CART hyperparameters
# Number of estimators
# Bootstrap
# ...

# Tuning is computationally expensive and sometimes leads to only slight improvement
# Weight the impact of tuning on the whole project

# Inspecting RF hyperparams in sklearn
# Import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

# Set seed for reproducibility
SEED = 1

# Instantiate a random forests regressor 'rf'
rf = RandomForestRegressor(random_state=SEED)

# Inspect rf's hyperparameters
rf.get_params()

# Basic imports
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import GridSearchCV

# Define a grid of hyperparameter 'params_rf'
params_rf = {
    'n_estimators': [300, 400, 500],
    'max_depth': [4, 6, 8],
    'min_samples_leaf': [0.1, 0.2],
    'max_features': ['log2', 'sqrt']
}

# Instantiate 'grid_rf'
grid_rf = GridSearchCV(estimator=rf,
                       param_grid=params_rf,
                       cv=3,
                       scoring='neg_mean_squared_error',
                       verbose=1,  # The higher its value the more messages are printed during fitting
                       n_jobs=1)

# Fit 'grid_rf' to the training set
grid_rf.fit(X_train, y_train)

# Extract best hyperparams from 'grid_rf'
best_hyperparams = grid_rf.best_params_

print('Best hyperparameters:\n', best_hyperparams)

# Extract best model from 'grid_rf'
best_model = grid_rf.best_estimator_
# Predict the test set labels
y_pred = best_model.predict(X_test)
# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred) ** 0.5
# Print the test set RMSE
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))















