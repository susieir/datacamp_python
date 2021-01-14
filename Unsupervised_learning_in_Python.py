""" Unsupervised learning in Python"""

"""Chapter 1 - Clustering for data exploration"""
# Finds patterns in data
# E.g. clustering customers by their purchases
# Compressing the data using purchase patterns (dimension reduction)

# Supervised learning finds patterns for a prediction task
# E.g. classifying tumours as benign or cancerous (labels)
# Unsupervised learning finds patterns in data
# ...But without a specific prediction task in mind

# Iris dataset
# 2D NumPy array
# Cols as measurements (features)
# Rows represent iris plants (the samples)
# Iris - 4D space
# Dimension = number of features

# k-means clustering
# Finds clusters of samples
# Number of clusters must be specified
# Implemented in scikit-learn

from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(samples)
labels = model.predict(samples)  # Returns a cluster label for each sample

# Cluster labels for new samples
# New samples can be assigned to existing clusters
# k-means remembers the mean of each cluster (the "centroids")
# Finds the nearest centroid to the sample

new_labels = model.predict(new_samples)

# Scatter plots
# Each point represents a sample, colored by label

import matplotlib.pyplot as plt
xs = samples[:,0]  # Sepal length
ys = samples[:,2]  # Petal length
plt.scatter(xs, ys, c=labels)
plt.show()

# Evaluating and clustering
# Can check correspondence with e.g. iris species
# ...but what if there are no species to check against?
# Measure quality of a clustering
# Informs choice of how many clusters to look for

# Cross tabulation with pandas
# e.g. cluster vs. species
# Use the pandas library
# Given the species of each sample as a list 'species'

import pandas as pd
df = pd.DataFrame({'labels': labels, 'species' : species})
ct = pd.crosstab(df['labels'], df['species'])

# Mostly, samples are not labelled
# Measuring clustering quality
# Using only samples and cluster labels
# A good clustering has tight clusters
# Samples in each cluster bunched together

# Intertia measures clustering quality
# Measures how spread out clusters are (lower is better)
# Distance from each sample to centroid of its cluster
# After fit(), available as attribute inertia_
# k-means attempts to minimize the intertia when choosing clusters


from sklearn.cluster import KMeans

model = KMeans(n_clusters=3)
model.fit(samples)
print(model.inertia_)

# The number of clusters
# More clusters - lower inertia
# A good clustering has tight clusters (low inertia)
# But not too many!
# Choose an "elbow" in the inertia plot
# Where inertia begins to decrease more slowly

# Transforming features for better clusters
# Piedmont wines dataset
# Clustering the wines

from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
labels = model.fit(samples)

df = pd.DataFrame({'labels' : labels,
                   'varieties' : varieties})
ct = pd.crosstab(df['labels'], df['varieties'])
print(ct)

# Not a good fit
# Wine features have very different variances
# Variance measures spread of its values

# StandardScaler
# In kmeans: feature variance = feature influence
# StandardScaler transforms each feature to have mean 0 and variance 1
# Features are said to be 'standardised'

# sklearn StandardScaler

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(samples)
StandardScaler(copy=True, with_mean=True, with_std=True)
samples_scaled = scaler.transform(samples)

# Similar methods
# APIs of StandardScaler and KMeans are similar
# Use fit() / transform() with StandardScaler
# Use fit() / predict() with KMeans

# StandardScaler, then KMeans
# Need to perform two steps: StandardScaler then KMeans
# Use sklearn pipeline to combine multiple steps
# Data flows from one step to next

# Pipelines combine multiple steps
from sklearn.preprocessing imoprt StandardScaler
from sklearn.cluster import KMeans
scaler = StandardScaler()
kmeans = KMeans(n_clusters=3)
from sklearn.pipeline import make_pipeline
pipeline = make_pipeline(scaler, kmeans)
pipeline.fit(samples)
labels=pipeline.predict(samples)

# Feature standardization improves clustering

# sklearn preprocessing steps
# StandardScaler is a "preprocessing" step
# MaxAbsScaler and Normalizer are other examples
# Normalizer - rescales each sample independently, to create unit norm

""" Chapter 2 - Visulaization with hierarchical clustering and t-SNE"""

# Visualizing hierarchies
# Visualizations communicate insight
# 't-SNE': Creates a 2D map of a dataset

# A hierarchy of groups
# Groups of living things can form a hierarchy
# E.g. annimals -> mammals / reptiles
# Clusters are contained in one another

# Eurovision scoring dataset
# Countries gave scores to songs performed at Eurovision in 2016
# 2D array of scores
# Rows are countries, columns are songs

# Hierarchical clustering can be visualised as a tree like diag - dendrogram
# Every country begins in a separate cluster
# At each step, the two closest clusters are merged
# Continue until all countries in a single cluster
# This is 'agglomerative' hierarchical clustering

# Divisive clustering - works the other way round

# Dendrogram
# Read from the bottom up
# Vertical lines represent clusters

# Hierarchical clustering with SciPy
# Given 'samples' (the array of scores) and 'country_names'
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
mergings = linkage(samples, method='complete')
dendrogram(mergings,
           labels=country_names,
           leaf_rotation=90,
           leaf_font_size=6)
plt.show()

# Cluster labels in hierarchical clustering
# Not only a visualization tool!
# Cluster labels at any intermediate stage can be recovered
# For use in e.g. cross-tabulations

# Intermediate clusterings & height on dendrogram
# E.g. at height 15:
#  - Bulgaria, Cyprus and Greece
#  - Russia & Moldova
#  - Armenia is in a cluster on its own

# Dendrograms show cluster differences
# Height on dendrogram = distance between merging clusters
# E.g. clusters with only Cyprus and Greece had distance approx. 6
# This new cluster distance approx. 12 from cluster with only Bulgaria
# Height on dendrogram specifies max distance between merging clusters
# Don't merge clusters further apart than this (e.g. 15)
# Defined by a 'linkage method'
# In a 'complete' linkage, distance between clusters is a max. distance between their samples
# Specified via method parameter e.g. linkage(sample, method='complete')
# Different linkage methods - different hierarchical clustering!

# Extracting cluster labels
# Use the fcluster() function
# Returns a NumPy array of cluster labels

from scipy.cluster.hierarchy import linkage
mergings=linkage(samples, method='complete')
from scipy.cluster.hierarchy import fcluster
labels = fcluster(mergings, 15, criterion='distance')  # 15 = height
print(labels)

# Aligning cluster labels with country names
# Given a list of strings country_names:
import pandas as pd
pairs = pd.DataFrame({'labels' : labels, 'countries' : country_names})
print(pairs.sort_values('labels'))  # Labels start at 1 not 0!

# Complete linkage = distance between clusters = distance bet furthest points of clusters
# Simple linkage = distance between clusters = distance bet closest points of clusters

# t-SNE for 2-dimensional maps
# t-SNE = 't-distributed stochastic neighbor embedding'
# Maps samples to 2D space (or 3D), so they can be visualised
# Map approximately preseves nearness of samples
# Great for inspecting datasets

# t-SNE on the iris dataset
# Iris dataset - 4D
# t-SNE maps to 2D space
# t-SNE didn't know there were 3 different species

# Interpreting t-SNE scatter plots
# 'Versicolor' and 'Verginica' harder to distinguish from one-another
# Consistent with k-means inertia plot: could argue for 2 clusters or 3

# t-SNE in sklearn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
model = TSNE(learning_rate=100)
transformed = model.fit_transform(samples)
xs = transformed[:,0]
ys = transformed[:,1]
plt.scatter(xs, ys, c=species)
plt.show()

# tSNE only has a fit_transform() method
# Simultaneously fits the model and transforms the data
# Has no separate fit() or transform() methods
# Can't extend the map to include new data samples
# Must start over each time!

# tSNE learning rate
# Choose learning rate for the dataset
# Wrong choice: points bunch together
# Try a few values between 50 and 200

# Axis - no interpretable meaning, different every time, even on same data
# ... however the clusters have the same position relative to each other

""" Chapter 3 - Decorrelating your data and dimension reduction"""

# Visualising the PCA transformation

# Dimension reduction
# Finds patterns in data and re-expresses it in a reduced form
# More efficient storage and computation
# Remove less informative 'noisy' features
# ...which can cause problems for prediction tasks, e.g. classification, regression

# Principal Component Analysis (PCA)
# Fundamental dimension reduction technique
# First step - decorrelation - doesn't change the dimension of the data
# Second step - reduces dimension

# PCA aligns data with axes
# Rotates data samples to be aligned with axes
# Shifts data samples so that they have mean 0
# No information is lost

# PCA follows the fit/transform pattern
# PCA is a scikit-learn component like KMeans or StandardScaler
# fit() learns the transformation from given data
# transform() applies the learned transformation
# transform() can be applied to new and unseen data

from sklearn.decomposition import PCA
model = PCA()  # Create object
model.fit(samples)  # Fit model to samples
transformed = model.transform(samples)  # Use model to transform samples

# PCA features
# One row for each transformed sample
# Columns correspond to "PCA features"
# Row gives PCA feature values of corresponding sample

# PCA features are not correlated
# Features of dataset are often correlated
# PCA aligns the data with the axes
# Resulting PCA features are not linearly correlated ('decorrelation')

# Pearson correlation
# Measures linear correlation of features : -1 to 1
# 0 - no linear correlation

# Principal components
# "Principal components" - directions of variance
# PCA aligns principal components with the axes
# After model fit, available as components_ attribute of PCA project
# NumPy array with one row for each principal component
# Each row defines displacement from mean

# Intrinsic dimension
# 2 features - latitude and longitude, e.g. flight paths of aircraft
# Dataset appears to be 2D
# But can be closely approximated using only one feature. Displacement along flight path
# Dataset is intrinsically 1 dimensional
# Intrinsic dimension = number of features needed to approximate the dataset
# Essential idea behind dimension reduction
# What is the most compact representation of the samples?
# Can be detected with PCA

# Versicolor dataset - one of the Iris species
# 3 features - sepal length, sepal width and petal width
# Samples are points in 3D space
# Samples lie close to a flat 2D sheet
# So can be approx using 2 features
# intrinsic dimension 2
# PCA identifies intrinsic dimension when samples have any number of features
# Intrinsic dimension = number of PCA features with significant variance

# Plotting the variances of PCA features
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(samples)
features = range(pca.n_components_)

plt.bar(features, pca.explained_variance_)
plt.xticks(features)
plt.yticks('variance')
plt.xlabel('PCA feature')
plt.show()

# Intrinsic dimension can be ambiguous
# Intrinsic dimension is an idealization
# ... there is not always one correct answer!

# First principal component - direction in which the data varies most

# Dimension reduction with PCA
# Represents the same data, using less features
# Important part of machine-learning pipelines
# Can be performed using PCA
# PCA features are in decreasing order of variance
# Assumes the low variance features are "noise"
# ...and high variance features are informative
# Need to specify how many PCA features to keep, e.g. n_components=2
# Intrinsic dimension is a good choice

# Dimension reduction of Iris dataset
# samples = array of iris measurements (4 features)
# species = list of iris species numbers

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(samples)
transformed = pca.transform(samples)

import matplotlib.pyplot as plt
xs = transformed[:,0]
ys = transformed[:,1]
plt.scatter(xs, ys, c=species)
plt.show()

# PCA has reduced the dimensions to 2
# Retained the 2 PCA features with the highest variance
# Important information preserved - species remain distinct

# Word frequency arrays
# Rows represent documents, columns represent words
# Entries measure presence of each word in each document
# ...measure using "tf-idf" (more later)

# Sparse arrays and csr_matrix
# "Sparse": most entries are 0
# Can use scipy.sparse.csr_matrix instead of NumPy array
# csr_matrix - remembers only non-zero entries (saves space!)

# TruncatedSVD and csr_matrix
# scikit-learn PCA doesn't support csr_matrix
# Use scikit-learn TruncatedSVD instead
# Performs same transformation as PCA

from sklearn.decomposition import TruncatedSVD
model = TruncatedSVD(n_components=3)
model.fit(documents)  # documents is csr_matrix
TruncatedSVD(algorithm='randomized', ...)
transformed = model.transform(documents)

""" Chapter 4 - Discovering Interpretable Features"""

# Non-negative matrix factorization (NMF)
# Dimension reduction technique
# NMF models are interpretable (unlike PCA)
# Easy to interpret means easy to explain!
# Cannot be applied to every dataset
# All sample features must be >= 0

# Interpretable parts
# NMF expresses documents as combinations of topics (or "themes")
# Decomposes documents as combinations of the common themes
# Decomposes images as combinations of patterns

# Using scikit-learn NMF
# Follows fit()/transform() pattern
# Must specify number of components (e.g. n_components=2)
# Works with NumPy arrays and with csr_matrix

# Example word-frequency array
# Word frequency array, 4 words, many documents
# Measure presence of words in each document using "tf-idf"
#  - "tf" - frequency of word in document, e.g. if 10% words are "datacamp", frequency is 0.1
#  - "idf" - weighting scheme that reduces the influence of frequent words, like "the"

from sklearn.decomposition import NMF
model = NMF(n_components=2)
model.fit(samples)
nmf_features = model.transform(samples)

# NMF Components
# NMF has components, just like PCA has principal components
# Dimension of components = dimension of samples
# Entries are non-negative
print(model.components_)

# NMF Features
# NMF feature values are non-negative
# Can be used to re-construct the samples
# ...combine feature values with components
print(nmf_features)

# Reconstruction of a sample
# nmf_features * model.components_ == Reconstruction of sample
# Multiply components by feature values and add up
# Can also be expressed as a product of matrices
# This is the "matrix factorisation" in "NMF"

# NMF examples:
# Word frequencies in docs
# Images encoded as arrays
# Audio spectograms
# Purchase histories on ecommerce sites

# NMF learns interpretable parts
# Components of NMF represent patterns that frequently occur in samples
# E.g. Word-frequency array articles (tf-idf)
# 20,000 articles
# 800 words

# Applying NMF to articles
from sklearn.decomposition import NMF
nmf = NMF(n_components=10)
nmf.fit(articles)

# Row/components live in a 800D space
# 1D for each word
# Aligning words of vocab with columns of nmf.components_ allows them to be interpreted

# NMF components
# For documents:
#  - NMF components represent topics
#  - NMF features combine topics into documents
# For images, NMF components are parts of images (frequently occurring patterns)

# Grayscale images - no colors, only shades of gray
# Encoded using the brightness of every pixel
# Represent with value between 0 and 1 (0 is black)
# Can convert to a 2D array of numbers

# Greyscale images as flat arrays
# Enumerate the entries
# Row-by-row
# From left to right, top to bottom

# Encoding a collection of images
# Collection of images of the same size - encode as a 2D array
# Each row represents an image
# Each col represents a pixel
# Data is arranged similarly to the word-frequency array
# ...can apply NMF!

bitmap = sample.reshape((2,3)) # Specify shape as tuple

# To represent as an image
from matplotlib import pyplot as plt
plt.imshow(bitmap, cmap='gray', interpolation='nearest')
plt.show()

# Building recommender systems using NMF
# Task: recommend articles similar to article being read by customer
# Similar articles should have similar topics

# Strategy
# Apply NMF to word-frequency array
# NMF features describe topics
# So similar articles will have similar features
# Compare NMF features values?

# Apply NMF to word frequency array
# 'articles' is a word frequency array
from sklearn.decomposition import NMF
nmf = NMF(n_components=6)
nmf_features = nmf.fit_transform(articles)
# Creates features for every article, given by cols of new array

# Versions of articles
# Different versions of the same document have same topic proportions
# ...exact features values may be different!
# E.g. because one version uses many meaningless words - reduces topic words overall
# Reduces the values of the NMF features representing the topics
# But on a scatter plot of NMF features, all versions lie on the same line through the origin

# Cosine similarity
# Uses the angles between the lines
# Higher values mean more similar
# Maximum value is 1, when angle is 0 degrees

# Calculating the cosine similarities
from sklearn.preprocessing import normalize
norm_features = normalize(nmf_features)
# if has index 23
current_article = norm_features[23,:]
similarities = norm_features.dot(current_article)
# Result is cosine similarities

# DataFrames and labels
# Label similarities with the article titles, using a DF
# Titles given as list: titles

import pandas as pd
norm_features = normlize(nmf_features)
df = pd.DataFrame(norm_features, index=titles)
current_article = df.loc['Dog bites man']  # Title of current article
similarities = df.dot(current_article)

print(similarities.nlargest())








