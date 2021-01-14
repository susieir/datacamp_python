"""Exploratory Data Analysis in Python"""

"""Chapter 1 - Read, Clean and Validate"""
# Dataframes and series
# Qu: What is the average birth weight of babies in the US?

#  - Find the appropriate data
#  - Read data in your development environment
#  - Clean and validate

# Reading data

import pandas as pd
nsfg = pd.read_hdf('nsfg.hdf5', 'nsfg')
type(nsfg)

nsfg.shape  # Attribute, no of rows and columns
nsfg.columns  # List of variable names

# Clean and validate
pounds.value_counts().sort_index()  # Can see what values appear and sort by value
pounds.describe()  # Computes summary statistics

# Replace
pounds = pounds.replace([98,99], np.nan, in_place=True)  # Replace lists of values
ounces.replace([98,99], np.nan, in_place=True)  # Replace lists of values, without making a copy (don't need to reassign)

# Combine into single series
birth_weight = pounds + ounces / 16.0

# Filter and visualise
# Pyplot doesn't work with nans, so have to dropna
import matplotlib as plt
plt.hist(birth_weight.dropna(), bins=30)

preterm = nsfg['prglngth'] < 37  # Returns boolean

# Filtering
preterm_weight = birth_weight[preterm]  # Can use ~ for not

# Can use logical operators to combine two boolean series
# & = and
# | = or

birth_weight[A & B]  # Both true
birth_weight[A | B]  # either or both true

# Resampling
# Some groups may be 'oversampled'
# We can correct by using: resample_rows_weighted()

# Probability mass functions
# GSS - General Social Survey

educ = gss['educ']

# PMF - probabiity mass function
# Contains unique values in series and how often they appear
pmf_educ = Pmf(educ, normalize=False)  # If normalise=True, frequencies add up to one
pmf_educ.head()  # Shows values on left and counts on right
pmf_educ[12]  # Look up 12 years education
# Bar chart of pmf better than hist - shows all unique values

# Cumulative distribution functions
# Represents possible values in a distribution and their probabilities
# From PMF to CDF
#  - If you draw a random element from a distribution:
#  -   PMF is the probability that you get exactly x
#  -   CDF is the probability that you get a value less than or equal to x
# Substitute Cdf for Pmf

# CDF is an invertible function

cdf = Cdf(gss['educ'])

p = 0.25
q = cdf.inverse(p)
print(q)

# - IQR - based on percentiles, doesn't get thrown off by extreme values or outliers, unlike variance
# Sometimes more robust than variance

# Comparing distributions
# Can plot multiple PMFs on the same axes
sex = gss['sex'] == 1
age = gss['age']
male_age = age[male]
female_age = age[~male]
Pmf(male_age).plot(label='Male')
Pmf(female_age).plot(label='Female')
plt.xlabel('Age (years)')
plt.ylabel('Count')
plt.show()

# Multiple CDFs
Cdf(male_age).plot(label='Male')
Cdf(female_age).plot(label='Female')
plt.xlabel('Age (years)')
plt.ylabel('Count')
plt.show()
# CDFs are smoother in general - can give a better view of real differences in distributions

# Income distribution
income = gss['realinc']
pre95 = gss['year'] < 1995
Pmf(income[pre95]).plot(label="Before 1995")
Pmf(income[~pre95]).plot(label="After 1995")

# CDF will give clearer picture - good for exploratory analysis

# Modeling distributions
# The normal distribution
sample = np.random.normal(size=1000)
Cdf(sample).plot()
# Produces sigmoid shaped distribution

# Scipy provides object called norm that represents the normal distribution
from scipy.stats import norm
xs = np.linspace(-3, 3)  # Creates an array of equally spaced points from -3 to 3
ys = norm(0, 1).\  # Creates an object the represents a normal distribution with mean 0 and std 1
    cdf(xs)  # Evaluates the CDF of the normal distribution

plt.plot(xs, ys, color='gray')
Cdf(sample).plot

# Want to compare the CDF to a normal distribution to see whether its a good fit

# The bell curve
xs = np.linspace(-3, 3)
ys = norm(0, 1).pdf(xs)
plt.plot(xs, ys, color='gray')

# Comparing the bell curve vs. the pmf won't work

# Kernel density estimation - Can use the points in the sample to estimate the PDF of the distribution they came from
# Getting from pmf (probability mass function) to pdf (probability density function)

# KDE plot
import seaborn as sns
sns.kdeplot(sample)
# Can compare the KDE plot and the normal pdf
xs = np.linspace(-3, 3)
ys = norm.pdf(xs)
plt.plot(xs, ys, color='gray')
sns.kdeplot(sample)

# PDF is a more sensitive way to look for differences, but often too sensitive

# Use CDFs for exploration - good for distracting from noise, but not well known
# Use PMFs if there are a small number of unique values
# Use KDE if there are a lot of values

# In many datasets, distribution of income is lognormal - i.e. log of incomes fits a normal distribution

"""Chapter 3 - Relationships"""

# Exploring relationships
# BRFSS - Behavioural Risk Factor Surveillance System - CDC data
# Use random sub-sample of 100,000

# Scatter plot
brfss = pd.read_hdf('brfss.hdf5', 'brfss')
height = brfss['HTM4']
weight = brfss['WTKG3']
# Faster to use plot with the format string 'o'
plt.plot(height, weight, 'o')  # Plots a circle for each dp
plt.xlabel('Height in cm')
plt.ylabel('Weight in kg')
plt.show()

# Overplotted - where datapoints are piled on top of each other - misleading results
# Can improve with transparency - alpha value
plt.plot(height, weight, 'o', alpha=0.02)  # Plots a circle for each dp
plt.xlabel('Height in cm')
plt.ylabel('Weight in kg')
plt.show()

# Can make markers smaller
plt.plot(height, weight, 'o', alpha=0.02, markersize=1)  # Reduce marker size
plt.xlabel('Height in cm')
plt.ylabel('Weight in kg')
plt.show()

# Can add random noise to values = jittering
height_jitter = height + np.random.normal(0, 2, size=len(brfss))
weight_jitter = weight + np.random.normal(0, 2, size=len(brfss))
plt.plot(height_jitter, weight_jitter, 'o', markersize=1, alpha=0.02)
plt.show()

# Zoom
plt.plot(height_jitter, weight_jitter, 'o', markersize=1, alpha=0.02)
plt.axis([140, 200, 0, 160])  # x range, y range
plt.show()

# Visualising relationships
# Violin plot - estimates the KDE plot for each column (within each group)
# Need to get rid of missing data before you can use it

data = brfss.dropna(subset=['AGE'],['WTKG3'])
sns.violinplot(x='AGE', y='WTKG3', data=data, inner=None)  # Inner = None Simplifies the plot slightly
plt.show()

# Each column - graphical representation of the distribution of weight in one age group
# Width - proportional to the estimated density - two vertical PDFs printed back to back

# Box plot
sns.boxplot(x='AGE', y='WTKG3', data=data, whis=10)  # Whis = 10, turns of feature don't need
plt.show()

# Each box - IQR
# Middle line - median
# Spine - min / max

# With data skewed towards higher values - sometimes useful to look at on logarithmic scale
# Can use pyplot function yscale
sns.boxplot(x='AGE', y='WTKG3', data=data, whis=10)  # Whis = 10, turns of feature don't need
plt.yscale('log')
plt.show()

# Correlation
# Correlation coefficient - -1 to 1 - quantifies strength of a linear relationship
# .corr() - result: correlation matrix
# If correl is non-linear, .corr() will generally underestimate the strength of the relationship

# Generate fake data
xs = np.linspace(-1, 1)  #Equally spaced points
ys = xs**2
ys += normal(0, 0.05, len(xs))  # x^2 + random noise

# Correl says nothing about slope
# Correl - can use one to predict the other
# Statistic we care about - the slop of the line

# Simple regression
# Strength of effect

from scipy.stats import linregress

# Hypothetical 1
res = linregress(xs, ys)
# Result - lin regress result object
# Slope - slope of the line of best fit
# Intercept - intercept
# rvalue - correlation

# Plotting the line of best fit - only works for linear relationships
fx = np.array([xs.min(), xs.max()])  # Take the min and max
fy = res.intercept + res.slope * fx
plt.plot(fx, fy, '-')

"""Chapter 4 - Multivariate thinking"""

# Limits of simple regression
# Regression is not symmetric
# Different because make different assumptions
# x = known quantity
# y = random

# Regression is not causation

# Multiple regression
# Scipy doesn't do multiple regression
# Switch to statsmodels

import statsmodels.formula.api as smf

results = smf.ols('INCOME2 ~ _VEGESU1', data=brfss)\  # First arg - formula string, income as a function of veggie cons
    .fit()  # Run .fit() to get results
results.params  # Contains slope and intercept

# Multiple regression
gss = pd.read_hdf('gss.hdf5', 'gss')
results = smf.ols('realinc ~ educ', data=gss).fit()  # realinc - trying to predict, using educ
results.params

# Adding age
gss = pd.read_hdf('gss.hdf5', 'gss')
results = smf.ols('realinc ~ educ + age', data=gss).fit()  # realinc - trying to predict, using educ and age
results.params

# Income and age
grouped = gss.groupby('age')  # Result - group by object, one group for each value of age
# Behaves like a dataframe

mean_income_by_age = grouped['realinc'].mean()  # Pandas series with mean income for each age group

plt.plot(mean_income_by_age, 'o', alpha=0.5)
plt.xlabel('Age (years)')
plt.ylabel('Income (1986 $)')
# Age and income have a non-linear relationship

# Adding a quadratic term
gss['age2'] = gss['age']**2

model = smf.ols('realinc ~ educ + age + age2', data=gss)
results = model.fit()
results.params

# Visualising regression results
# Generating preductions

df = pd.DataFrame()
df['age'] = np.linspace(18, 85)
df['age2'] = df['age'] ** 2

df['educ'] = 12
df['educ2'] = df['educ'] ** 2

pred12 = results.predict(df)  # Use results to predict average income for each age group holding education constant
# Result - series, one prediction for each row
plt.plot(df['age'], pred12, label="High school")
plt.plot(mean_income_by_age, '0', alpha=0.5)  # Plot of comparison data, avg income in each age group

plt.xlabel('Age (years)')
plt.ylabel('Income (1986 $)')
plt.legend()

# Can repeat for different levels of eduction, e.g. Associates degree, Batchelors degree
# Can help validate the model, can compare predictions against the data

# Logistic regression
# Categorical variables - e.g. sex, race
# Including as part of a regression - "C" indicates categorical variable

formula = 'realinc ~ educ + educ2 + age + age2 + C(sex)'
results = smf.old(formula, data=gss).fit()
results.params
# For cat variable - indicates the difference between the default and the other variable
# If only two values - boolean variable
# Variables need to be recoded so that 1 means yes and 0 means no
# E.g.
gss['gunlaw'].replace([2], [0], inplace=True)

# Logistic regression
formula = 'gunlaw ~ age + age2 + educ + educ2 + C(sex)'
results = smf.logit(formula, data=gss).fit()
# Params are in the form of log odds
# >0, make outcome more likely
# <0, make outcome less likely

# Generate predictions
df = pd.DataFrame()
df['age'] = np.linspace(18, 89)
df['educ'] = 12

df['age2'] = df['age'] ** 2
df['educ2'] = df['educ'] ** 2

df['sex'] = 1 ## Generates predictions for men
pred1 = results.predict(df)

df['sex'] = 0 ## Generates predictions for women
pred2 = results.predict(df)

# Visualising results
grouped = gss.groupby('age')
favor_by_age = grouped['gunlaw'].mean()
plt.plot(favor_by_age, 'o', alpha=0.5)

plt.plot(df['age'], pred1, label='Male')
plt.plot(df['age'], pred2, label='Female')

plt.xlabel('Age')
plt.ylabel('Probability of favoring gun law')
plt.legend()







