"""Statistical thinking in python"""

"""Chapter 1 - Graphical Exploratory Data Analysis"""

# Introduction to exploratory data analysis (EDA)
# The process of organising, plotting and summarising a data set

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
bin_edges = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # Specifies edges of hist bins
_ = plt.hist(df_swing['dem_share'], bins=bin_edges)  # Returns 3 arrays that not int in, only want plot, therefore assign dummy variable
_ = plt.xlabel('percent of vote for Obama')
_ = plt.ylabel('number of counties')
plt.show()

# Rule for number of bins - square root of sample size

# Plot all of your data - bee swarm plots
# Choice of bins is arbitrary - but impacts the interpretation - can lead to binning bias
# Losing actual values of data
# Bee swarm plot - remedies, no bin bias, all data displayed

# Requires pandas data frame - each col a feature, each row an observation

_ = sns.swarmplot(x='state', y='dem_share', data=df_swing)
_ = plt.xlabel('state')
_ = plt.ylabel('share of vote for Obama')
plt.show()

# Plot all of your data: ECDFs
# Bee swarm - not best option for larger sample sizes
# Instead can compute empirical cumulative distribution function (ECDF)
import numpy as np
x = np.sort(df_swing['dem_share'])
y = np.arange(1, len(x)+1) / len(x)
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.xlabel('percent of vote for Obama')
_ = plt.ylabel('ECDF')
plt.margins(0.02)  # Keeps data off plot edges
plt.show()

# Unpacking, e.g. x, y = foo(data), where a function returns two arrays
# In real life - always start with graphical EDA

"""Chapter 2 - Quantitative Exploratory Data Analysis"""
# Introduction to summary statistics

import numpy as np
np.mean(dem_share_PA)
# Heavily influenced by outliers
# Median - middle value of a dataset - immune to extreme values
np.median(dem_share_PA)

# Percentiles, outliers and box plots
np.percentile(df_swing['dem_share'], [25, 50, 75])  # Outputs an array of percentiles
# Boxplots - great alternative to bee swarm when dataset is large

import matplotlib.pyplot as plt
import seaborn as sns
_ = sns.boxplot(x='east_west', y='dem_share', data=df_all_states)
_ = plt.xlabel('region')
_ = plt.ylabel('percent of vote for Obama')
plt.show()

# Variance and standard deviation

# Variance - the mean squared distance of the data from their mean
# Informally, a measure of spread of the data
# np.var
np.var(dem_share_FL)
# Units are different because it has been squared - therefore interested in square root - standard deviation
np.sqrt(np.var(dem_share_FL))
# or
np.std(dem_share_FL)

# Covariance and Pearson correlation coefficient

# Scatter plots - graphical EDA technique
# Covariance - measure of how two variables move together
#   - Mean of the product of the differences between the distances from the variable means
#   - If both tend to be above or below the mean - positive covariance
#   - If one is above and the other is below - negative covariance
# Ideally want a dimensionless measure (i.e. one with no units)
# Therefore, divide the covariance by the product of the standard deviation of the two variables (i.e. normalising)
# This gives - Pearson Correlation Coefficient
#   - Variability due to covariance / independent variability (inherent variability)
#   - Ranges from -1 to 1

# Defining covariance
# np.cov(x, y)
# Returns an array - covariance matrix
# [0, 1], [1, 0] - Covariances
# [0, 0], [1, 1] - variance of x and y respectively

# Pearson correlation coefficient
# np.corrcoef()
# [0, 0], [1, 1] - Always equal to 1
# [0, 1] - Returns pearson correl coeff

"""Chapter 3 - Thinking Probabilistically - Discrete Variables"""
# Probabilistic logic and statistical inference
# Probabilistic reasoning allows us to describe uncertainty
# Gets us from measured data to probabilistic conclusions about the population

# Random number generators and hacker statistics

# Hacker statistics - uses simulated repeated measurements to compute probabilities
# np.random module - suite of functions based on random number generation
# np.random.random() - draw a number between 0 and 1
# Bernoulli trial - an experiment that has two options, "success" (true) and "failure" (false)

# Random number seed
#  - Integer fed into random number generator
#  - Manually seed random number generator if you want reproducibility
#  - Specified using np.random.seed()

# Simulating four coin flips

import numpy as np
np.random.seed(42)
random_numbers = np.random.random(size=4)  # Size - determines how many random numbers we get
random_numbers

heads = random_numbers < 0.5
heads

np.sum(heads)

# Repeating it and calculating the percentage of the time all heads are returned

n_all_heads = 0  # Initialise number of 4-heads trials
for _ in range(10000):
    heads = np.random.random(size=4) < 0.5
    n_heads = np.sum(heads)
    if n_heads == 4:
        n_all_heads += 1

n_all_heads / 10000

# Hacker stats probabilities
#  - Determine how to simulate data
#  - Simulate many times
#  - Probability is approximately fraction of trials with the outcome of interest

# Probability distribution and stories - the binomial distribution

# Probability Mass Function (PMF)
# The set of probabilities of discrete outcomes

# Discrete uniform PMF - e.g. rolling a die

# Probability distribution
# - mathematical description of outcomes

# Binomial distribution
# - The number r of successes in n Bernoulli trials with probability p of success, is binomially distributed
# - For example - The number of r of heads in 4 coin flips with a probability of 0.5 is binomially distributed

# Sampling from the binomial distribution
np.random.binomial(4, 0.5)  # Single trial

np.random.binomial(4, 0.5, size=10)  # Repeated 10 times

# The binomial PMF
samples = np.random.binomial(60, 0.1, size=10000)
n = 60
p = 0.1

# The binomial CDF
import matplotlib as plt
import seaborn as sns
sns.set()
x, y = ecdf(samples)
_ = plt.plot(x, y, marker='.', linestyle='none')
plt.margins(0.02)
_ = plt.xlabel('number of successes')
_ = plt.ylabel('CDF')
plt.show()

# Poisson processes and the poisson distribution
# Poisson process - the timing of the next event is completely independent of when the previous event happened
# E.g. natural births in a given hospital, hits on a website, aviation incidents

# Poisson distribution - single parameter
# Number of r arrivals in a given time interval with an average rate of ? arrivals per interval is Poisson distributed
# The number r of hits on a given website in one hour with an average rate of 6 hits per hour is Poisson distributed

# Poisson distribution
# Looks similar to binomial distribution
# Limit of the binomial distribution for low probability of success and large number of trials
# That is, for rare events

samples = np.random.poisson(6, size=10000)
x, y = ecdf(samples)
_ = plt.plot(x, y, marker='.', linestyle='none')
plt.margins(0.02)
_ = plt.xlabel('number of successes')
_ = plt.ylabel('CDF')
plt.show()


""" Chapter 4 - Thinking Probabilistically - Continuous Variables"""

# Probability density functions
# Continuous variables - can take any value

# Probability density function (PDF)
# Continuous analog to the PMF
# Mathematical description of a relative likelihood of observing a value of a continuous variable
# Area under the PDF is what is important

# Introduction to the normal distribution
# Describes a single continuous variable whose PDF has a single symmetric peak
# Two parameters - mean (middle of peak), standard deviation (width of peak)

# Computing the theoretical normal distribution
import numpy as np
mean = np.mean(michaelson_speed_of_light)
std = np.std(michaelson_speed_of_light)
samples = np.random.normal(mean, std, size=100000)
x, y = ecdf(michaelson_speed_of_light)
x_theor, y_theor = ecdf(samples)

# Plot theoretical and actual on same plot
sns.set()
_ = plt.plot(x_theor, y_theor)
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.xlabel('speed of light (km/s)')
_ = plt.ylabel('CDF')
plt.show()

# The normal distribution - properties and warnings
# Beware of overapplying it
# Normal distribution - outliers are unlikely
# If there are lots of outliers - normal distribution may not be appropriate

# The exponential distribution
# The waiting time between arrivals of a poisson process is exponentially distributed
# Possible poisson process: nuclear incidents

mean = np.mean(inter_times)
samples = np.random.exponential(mean, size=1000)
x, y = ecdf(inter_times)
x_theor, y_theor = ecdf(samples)
_ = plt.plot(x_theor, y_theor)
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.xlabel('time (days)')
_ = plt.ylabel('CDF')
plt.show()