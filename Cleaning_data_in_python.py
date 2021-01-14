# Cleaning data in python
# Common problems
# Data type constraints

# Why do we need to clean data?
# Human error and technical error can impact data
# Without cleaning it - it will impact the report insights

# Data type constraints
# Python has a set of specific data types for objects
# Need to ensure variables have the correct datatypes
# Strings to integers

sales.dtypes  # Checks data types
sales.info  # Checks data types and numbers of missing values

# Remove $ from Revenue column
sales['Revenue'] = sales['Revenue'].str.strip('$')  # Strips out the $
sales['Revenue'] = sales['Revenue'].astype('int')  # Then convert to integer
# If decimal - would have converted to float

# Verify that Revenue is now an integer
assert sales['Revenue'].dtype == 'int'  # Returns nothing if correct, otherwise AssertionError

# Numeric or categorical? If finite set of categories
# If categorical saved as integer, will produce misleading results when looking at data summaries

# Convert to categorical
df["marriage_stats"] = df["marriage_status"].astype('category')
df.describe()  # Returns more aligned summary statistics

# Data range constraints
# Dealing with out of range data
# Could drop it - could lose essential info - only do it when a small prop of data is out of range
# Set custom mins or maxes to columns
# Can treat as missing and impute
# Set custom value based on business assumptions

# Movie example - movies with ratings higher than 5

import pandas as pd

# Output movies with rating > 5
movies[movies['avg_rating'] > 5]

# Drop values using filtering
movies = movies[movies['avg_rating'] <= 5]  # Creates a new filtered data frame
# Drop values using .drop()
movies.drop(movies[movies['avg_rating'] > 5].index, inplace=True)  # Takes row indices of movies with >5 rating,
# don't need to create new column

# Assert results
assert movies['avg_rating'].max() <= 5

# Convert avg_rating > 5 to 5
movies.loc[movies['avg_rating'] > 5, 'avg_rating'] = 5  # Setting value of avg rating column to 5 if exceeds 5

# Date range example
# Converting date from object to date
# Convert to DateTime, using  datetime
import datetime as dt

user_signups['subscription_date'] = pd.to_datetime(user_signups['subscription_date'])
# Assert that conversion happened
assert user_signups['subscription_date'].dtype == 'datetime64[ns]'  # How dt represented in pandas

today_date = dt.date.today()  # Allows us to store todays date

# Drop dates using filtering
user_signups = user_signups[user_signups['subscription_date'] < today_date]
# Drop values using .drop()
user_signups.drop(user_signups[user_signups['subscription_date'] > today_date].index, inplace=True)

# Hardcode dates with upper limit
# Drop values using filtering
user_signups.loc[user_signups['subscription_date'] > today_date, 'subscription_date'] = today_date
# Assert is true
assert user_signups['subscription_date'].max().date() <= today_date

# Uniqueness constraints
# Duplicate values - the same info repeated across rows/columns
# Why do they happen:
## Data entry and human error
## Bugs and design errors
## Join or merge errors

# Finding duplicate values
# Get duplicates across all columns
duplicates = height_weight.duplicated()
print(duplicates)  # Returns a series of boolean values - true for duplicate values

# Get duplicate rows - all columns are required to have duplicate values by default -
# All marked as true except for first occurrence
duplicates = height_weight.duplicated()
height_weight[duplicates]

# Limits ability to diagnose what kind of duplication we have and how to treat it
# The duplicated method:
## subset - list of column names to check for duplication
## keep - whether to keep first, last or all duplicate values

# Column names to check for duplication
column_names = ['first_name', 'last_name', 'address']
duplicates = height_weight.duplicated(subset=column_names, keep=False)  # Keeps all duplicates

# Output duplicate values
height_weight[duplicates].sort_values(by='first_name')

# Complete duplicates - keep one and discard the others
# .drop_duplicates() method
height_weight.drop_duplicates(
    inplace=True)  # Argument uses First as default (dont need to use subset for complete duplicates)

# For partial duplicates can use statistical measures to combine duplicate values
# Output duplicate values
column_names = ['first_name', 'last_name', 'address']
duplicates = height_weight.duplicated(subset=column_names, keep=False)
height_weight[duplicates].sort_values(by='first_name')

# The .groupby() and .agg() methods
# Group by column names and produce statistical summaries
column_names = ['first_name', 'last_name', 'address']
summaries = {'height': 'max', 'weight': 'mean'}  # Instructs groupby summary stats for different cols
height_weight = height_weight.groupby(by=column_names).agg(summaries).reset_index()  # Numbered indices of final output

# Make sure aggregation is done
duplicates = height_weight.duplicated(subset=column_names, keep=False)
height_weight[duplicates].sort_values(by='first_name')  # Should return an empty set

# Membership constraints
# Text and categorical data problems
# Categorical variables - often coded as number to run machine learning programmes

# Problems could be due to:
#  - Data entry errors - free text or drop downs
#  - Parsing errors

# How to treat:
#  - Dropping data
#  - Remapping categories
#  - Inferring categories

# Good idea to keep a log of all possible values of categorical data

# A note on joins:
#  - Anti-joins - when A is not in B
#  - Inner joins - what is in both A and B

# Finding inconsistent categories
inconsistent_categories = set(study_data['blood_type']).difference(categories['blood_type'])
# Set blood type - stores unique values
# Difference - takes the difference between the categories data frame
print(inconsistent_categories)

# Get and print rows with inconsistent categories
inconsistent_rows = study_data['blood_type'].isin(inconsistent_categories)

# Dropping inconsistent categories
consistent_data = study_data[~inconsistent_rows]  # Subsets everything except inconsistent rows

# What type of errors could we have?
# -- Value inconsistency - inconsistent fields, trailing white spaces
# -- Collapsing too many categories to too few - creating new groups, mapping groups to new ones
# -- Making sure data is of the right type

# Inconsistency, e.g. capitalisation
# Capitalise
marriage_status['marriage_status'] = marriage_status['marriage_status'].str.upper()
marriage_status['marriage_status'].value_counts()

# Lower case
marriage_status['marriage_status'] = marriage_status['marriage_status'].str.lower()
marriage_status['marriage_status'].value_counts()

# Leading or trailing spaces
# Strip all spaces
demographics = demographics['marriage_status'].str.strip()
demographics['marriage_status'].value_counts()

# Collapsing data into categories
# Create categories out of data

# Using qcut() function from pandas
import pandas as pd

group_names = ['0-200K', '200-500K', '500K+']
demographics['income_group'] = pd.qcut(demographics['household_income'], q=3,
                                       labels=group_names)  # Cuts based on distribution

# Better off using cut() - create category names and ranges
ranges = [0, 200000, 500000, np.inf]  # np.inf - infinity
group_names = ['0-200K', '200-500K', '500K+']
# Create income group column
demographics['income_group'] = pd.cut(demographics['household_income'],
                                      bins=ranges,
                                      labels=group_names)

# Map categories to fewer ones
# Create mapping dictionary and replace
mapping = {'Microsoft': 'DesktopOS', 'Linux': 'DesktopOS', 'IOS': 'MobileOS', 'Android': 'MobileOS'}
devices['operating_system'] = devices['operating_system'].replace(mapping)
devices['operating_system'].unique()  # Shows list of unique values

# Cleaning text data
# Common text data problems
#  - Data inconsistency
#  - Fixed length violations
#  - Typos

# Replace + with "00"
phones["Phone number"] = phones["Phone number"].str.replace("+", "00")

# Replace - with nothing
phones["Phone number"] = phones["Phone number"].str.replace("-", "")

# Replace phone numbers with less than 10 digits to NaN
digits = phones["Phone number"].str.len()  # Returns string length
phones.loc[digits < 10, "Phone number"] = np.nan  # Index rows where length below 10 and replace with NaN

# Find length of each row in Phone number column
sanity_check = phone["Phone number"].str.len()

# Assert min phone number length is 10
assert sanity_check.min() >= 10

# Assert all numbers do not have "+" or "-"
assert phone["Phone number"].str.contains("+|-").any() == False  # | is or statement
# .any() - if any element is True

# Regular expressions
# Replace letters with nothing
phones['Phone number'] = phones['Phone number'].str.replace(r'\D+', '')  # Pattern we want to replace and empty string
# r'\D+' - anything that is not a digit

# Advanced data problems
# Uniformity - e.g. measurements in different units

# Identify all rows where temperature is above 40
temp_fah = temperatures.loc[tempoeratures['Temperature'] > 40, 'Temperature']
# Convert to celsius
temp_cels = (temp_fah - 32) * (5/9)
temperatures.loc[temperatures['Temperature > 40'], 'Temperature'] = temp_cels  # Replaces fah temps with cel

# Assert conversion is correct
assert temperatures['Temperature'].max() < 40

# Date uniformity
# datetime is useful for representing dates
# pandas.to_datetime()
#  - Can recognise most formats automatically
#  - Sometimes fails with erroneous or unrecognisable formats

# Treating date data
birthdays['Birthday'] = pd.to_datetime(birthdays['Birthday'])
# Won't work with multiple formats

# Will work!
birthdays['Birthday'] = pd.to_datetime(birthdays['Birthday'],
                                       infer_datetime_format=True,  # Attempt to infer the format of each date
                                       errors='coerce')  # Return NA for rows where conversion failed

# Treating date data
birthdays['Birthday'] = birthdays['Birthday'].dt.strftime('%d-%m-%Y')  # Accepts a chosen format

# Treating ambiguous date data
# Convert to NA and treat accordingly
# Infer format by understanding source
# Infer format by previous and subsequent data in DF

# Cross field validation
# The use of multiple fields to sanity check data integrity
sum_classes = flights[['economy_class', 'business_class','first_class']].sum(axis=1)  # Axis=1 indicates row-wise summing
passenger_equ = sum_classes == flights['total_passengers']
# Find and filter out rows with inconsistent passenger totals
inconsistent_pass = flights[~passenger_equ]
consistent_pass = flights[passenger_equ]

# Convert to datetime and get todays date
users['birthday'] = pd.to_datetime(users['Birthday'])
today = dt.date.today()
# For each row in the birthday column, calculate year difference
age_manual = today.year - users['Birthday'].dt.year
# Find instances where ages match
age_equ = age_manual == users['Age']
# FInd and filter rows with inconsistent age
inconsistent_age = users[~age_equ]
consistent_age = users[age_equ]

# What to do with inconsistencies?
#  - Drop data
#  - Set to missing and impute
#  - Appy rules from domain knowledge

# Completeness
# Missing data - when no data is stored for an observation
# Can be represented as NA, nan, 0, .
# Due to either Technical or Human error

# Return missing values
airquality.isna()  # Returns boolean

# Get summary of missingness
airquality.isna().sum()

# Missingno packages - useful for visualising and understanding missing data
import missingno as msno
import matplotlib.pyplot as plt

# Visualise missingness
msno.matrix(airquality)
plt.show()  # Shows distribution of missing values across a column

# Isolate missing and complete values aside
missing = airquality[airquality['CO2'].isna()]
complete = airquality[~airquality['CO2'].isna()]

# Can be clearer to see with a sorted matrix
sorted_airquality = airquality.sort_values(by = 'Temperature')
msno.matrix(sorted_airquality)
plt.show()

# Missingness types
#  - Missing completely at random (MCAR) - no systematic relationship between missing data and other values (data entry)
#  - Missing at random (MAR) - systematic relationship between missing and other observed values (e.g. sensor failure)
# - Missing not at random (MNAR) - systematic relationship between missing data and unobserved values (e.g. missing high temps)
        # No way to tell from missing data

# Simple approaches to dealing with
#  - Drop missing data
#  - Inpute with statistical measures

# More complex approaches
#  - Imputing using algorithmic approach
#  - Impute with machine learning models

# Drop missing values
airquality_dropped = airquality.dropna(subset = ['CO2'])  # Picks which columns missing values to drop

# Replace with statistical measures
CO2_mean = airquality['CO2'].mean()
airquality_imputed = airquality.fillna({'CO2': co2_mean})

# Record linkage
# Comparing strings

# Minimum edit distance - least possible amount of steps needed to transition one string to another
# Operators:
#  - Insertion
#  - Deletion
#  - Substitution
#  - Transposition

# Lower edit distance - closer two words
# Min edit distance algorithms:
#  - Damerau-Levenshtein - insertion, substituion, deletion, transposition
#  - Levenshtein - insertion, substitution, deletion (most general - used here)
#  - Hamming - substitution only
#  - Jaro distance - transposition only

# Possible packages: nltk, fuzzywuzzy (used here), textdistance

# Simple string comparison
from fuzzywuzzy import fuzz

# Compare reeding vs. reading
fuzz.WRatio('Reeding', 'Reading')  # Output - score from 0 to 100
# Works well for partial strings and different orderings

# Comparison with arrays
# Import process
from fuzzywuzzy import process

# Define string and array of possible matches
string = "Houston Rockets vs Los Angeles Lakers"
choices = pd.Series(['Rockets vs Lakers', 'Lakers vs Rockets', 'Houson vs Los Angeles', 'Heat vs Bulls'])

process.extract(string, choices, limit = 2)  # Number of possible matches to returned

# Collapsing categories with string similarity
# For each correct category
for state in categories['state']:
    # Find potential matches in states with typos
    matches = process.extract(state, survey['state'], limit = survey.shape[0])  # Limit - length of survey dataframe
    # For each potential match match
    for potential_match in matches:  # Iterate over each potential match
        # If high similarity score
        if potential_match[1] >= 80:
            # Replace typo with correct category
            survey.loc[survey['state'] == potential_match[0], 'state'] = state

# Record linkage
#  - Generate pairs
#  --- ideally want to generate all possible pairs
#  --- but not scalable with large datasets
#  --- technique called blocking - creates pairs based on a matching column, reduces possible matching pairs
#  - Compare pairs
#  - Score pairs
#  - Link data

# using recordLinkage package
import recordlinkage

# create indexing object
indexer = recordlinkage.Index()  # Object to generate pairs

# Generate pairs blocked on state column
indexer.block('state')
pairs = indexer.index(census_A, census_B)
# Output - pandas MultiIndex object - array with possible pairs of indeces, makes it easier to subset

# Create a compare object
compare_cl = recordlinkage.Compare() # Responsible for assigning different comparison procedures for pairs

# Find exact matches for pairs of date_of_birth and state
compare_cl.exact('date_of_birth', 'date_of_birth', label='date_of_birth')
compare_cl.exact('state', 'state', label= 'state') # Col name for each df and column name
# Find similar matches for pairs of surname and address_1 using string similarity
compare_cl.string('surname', 'surname', threshold=0.85, label='surname')  # Threshold - value 0 to 1
compare_cl.string('address_1', 'address_1', threshold=0.85, label='address_1')

# Find matches
potential_matches = compare_cl.compute(pairs, census_A, census_B)

# Finding the only pairs we want
potential_matches[potential_matches.sum(axis = 1) => 2]

# Linking dataframes

# Probable matches
# Get indices from census B only
duplicate_rows = matches.index.get_level_values(1)  # Takes the column, can input name or order

# Finding duplicates in census B
census_B_duplicates = census_B[census_B.index.isin(duplicate_rows)]

# Finding new rows in census B
census_B_new = census_B[~census_B.index.isin(duplicate_rows)]

# Link the dataframes
full_census = census_A.append(census_B_new)

