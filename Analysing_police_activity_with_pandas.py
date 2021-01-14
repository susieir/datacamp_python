"""Analysing police activity with pandas"""

"""Chapter 1 - Preparing the data for analysis"""

# Stanford Open Policing Project
# Preparing the data
# - Examine
# - Clean

# Locating missing values
ri.isnull()  # Dataframe of T/F null values
ri.isnull().sum()  # Number of missing values

# Drop county name - all NaN
ri.drop('county_name', axis=columns, inplace=True)  # Inplace Avoids an assignment statement
# .dropna()  # Drops rows where a missing value exists
# Drop rows where no stop date or stop time
ri.dropna(subset=['stop_date', 'stop_time'], inplace=Truce)

# Using proper data types
ri.dtypes

# Object dtype - usually made up of python strings, can include others incl. lists
# Dtype affects which operations you can performn on a given series
# Avoid storing data as strings where possible

# Int, float - enables mathematical operations
# datetime - enables date based attributes and methods
# category - less memory usage, faster processing
# bool - logical and mathematical operations

apple['Price'] = apple.price.astype('float')  # Overwrite original series, changes dtype
# dot notation used - means same thing
# Bracket notation must be used to overwrite an existing series or create a new one

# Creating a DateTimeIndex
# Makes sense to combine stop date and stop time into a single column
# Convert to datetime format

apple.date.str.replace('/','-') # Temp change as haven't saved

combined = apple.date.str.cat(apple.time, sep=' ')  # Cat - concatenate

apple['date_and_time'] = pd.to_datetime(combined)

# Setting the index
apple.set_index('date_and_time', inplace=True)
# No longer considered to be one of the dataframe columns

"""Chapter 2 - The relationship between gender and policing"""
# Useful tools

# .value_counts(): Counts the unique values in a series
# Best suited for categorical data

# Expresing counts as proportions
# Raw counts
ri.stop_outcome.value_counts() # Outputs counts
ri.stop_outcome.valuecounts(normalize=True) # Outputs proportions

# Filtering dataframe rows
white = ri[ri.driver_race == 'White']
white.shape()  # Subset of white drivers

# Does gender affect who gets a ticket for speeding?
# Filtering by multiple conditions
female_and_arrested = ri[(ri.driver_gender == 'F') & (ri.is_arrested == True)]
# Each condition surrounded by parentheses
# & represents and operator
# | represents or operator

# Does gender affect whose vehicle is searched
# Math with boolean values

ri.is_arrested.mean()

# Comparing groups using groupby

ri.district.unique()

# For one district
ri[ri.district == 'Zone K1'].is_arrested.mean()
# For all districts
ri.groupby('district').is_arrested.mean()

# Group by multiple categories
ri.groupby(['district', 'gender']).is_arrested.mean()

# Does gender affect who is frisked during a search
# .value_counts() excludes missing values by default
# dropna=False displays missing values

# Locate 'inventory' among multiple search types
# Searching for a string
ri['inventory'] = ri.search_type.str.contains('Inventory', na=False)
# Returns True if found, False if not found
# na=False, returns False when finds a missing value

# Calculating the inventory rate
searched = ri[ri.search_conducted == True]
searched.inventory.mean()

"""Chapter 3 - Visual Exploratory Data Analysis"""

# Does time of day affect arrest rate
# Calculating the monthly mean price

apple.groupby(apple.index.month).price.mean()

# Are drug related stops on the rise?
# Resampling - change frequency of time series observations
# Resample price column by month
apple.price.resample("M").mean()

# Concatenating price and volume
monthly_price = apple.price.resample('M').mean()
monthly_volume = apple.volume.resample('M').mean()

monthly = pd.concat([monthly_price, monthly_volume], axis = 'columns')  # Concatenates along specified axis

monthly.plot(subplots=True)  # Two separate plots, independent y axes
plt.show()

# What violations are caught in each district?
# Computing a frequency table
pd.crosstab(ri.driver_race, ri.driver_gender)
# Tally of no times each combination of values occurs

# Selecting a dataframe slice
# .loc - select by label
table.loc['Asian':'Hispanic']

table.plot(kind='bar')
plt.show()

# Stacking the bars
table.plot(kind='bar', stacked=True)
plt.show()

# How long might you be stopped for a violation?
# Mapping one set of values to another
# Dict maps values you have to values you want
mapping = {'up': True,
           'down': False}

apple['is_up'] = apple.change.map(mapping)

# Ordering bars - bar chart
search_rate.sort_values().plot(kind='bar')

# Rotating bars
search_rate.sort_values().plot(kind='barh')
# Horizontal bar chart

"""Chapter 4 - Analysing the effect of weather on policing"""
# Exploring the weather dataset
# NOAA - National Centers for Environmental Information
# Boxplot
weather = pd.read_csv('weather.csv')
weather[['AWND', 'WSF2']].plot(kind='box')
plt.show()

# Hist
weather['WDFF'] = weather.WSF2 - weather.AWND
weather.WDFF.plot(kind='hist', bins=20)
plt.show()
# Normal distribution (for naturally generated data) suggests data trustworthy

# Categorising the weather
temp = weather.loc[:, 'TAVG': 'TMAX']

temp.sum(axis='columns').head()  # Sums across columns
# Axis specifies that array dimension that is being aggregated

# Changing data from object to category
# Stores the data more efficiently
# Allows you to specify a logical order for categories
ri.stop_length.memory_usage(deep=True)
cats = ['short', 'medium', 'long']  # Defines logical order of categories

ri['stop_length'] = ri.stop_length.astype('category', ordered=True, categories=cats)
# Ordering allows use of comparison operators

ri[ri.stop_length > 'short'].shape

# Sorts calculations logically rather than alphabetically

# Merging datasets
apple.reset_index(inplace=True)  # Resets to default index
apple_high = pd.merge(left = apple, right = high, left_on = 'date', right_on = 'DATE', how='left')

# Does weather affect the arrest rate
search_rate = ri.groupby(['violation', 'driver_gender']).search_conducted.mean()
# Result - pandas multi-index series
# Similar properties to a df
search_rate.loc['Equipment']  # Picks up equipment rows

# Converting a multi-index series to a dataframe
search_rate.unstack()


ri.pivot_table(index='violation',
               columns='driver_gender',
               values='search_conducted')
# Mean is default aggregation function