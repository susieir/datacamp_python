# Working with dates and times in python
two_hurricanes = ["10/7/2016", "6/21/2017"]

# Import date
from datetime import date

# Create dates
two_hurricanes_dates = [date(2016, 10, 7), date(2017, 6, 21)]

# Access attributes
print(two_hurricanes_dates[0].year)
print(two_hurricanes_dates[0].month)
print(two_hurricanes_dates[0].day)

# Finding the weekday of a date
print(two_hurricanes_dates[0].weekday())  # Monday = 0

# Math with dates

# Create dates
d1 = date(2017, 11, 5)
d2 = date(2017, 12, 4)
l = [d1, d2]
print(min(l))

# Subtract two dates
delta = d2 - d1
print(delta.days)  # Time delta - elapsed time between events

# Import timedelta
from datetime import timedelta

# Create a 29 day timedelta
td = timedelta(days=29)
print(d1 + td)

# Turning dates into strings
# ISO 8601 format
from datetime import date

# Example date
d = date(2017, 11, 5)
# ISO format: YYYY-MM-DD - default format - always the same length
print(d)
# Express the date in ISO 8601 format and put it in a list
print([d.isoformat()])

# If don't want to put in ISO format
# strf time
print(d.strftime("%Y"))
print(d.strftime("Year is %Y"))
print(d.strftime("%Y/%m/%d"))

# %B - month's full name
# %j - day of the year

# Adding time to the mix

from datetime import datetime

dt = datetime(2017, 10, 1, 15, 23, 25)  # Year, month, day, hour, min, second - all args need to be whole numbers
# Can add microseconds - 500000 is 0.5 seconds. Seconds are millionths
# Nanoseconds are also an option
# Microseconds - defaults to 0
# Can use named args

dt_hr = dt.replace(minute=0, second=0, microsecond=0)  # Update to round down to start of hour
print(dt_hr)

# Printing and parsing datetimes

# Create datetime
dt = datetime(2017, 12, 30, 15, 19, 13)
print(dt.strftime("%Y-%m-%d"))
print(dt.strftime("%Y-%m-%d %H:%M:%S"))

# Print ISO format
print(dt.isoformat())

# Import datetime
from datetime import datetime

dt = datetime.strptime("12/30/2017 15:19:13", # Date string
                       "%m/%d/%Y %H:%M:%S")  # Format string
# Exact match needed for string conversion

# Parsing datetimes with pandas
# Unix timestamp - number of seconds since 1 Jan 1970
# A timestamp
ts = 1514665153.0
# Convert to datetime and print
print(datetime.fromtimestamp(ts))

# Working with durations
# Create example datetimes
start = datetime(2017, 10, 8, 23, 46, 47)
end = datetime(2017, 10, 9, 0, 10, 57)

# Subtract datetimes to create a time delta
duration = end - start
print(duration.total_seconds())  # Number of seconds in timedelta

# Import timedelta
from datetime import timedelta

# Create a timedelta
delta1 = timedelta(seconds=1)

# One second later
print(start + delta1)

# Create a one day and one second time delta
delta2 = timedelta(days=1, seconds=1)

# Create a negative timedelta of one week
delta3 = timedelta(weeks=-1)
# Can also subtract timedeltas, which does the same as a negative timedelta

# UTC offsets
# Import relevant classes
from datetime import datetime, timedelta, timezone

# US Eastern Standard time zone
ET = timezone(timedelta(hours=-5))
# Timezone aware datetime
dt = datetime(2017, 12, 30, 15, 9, 3, tzinfo = ET)
print(dt)

# India standard timezone
IST = timezone(timedelta(hours=5, minutes=30))
# Convert to IST
print(dt.astimezone(IST))

print(dt.replace(tzinfo=timezone.utc))  # UTC is a preset object
# Clock stays the same but the UTC timezone has shifted

# Change original to match UTC
print(dt.astimezone(timezone.utc))  # Changes UTC offset and the clock itself

# Timezone database
from datetime import datetime
from dateutil import tz  # Updated 3-4x per year

# Eastern time
et = tz.gettz('America/New_York')  # Format: Continent/City

# Other examples:
# - America/Mexico_City
# - Europe/London
# - Africa/Accra

# Last ride
last = datetime(2017, 12, 30, 15, 9, 3, tzinfo=et)
print(last)

# First ride
first = datetime(2017, 10, 1, 15, 23, 25, tzinfo=et)
print(first)
# Automatically updates based on the date and time

# tz can be used for historical dates and timestamps

# Starting daylight saving time
spring_ahead_159am = datetime(2017, 3, 12, 1, 59, 59)
print(spring_ahead_159am.isoformat())

spring_ahead_3am = datetime(2017, 3, 12, 3, 0, 0)
print(spring_ahead_3am.isoformat())

print((spring_ahead_3am - spring_ahead_159am).total_seconds())
# To fix problem of not recognising clock change, use timezone objects

from datetime import timezone, timedelta

EST = timezone(timedelta(hours=-5))
EDT = timezone(timedelta(hours=-4))

spring_ahead_159am = spring_ahead_159am.replace(tzinfo=EST)
print(spring_ahead_159am.isoformat())

spring_ahead_3am = spring_ahead_3am.replace(tzinfo=EDT)
print(spring_ahead_3am.isoformat())

print((spring_ahead_3am - spring_ahead_159am).total_seconds())

# Using dateutil - saves us having to know rules
from dateutil import tz

# Create Eastern timezone
eastern = tz.gettz('America/New_York')
spring_ahead_159am = datetime(2013, 3, 12, 1, 59, 59, tzinfo=eastern)
spring_ahead_3am = datetime(2013, 3, 12, 3, 0, 0, tzinfo=eastern)
print((spring_ahead_3am-spring_ahead_159am).seconds)

# Ending daylight saving time
eastern = tz.gettz('US/Eastern')
first_1am = datetime(2017, 11, 5, 1, 0, 0, tzinfo=eastern)
print(tz.datetime_ambiguous(first_1am))  # Need to tell it apart - could occur at two different UTC moments
second_1am = datetime(2017, 11, 5, 1, 0, 0, tzinfo=eastern)
second_1am = tz.enfold(second_1am)  # This belongs to second time, doesn't change behaviour. Python doesn't take into acc
# Need to convert to UTC - unambiguous
first_1am = first_1am.astimezone(tz.UTC)
second_1am = second_1am.astimezone(tz.UTC)
print((second_1am - first_1am).total_seconds())

# When you care about accounting for DST accurately, switch into UTC for comparing events

# Reading date and time in Pandas
import pandas as pd
# Import file
rides = pd.read_csv('capital-onebike.csv')
# Importing whilst treating date columns as datetimes
rides = pd.read_csv('capital-onebike.csv',
                    parse_dates = ['Start date', 'End date'])  # Set as list of column names
# Datetime conversion will be intelligent, but sometimes you'll need:
rides['start date'] = pd.to_datetime(rides['start date'],
                                     format= "%Y-%m-%d %H:%M:%S")  # Can specify format
# Now get a pandas timestamp

# Create duration col - get timedelta
rides['Duration'] = rides['End date'] - rides['Start date']

# Pandas code is often written in methods chaining style
rides['Duration']\
    .dt.total_seconds()\
    .head()

# Use \ for readability
# dt. all typical datetime methods

# Summarising data in pandas
# Average time out of the dock
rides['Duration'].mean()  # Gives timedelta

# Total time out of the dock
rides['Duration'].sum()

# Percent of time out of the dock
rides['Duration'].sum() / timedelta(days=91)

# Count how many times the bike started at each station
rides['Member type'].value_counts()  # How many times a given value appears for each member type

# Percent of rides by member
rides['Member type'].value_counts() / len(rides)

# Add duration (in seconds) column
rides['Duration seconds'] = rides['Duration'].dt.total_seconds()
# Average duration by member type
rides.groupby('Member type')['Duration seconds'].mean()

# Average duration by month
rides.resample('M', on = 'Start date')['Duration seconds'].mean()  # Groups rides by month
# Resample takes a unit of time - e.g. 'M' for month
# And a datetime column to group on

# Size by group
rides.groupby('Member type').size()
# First ride by group
rides.groupby('Member type').first()  # Frist row of each group

# Plotting results
# Average duration by month
rides\
    .resample('M', on='Start date')\
    ['Duration seconds']\
    .mean()\
    .plot()

# Average duration by day
rides\
    .resample('D', on='Start date')\
    ['Duration seconds']\
    .mean()\
    .plot()

# Additional datetime methods in pandas
# Datetime objects start off as timezone naive
# Put times into timezone
rides['Start date'].head(3)\
    .dt.tz_localize('America/New_York')
# But - may throw an ambiguous time error

# Handle ambiguous datetimes
rides['Start date'] = rides['Start date']\
    .dt.tz_localize('America/New_York', ambiguous='NaT')

rides['End date'] = rides['Start date']\
    .dt.tz_localize('America/New_York', ambiguous='NaT')
# Sets ambiguous results to not a time - skips over NaT

# Year
rides['Start date']\
    .head(3)\
    .dt.year

# also - dt.weekday_name

# Shift the indexes forward one, padding with NaT
rides['End date'].shift(1).head(3) # First row is set to NaT






