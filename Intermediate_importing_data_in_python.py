# Intermediate importing data in python
# Importing flat files from the web
# Previous imports all worked if in local directory

# Downloading via URL:
# Reproducibility issues
# Not scalable

# The urllib package
# Provides interface for fetching data across the web
# urlopen() - accepts universal resource locators instead of file names

# Automate file download in python

from urllib.request import urlretrieve
import pandas as pd

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
urlretrieve(url, 'winequality-white.csv')  # Saves url locally - could skip this step and import directly to df

df = pd.read_csv('winequality-white.csv', sep=';')  # Could use url to skip the step above
print(df.head())

# HTTP requests to import files from the web
# URL - Uniform/universal resource locator
# References to web resources
# Focus - webaddresses
# Can also be FTP and database access
# Ingredients:
#  - Protocol identifier - http:
#  - Resource name - datacamp.com
# The specify web address uniquely

# http - HyperText transfer protocol
# Foundation of data communication for the web
# https - a more secure version of http
# Going to a website - sending HTTP request
#  - GET request
# urlretrieve() - performs a GET request and saves data locally
# HTML - HyperText markup language - standard language for the web

# GET requests using urllib

from urllib.request import urlopen, Request
url = "https://www.wikipedia.org/"
request = Request(url)  # GET request
response = urlopen(request)  # Send request and catch response
html = response.read()   # Apply read method to response, returns html as string
response.close()  # Closes the response

# Requests package - API for making requests - higher level interface, less code
# One of the most downloaded python packages of all time!

import requests
url = "https://www.wikipedia.org/"
r = requests.get(url)
text = r.text

print(text)

# Scraping the web in python

# HTML - mix of both structured and unstructured data
# Structured data -
#  - Has a predefined data model, or
#  - Is organised in a predefined manner
# Unstructured data - does not possess either of these properties
# Contains tags that point to where headings can be found and hyperlinks

# BeautifulSoup
# Parse and extract structured data from HTML
# Tagsoup - unstructured html data on webpage
# Makes tagsoup beautiful and extracts information

from bs4 import BeautifulSoup
import requests  # Used to scrape the info from the web
url = 'https://www.crummy.com/software/BeautifulSoup/'
r = requests.get(url)
html_doc = r.text
soup = BeautifulSoup(html_doc)  # Create beautifulsoup object from the html and prettify it

print(soup.prettify())  # Output is indented in the way you would expect

# Many methods, such as:
print(soup.title)  # Prints title
print(soup.get_text)  # Prints text

# find_all() - extracts all the hyperlinks in the html

for link in soup.find_all('a'):  # A tags define hyperlinks
    print(link.get('href'))

# Introduction to APIs and JSONs
# APIs - application programming interfaces
# Set of protocols and routines
#  - Building and interacting with software applications

# JSON file format - standard file format for transferring data through APIs
# Javascript Object Notation
# Real-time server to browser communication
# Human readable
# Naturally stored as dict in python

# Loading JSONs in Python

import json
with open('snakes.json', 'r') as json_file:
    json_data = json.load(json_file)  # Python imports as dict

for key, value in json_data.items():
    print(key + ':', value)

# APIs and interacting with the world wide web

# Much of data from APIs packaged in JSONs
# API - set of protocols and routines for interacting with software applications
# Bunch of code that allows two software programs to communicate with each other
# Connecting to an API in python

import requests
url = 'http://www.omdbapi.com/?t=hackers'  # Assign URL
r = requests.get(url)  # Package, send request, catch response
json_data = r.json()  # Built in json decoder for when dealing with json data
for key, value in json_data.items():  # Prints key, value pairs in dictionary
    print(key + ':', value)

# http - making http request
# www.omdbapi.com - querying the OMDB API
# ?t=hackers - Query string - begins with '?'
# Parts of the URL that do not fit in to the conventional hierarchical path structure
# t=hackers - Return data for a movie with title (t) 'Hackers'
# Can work this out from the documentation on the APIs home page
# Can navigate to the URL
# Chrome extension JSON formatter can make it look a bit prettier

# The Twitter API and authentication
# Twitter requires that you have an account
# Login to Twitter Apps and create a new app
# Keys and access tokens tab > copy API key, API secret, access token, access token secret
# Credentials required to access API

# Twitter has a number of APIs
# REST API - representational state transfer, allows user to read and write twitter data
# To read in real time - streaming API > public stream
# To read and process tweets - use GET statuses/sample API - returns a small random sample of public streams
# To get all - would need to use firehose API - not publicly available
# Field guide - tweets returned as JSONs - field guide will help to decode

# Tweepy - authentication handler
tw_auth.py

import tweepy, json
access_token = "..."
access_token_secret = "..."
consumer_key = "..."
consumer_secret = "..."
auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Define stream listener class
st_class.py

class MyStreamListener(tweepy.StreamListener):
    def __init__(self, api=None):
        super(MyStreamListener, self).__init__()
        self.num_tweets = 0
        self.file = open("tweets.txt", "w")
    def on_status(self, status):
        tweet = status._json
        self.file.write(json.dumps(tweet) + '\\n')
        tweet_list.append(status)
        self.num_tweets += 1
        if self.num_tweets < 100:
            return True
        else:
            return False
        self.file.close()

# Create a tweet listener that creates a file called tweets.txt, collects streaming tweets and writes them to the file
# Once 100 tweets streamed, listener closes file and stops listening

# Then need to create an instance of the listener class and authenticate it
tweets.py

# Create streaming object and authenticate
l = MyStreamListener()
stream = tweepy.Stream(auth, l)
# This line filters Twitter Streams to capture data by keywords:
stream.filter(track=['apples', 'oranges'])
        

