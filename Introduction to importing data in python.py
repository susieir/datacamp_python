# Introduction to importing data in python

# Flat files - txt, csv
# Files from other software - matlab, stata
# Relational databases - SQLLite, MYSQL

# Plain text files - plain text
# Table data - e.g. titanic.csv

# Reading a text file
# Basic open function
filename = 'huck_finn.txt'
file = open(filename, mode='r')  # 'r' is to read only
text = file.read()  # Assign the text from the file to a variable text by applying the method read to the connection
# to the file
file.close()  # Closes the connection to the file

# mode='w' - allows write to file

with open('huck_finn.txt', 'r') as file:
    print(file.read())
# creates context to execute commands with the file open
# once out of the context, the file is no longer open - avoids needing to close the connection to the file
# 'with' - context manager
# binding a variable in the context manager construct -
# filename will be bound to open filename r whilst within the construct
# 'with' is best practice - as don't need to close files

# IPython - magic commands - interactive python shell
# ! - gives you complete shell access
# e.g. ! ls - displays the contents of local directory

# To check whether file is closed
print(file.closed)

# Read the first line
file.readline()

# The importance of flat files in data science
# Text files containing records
# Table data - without relationships
# Consist of records - row of fields or attributes
# Can have a header - a row that occurs as first row, describes contents of columns

# File extension
# .csv - comma separated value
# .txt - text file
# commas, tabs - delimiters

# How to import flat files
# Two main packages - numpy, pandas
# If entirely numbers, can use numpy
# If want to store in dataframe - would use pandas

# Importing flat files using NumPy
# NumPy arrays - python standard for storing numerical data
# Often essential for other packages such as scikit-learn - machine learning package for Python

#loadtxt()
#genfromtxt()

import numpy as np
filename = 'MNIST.txt'
data = np.loadtxt(filename, delimiter=',',  # Default delimiter is any white space - usually need to specify
                  skiprows=1,  # Skip the first row, e.g. header
                  usecols=[0,2])  # Specifies which rows to pick up
data

# Can also import different data types
data = np.loadtxt(filename, delimiter=',', dtype=str)  # Ensures all entries imported as string

# Loadtxt great for basic cases, breaks down with mixed datatypes

# tab delimited - '\t'
# skiprows - specifies how many rows you wish to skip
# usecols - takes a list of columns you wish to keep

# np.genfromtxt() - can handle mixed data types
# dtype=0 will figure out what types each column should be
# names=True - tells it there is a header

# Array with different types - structured array
# 1D array where each element of the array is row of the flat file imported

# np.recfromcsv() - similar to genfromtxt, with default dtype set to None, default headers and default csv

# Importing flat files using pandas
# Numpy arrays can have 2D labelled data structures with columns of two different types
# Pandas avoids switching to a programme like R
# A matrix has rows and columns - a dataframe has observations and variables

# Manipulating dataframes
# Standard and best practice to use pandas to import data as dataframes

import pandas as pd
filename = 'winequality-red.csv'
data = pd.read_csv(filename)
data.head()

data_array = data.values  # Converts data to a numpy array

pd.read_csv(filename, nrows=5, header=None)  # If you want to pass just the first 5 rows and there is no header

pd.read_csv(filename,
            sep='\t',  # Tab delimited
            comment='#',  # Recognises comments after '#'
            na_values='Nothing')  # Adds 'Nothing' where there are no values

# Feather - a fast, language-agnostic data frame file format

# Introduction to other file types

# Pickled files - file type native to python
# Many data types for which it isn't obvious how to store
# Pickled files are serialised - converted into byte strings

import pickle
with open('pickled_fruit.pkl', 'rb') as file:  # Open and specify read only, binary (computer readable, not human readable)
    data = pickle.load(file)
print(data)

# Importing excel spreadsheets

import pandas as pd
file = 'urbanpop.xls'
data = pd.ExcelFile(file)
print(data.sheet_names)  # Figure out what the sheets are

df1 = data.parse('1960-1966')  # sheet name, as a string
df2 = data.parse(0)  # sheet index, as a float (alternative to sheet name)

# Library os, consists of misc operating system interfaces
import os
wd = os.getcwd()  # Stores the name of the current directory as 'wd'
os.listdir(wd)  # outputs the contents of the directory to a list in the shell

# Excel
df1 = xls.parse(0, skiprows=[1], names=['Country', 'AAM due to War (2002)'])  # Skips first row (header), renames cols

df2 = xls.parse(1, usecols=[0], skiprows=1, names=['Country'])  # Parses only the first column

# Importing SAS/STATA files using Pandas
# SAS - statistics analysis system - used in business analytics and biostatistics
# Performs advanced analytics, multivariate analysis, business intelligence, data management, predictive analytics, comp analysis

import pandas as pd
from sas7bdat import SAS7BDAT
with SAS7BDAT('urbanpop.sas7bdat') as file:
    df_sas = file.to_data_frame()

# STATA - academic social sciences research

import pandas as pd
data = pd.read_stata('urbanpop.dta')  # Don't need to initialise context manager

# Importing HDF5 files
# Hierarchical data format version 5
# Standard for storing large quantities of numerical data
# Data can be hundreds of gigabytes or terabytes in size
# HDF5 can scale to exabytes

import h5py
filename = 'H-H1_LOSC_4_V1-815411200-4096.hdf5'
data = h5py.File(filename, 'r')  # r is to read
print(type(data))

# Can explore the structure using the method keys

for key in data.keys():
    print(key)

# Returns HDF groups - like directories
# e.g. want to explore meta data, can print keys

for key in data['meta'].keys():
    print(key)

print(data['meta']['Description'].value)  # Prints the value of the two keys

# Importing MATLAB files
# Matrix Laboratory
# Industry standard in engineering and science
# Powerful linear algebra and matrix capabilities
# .mat files

# SciPy
# scipy.io.loadmat() - read .mat files
# scipy.io.savemat() - write .mat files

# Workspace - can contain strings, floats, vectors and arrays
# When importing - expect to see a number of different variables and objects

import scipy.io
filename = 'workspace.mat'
mat = scipy.io.loadmat(filename)
print(type(mat))  # Tells imported as dict

# keys - matlab variable names
# values - objects assigned to variables

# Introduction to relational databases
# E.g. Northwind database
# Orders table
# Customers table
# Employess table

# Table - one entity type - analagous to a dataframe
# Each row - instance of entity type
# Each column - attribute of each instance
# Each row must contain unique identifier - primary key

# Tables are linked
# Relational model - widely adopted
# Codd's 12 rules/commandments
#- Consists of 13 rules (zero-indexed!)
#- Describes what a Relational Database Management System should adhere to to be considered relational

# Relational database management systems
# PostgreSQL
# MySQL
# SQLite

# Creating a database engine in python
# Connecting to a database
# E.g. SQLite - fast and simple
# Packages to access SQLite database - use SQLAlchemy here - works with many other RDMSystems

from sqlalchemy import create_engine
engine = create_engine('sqlite:///Northwind.sqlite')  # Engine that communicates queries to database

# Getting table names
table_names = engine.table)names()
print(table_names)

# Querying relational databases in python

# Basic SQLQuery
SELECT * from Table_Name  # Returns all columns from all rows of the table

# Workflow of SQL querying:
# Import packages and functions
# Create the database engine
# Connect to the engine
# Query the database
# Save query results to a DataFrame
# Close the connection

from sqlalchemy import create_engine
import pandas as pd
engine = create_engine('sqlite://Northwind.sqlite')  # Creates the database engine
con = engine.connect()  # Connects to the engine
rs = con.execute("SELECT * FROM Orders")  # Applys the method execute to the connection con, passes the relevant SQL query
# Creates a single SQL result, assigned to the variable rs
# Turn result into DataFrame - apply fetchall to rs
df = pd.DataFrame(rs.fetchall()) # Saves as dataframe, fetchall - all rows
df.columns = rs.keys()  # Sets the column names
con.close()  # Closes the connection

# Context manager can be used to open a connection - saves trouble of closing

from sqlalchemy import create_engine
import pandas as pd
engine = create_engine('sqlite:///Northwind.sqlite')

with engine.connect() as con:
    rs = con.execute("SELECT OrderID, OrderDate, ShipName from Orders")  # Pulls column names
    df = pd.DataFrame(rs.fetchmany(size=5))  # Fetchmany - imports 5 rows instead of all rows
    df.columns = rs.keys()

# Querying relational databases with Pandas
# Can reduce to one line of code using pd.read_sql_query()

df = pd.read_sql_query("SELECT * FROM Orders", engine) # First arg is query, second arg is engine want to connect to

# Advanced querying - exploiting table relationships
# Joining tables

from sqlalchemy import create_engine
import pandas as pd
engine = create_engine('sqlite:///Northwind.sqlite')
df = pd.read_sql_query("SELECT OrderID FROM Orders INNER JOIN Customers on Orders.CustomerID = Customers.CustomerID",engine)
print(df.head())








