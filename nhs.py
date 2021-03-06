# Import packages
import pandas as pd
import matplotlib as plt
import seaborn as sns
import requests
import zipfile

# Load data from NHS website - start with Mar20

# Link to file path [Next step - scrape from web]
file = 'C:\\Users\\susie\\Documents\\Home\\Development\\Python\\NHS Data\\2003_NHS_RTT.csv'

# Load data into a dataframe
df = pd.read_csv(file,
                 header=0,
                 names=['Period',
                        'Provider Parent Org Code',
                        'Provider Parent Name',
                        'Provider Org Code',
                        'Provider Org Name',
                        'Commissioner Parent Org Code',
                        'Commissioner Parent Name',
                        'Commissioner Org Code',
                        'Commissioner Org Name',
                        'RTT Part Type',
                        'RTT Part Description',
                        'Treatment Function Code',
                        'Treatment Function Name',
                        '0 to 1 wks',
                        '1 to 2 wks',
                        '2 to 3 wks',
                        '3 to 4 wks',
                        '4 to 5 wks',
                        '5 to 6 wks',
                        '6 to 7 wks',
                        '7 to 8 wks',
                        '8 to 9 wks',
                        '9 to 10 wks',
                        '10 to 11 wks',
                        '11 to 12 wks',
                        '12 to 13 wks',
                        '13 to 14 wks',
                        '14 to 15 wks',
                        '15 to 16 wks',
                        '16 to 17 wks',
                        '17 to 18 wks',
                        '18 to 19 wks',
                        '19 to 20 wks',
                        '20 to 21 wks',
                        '21 to 22 wks',
                        '22 to 23 wks',
                        '23 to 24 wks',
                        '24 to 25 wks',
                        '25 to 26 wks',
                        '26 to 27 wks',
                        '27 to 28 wks',
                        '28 to 29 wks',
                        '29 to 30 wks',
                        '30 to 31 wks',
                        '31 to 32 wks',
                        '32 to 33 wks',
                        '33 to 34 wks',
                        '34 to 35 wks',
                        '35 to 36 wks',
                        '36 to 37 wks',
                        '37 to 38 wks',
                        '38 to 39 wks',
                        '39 to 40 wks',
                        '40 to 41 wks',
                        '41 to 42 wks',
                        '42 to 43 wks',
                        '43 to 44 wks',
                        '44 to 45 wks',
                        '45 to 46 wks',
                        '46 to 47 wks',
                        '47 to 48 wks',
                        '48 to 49 wks',
                        '49 to 50 wks',
                        '50 to 51 wks',
                        '52 wks+',
                        'Total',
                        'Unknown Start Date',
                        'Grand Total'],
                 dtype={'Period': 'object',
                        'Provider Parent Org Code': 'category',
                        'Provider Parent Name': 'category',
                        'Provider Org Code': 'category',
                        'Provider Org Name': 'category',
                        'Commissioner Parent Org Code': 'category',
                        'Commissioner Parent Name': 'category',
                        'Commissioner Org Code': 'category',
                        'Commissioner Org Name': 'category',
                        'RTT Part Type': 'category',
                        'RTT Part Description': 'category',
                        'Treatment Function Code': 'category',
                        'Treatment Function Name': 'category',
                        'Gt 00 To 01 Weeks SUM 1': 'float',
                        'Gt 01 To 02 Weeks SUM 1': 'float',
                        'Gt 02 To 03 Weeks SUM 1': 'float',
                        'Gt 03 To 04 Weeks SUM 1': 'float',
                        'Gt 04 To 05 Weeks SUM 1': 'float',
                        'Gt 05 To 06 Weeks SUM 1': 'float',
                        'Gt 06 To 07 Weeks SUM 1': 'float',
                        'Gt 07 To 08 Weeks SUM 1': 'float',
                        'Gt 08 To 09 Weeks SUM 1': 'float',
                        'Gt 09 To 10 Weeks SUM 1': 'float',
                        'Gt 10 To 11 Weeks SUM 1': 'float',
                        'Gt 11 To 12 Weeks SUM 1': 'float',
                        'Gt 12 To 13 Weeks SUM 1': 'float',
                        'Gt 13 To 14 Weeks SUM 1': 'float',
                        'Gt 14 To 15 Weeks SUM 1': 'float',
                        'Gt 15 To 16 Weeks SUM 1': 'float',
                        'Gt 16 To 17 Weeks SUM 1': 'float',
                        'Gt 17 To 18 Weeks SUM 1': 'float',
                        'Gt 18 To 19 Weeks SUM 1': 'float',
                        'Gt 19 To 20 Weeks SUM 1': 'float',
                        'Gt 20 To 21 Weeks SUM 1': 'float',
                        'Gt 21 To 22 Weeks SUM 1': 'float',
                        'Gt 22 To 23 Weeks SUM 1': 'float',
                        'Gt 23 To 24 Weeks SUM 1': 'float',
                        'Gt 24 To 25 Weeks SUM 1': 'float',
                        'Gt 25 To 26 Weeks SUM 1': 'float',
                        'Gt 26 To 27 Weeks SUM 1': 'float',
                        'Gt 27 To 28 Weeks SUM 1': 'float',
                        'Gt 28 To 29 Weeks SUM 1': 'float',
                        'Gt 29 To 30 Weeks SUM 1': 'float',
                        'Gt 30 To 31 Weeks SUM 1': 'float',
                        'Gt 31 To 32 Weeks SUM 1': 'float',
                        'Gt 32 To 33 Weeks SUM 1': 'float',
                        'Gt 33 To 34 Weeks SUM 1': 'float',
                        'Gt 34 To 35 Weeks SUM 1': 'float',
                        'Gt 35 To 36 Weeks SUM 1': 'float',
                        'Gt 36 To 37 Weeks SUM 1': 'float',
                        'Gt 37 To 38 Weeks SUM 1': 'float',
                        'Gt 38 To 39 Weeks SUM 1': 'float',
                        'Gt 39 To 40 Weeks SUM 1': 'float',
                        'Gt 40 To 41 Weeks SUM 1': 'float',
                        'Gt 41 To 42 Weeks SUM 1': 'float',
                        'Gt 42 To 43 Weeks SUM 1': 'float',
                        'Gt 43 To 44 Weeks SUM 1': 'float',
                        'Gt 44 To 45 Weeks SUM 1': 'float',
                        'Gt 45 To 46 Weeks SUM 1': 'float',
                        'Gt 46 To 47 Weeks SUM 1': 'float',
                        'Gt 47 To 48 Weeks SUM 1': 'float',
                        'Gt 48 To 49 Weeks SUM 1': 'float',
                        'Gt 49 To 50 Weeks SUM 1': 'float',
                        'Gt 50 To 51 Weeks SUM 1': 'float',
                        'Gt 51 To 52 Weeks SUM 1': 'float',
                        'Gt 52 Weeks SUM 1': 'float',
                        'Total': 'float',
                        'Patients with unknown clock start date': 'float',
                        'Total All': 'float'})

print(df.head())

# Check data types
df.dtypes

# Update data types
# Save category tables
# Convert period to Mar-2020
# Change column names for data
# Check Total all adds up to summed columns
