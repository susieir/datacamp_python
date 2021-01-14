# Chapter 1
# Inner join
wards_census = wards.merge(census, on='wards') #Adds census to wards, matching on the wards field
# Only returns rows that have matching values in both tables
# Suffixes automatically added by the merge function to differentiate between fields with the same name in both source tables
wards_census = wards.merge(census, on='ward', suffixes=('_cen','_ward'))
# Can customise the suffixes

#One to many relationships - pandas takes care of one to many relationships, and doesn't require anything different

#Merging multiple dataframes
#Merging on multiple column names
grants.merge(licenses, on=['address', 'zip']) #Uses multiple columns in merge
grants_licenses_ward = grants.merge(licenses, on=['address', 'zip']) \ #backslash line continuation method, reads as one line of code
    .merge(wards, on='ward', suffixes=('_bus', '_ward'))
#Can continue to merge as needed

# Chapter 2
#Left join
movies_taglines = movies.merge(taglines, on='id', how='left')

#Other joints
#Right join
# Different column names for join column
tv_movies = movies.merge(tv_genre, how='right', left_on='id', right_on='movie_id')

#Outer join
family_comedy = family.merge(comedy, on='movie_id', how='outer', suffixes=('_fam', '_com'))

#Self join

# Index join
movies_genres = movies.merge(movie_to_genres, left_on = 'id', left_index=True, right_on = 'movie_id', right_index=True)

# Filtering join
# Mutating joins - combines data from two tables based on matching observations in both tables
# Filtering joins - filter observations from table based on whether or not they match an observation in another table

#Semi join
# Returns the intersection, similar to an inner join. But returns only columns from the left table and not the right. No duplicates returned
genres['gid'].isin(genres_tracks['gid']) #Returns True or False list

#Semi-join - filters genres table by what's in the top tracks table
genres_tracks = genres.merge(top_tracks, on='gid')
top_genres = genres[genres['gid'].isin(genres_tracks['gid'])]

#Anti-join - returns observations in left table that don't have a matching observations in right table, incl. only left table columns
genres_tracks = genres.merge(top_tracks, on='gid', how='left', indicator=True) #Adds merge columns telling source of each row
gid_list = genres_tracks.loc[genres_tracks['_merge'] == 'left_only'], 'gid'] #List of GIDs not in tracks method
non_top_genres = genres[genres['gid'].isin(gid_list)]

#Concatenate two tables vertically
# Pandas .concat() can concatenate both vertical and horizontal
# axis=0, vertical
pd.concat([inv_jan, inv_feb, inv_mar], ignore_index=True) #Combined in order passed in, axis=0 is the default, ignores index
# Setting labels to original tables
pd.concat([inv_jan, inv_feb, inv_mar], ignore_index=False, keys=['jan','feb','mar']) #Cant add a key and ignore index at same time
# Concat tables with different column names - will be automatically be added
pd.concat([inv_jan, inv_feb], sort=True) #Sorts alphabetically
# If only want matching columns, set join to inner
pd.concat([inv_jan, inv_feb], join='inner') #Default is equal to outer, why all columns included as standard

#Append - simplified concat methods
# Supports ignore_index and sort
# Does not support keys or join - always an outer join
inv_jan.append([inv_feb, inv_mar], ignore_index=True, sort=True)

# Validating merges
.merge(validate=None) #Checks if merge is of specifed type
""" 
'one_to_one'
'one_to_many'
'many_to_one'
'many_to_many'"""
tracks.merge(specs, on='tid', validate='one_to_one') #Raises error if issue

.concat(verify_integrity=False) #Checks for duplicate indexes and raises error if there are
pd.concat([inv_feb, inv_mar], verify_integrity=True)

#Using merge_ordered()
# Similar to standard merge with outer join, sorted
# Useful for ordered or timeseries data
# Similar methodology, but default is outer
# Calling method is different
pd.merge_ordered(df1, df2)
pd.merge_ordered(appl, mcd, on='date', suffixes=('_aapl','_mcd'))
# Forward fill - fills in with previous value
pd.merge_ordered(appl, mcd, on='date', suffixes=('_aapl','_mcd'), fill_method='ffill')

# Merge_asof() - ordered left join, matches on nearest key column and not exact matches
# merged "on" columns must be sorted
# Takes nearest less than or equal to value
pd.merge_asof(visa, ibm, on='date_time', suffixes=('_visa','_ibm'))
pd.merge_asof(visa, ibm, on='date_time', suffixes=('_visa','_ibm'), direction='forward') #Changes to select first row to greater than or equal to
# nearest - sets to nearest regardless of whether it is forwards or backwards

# Useful when dates or times don't excactly align
# Useful for training set where do not want any future events to be visible

# Selecting data with .query()
""" 
Accepts an input string
-- Used to determine what rows are returned
-- Similar to a WHERE clause in an SQL statement"""
# Query on a single condition
stocks.query('nike>=90')
# Query on multiple conditions, 'and' 'or'
stocks.query('nike>90 and disney<140')
stocks.query('nike>90 or disney<140')
stocks_long.query('stock=="disney" or (stock=="nike" and close<90)') #Double quotes used to avoid unintentionally ending statement

# Reshaping data with .melt()
# Wide vs. long data
# Wide formatted easier to read by people
# Long format data more accessible for computers
# Melt allows unpivoting of dataset
social_fin_tall = social_fin.melt(id_vars=['financial','company'])
# ID vars are columns that we do not want to change

social_fin_tall = social_fin.melt(id_vars=['financial','company'], value_name=['2018', '2019'])
# Value vars controls which columns are unpivoted - output will only have values for those years

social_fin_tall = social_fin.melt(id_vars=['financial','company'], value_name=['2018', '2019'], var_name=['year'], \
    value_name='dollars')
