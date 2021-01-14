""" Data visualisation
	- Explore the dataset
	- Report insights
• Matplotlib
	- Pyplot
• Line plot
	- Plt.plot(year, pop) --> tells what to plot and how to plot
	- Plt.show() --> Displays plot
• Scatter plot
	- Plt.scatter(year, pop, s=size, c=colour)
• Putting into logarithmic scale
	- Plt.xscale('log')
• Histogram
	- Explore dataset
	- Get idea about distribution
	- Divides dataset into bins - shows number of datapoint in bin
	- X = list of values want to build hist for
	- Bins - number of bins data is divided into - default = 10
	- Plt.hist(values, bins=3)
	- Plt.clf() - allows multiple charts to be shown (???)
• Data visualisation
	- Axis labels
		○ Plt.xlabel('Year')
		○ Plt.xlabel('Pop')
		○ Must call before show function
	- Titles
		○ Plt.title('XYZ')
	- Plt.yticks([0,2,4,6,8,10], [0, 2BN, 4BN, 6BN, 8BN, 10BN])
		○ Sets axis ticks
		○ Second list - display names
	- Add more data
		○ year = [x, y, z] + year
		○ Pop = [x,y,z] + year
	- Plt.grid(True)
• Dictionaries
	- Can use as a lookup
	- World = {"afghanistan" :30.55, "albania":2.77, "algeria":39.21}
	- World["albania"]
	- Very efficient
	- Keys must be immutable objects - cannot be changed after creation
		○ Strings, booleans, integers and floats are immutable
		○ Lists are mutable
	- Adding to dictionary
		○ World["sealand"] = 0.00027
		○ Check: "sealand" in world
	- Deleting from dictionary
		○ Del(world["sealand"])
	- List vs. Dictionary
		○ List - sequence of numbers, indexed by range of numbers - useful if order matters, with the ability to select entire subsets
		○ Dictionary - Indexed by unique keys, lookup table with unique keys, quick to lookup
• Pandas
	- Rectangular dataset -
	- Numpy an option
	- Pandas package
		○ Data manipulation tool, built on numpy
		○ Data stored in Dataframe
	- Can build dataframe manually from dictionary
	- Pd.DataFrame(dict)
	- Manually label index
		○ Brics.index = ["BR", "RU",...]
	- Importing data
		○ Brics.csv
		○ Brics = pd.read_csv("path/to/brics.csv", index_col = 0)
		○ To tell dataframe that row index is first column --> index_col = 0
• Pandas 2
	- Column access
		○ Brics["country"]
		○ Series - 1D labelled array
		○ Brics[["country"]]
		○ Dataframe
		○ Can extend to two columns - subdataframe
	- Row access
		○ Brics[1:4] -> slice
		○ End of slice exclusive
		○ Index starts 0
	- Loc
		○ Select parts of table based on labels
		○ Brics.loc["RU"]
			§ Row as pandas series
		○ Brics.loc[["RU", "IN", "CH"]] --> selected rows
		○ Brics.loc[["RU","IN","CH"],["country","capital"]] --> subset
		○ Brics.loc[:, ["country", "capital"]] --> returns all columns for two countries
	- Iloc
		○ Based on position
		○ Brics.iloc[[1,2,3]]
		○ Brics.iloc[[1,2,3],[0,1]
		○ Brics.iloc[:,[0,1]]
• Logic, control flow and filtering
	- Comparison operators
		○ Operators that can tell how too values relate and result in a boolean
		○ Integer and string - not comparable - can't tell how objects of different types relate
		○ Floats and ints are the exceptions
		○ != --> not equal
		○ == --> equal
	- Boolean operators
		○ And
			§ X>5 and x<15 --> True and True --> True
		○ Or
			§ Y=5
			§ Y < 7 or y>13
			§ True
		○ Not
			§ Negates the boolean value its used on
			§ Not True = False
			§ Not False = True
	- NumPy - doesn't want an array of booleans, array equivalents:
			§ Logical_and()
			§ Logical_or()
			§ Logical_not()
			§ Np.logical_and(bmi > 21, bmi < 22)
			§ Operation is performed element-wise
			§ To select the array:
				□ Bmi[np.logical_and(bmi>21,bmi<22)]
	- If, elif, else
			§ Z=4
			§ If z % 2 == 0 :
				□ Print("z is even")
			§ Elif z % 3 == 0 :
				□ Print("z is divisible by 3")
			§ Else :
				□ Print("z is divisible by 2 nor 3")
			§ If first condition satisfied, second condition is never reached
	- Filtering pandas dataframes
			§ Get column --> brics["area"] --> want series
			§ Is_huge = Brics["area"] > 8 --> Get a boolean
			§ Brics[is_huge] --> selects countries with an area greater than 8
			§ Shorten: brics[brics["area"] > 8]
		○ Boolean operators
			§ Brics[np.logical_and(brics["area"] > 8, brics["area"] < 10)]




"""

# While loop = repeated if statement
""" 
syntax:
while condition:
    expression"""

# Repeating an action until a condition is met

error = 50.0
while error > 1:
    error = error / 4
    print(error)

""" For Loop
for var in seq :
    expression"""

# Simple for loop
fam = [1,2,3,4]
for height in fam :
    print(height)

# Displays index
for index, height in enumerate(fam) :
    print("index" + str(index) + ":" str(height))

# Could iterate over every character in a string

for c in "family" :
    print(c.capitalize())

# Loop data structures - part 1

world = {"afghanistan" : 30.55,
         "albania" : 2.77,
         "algeria" : 39.21}

for key, value in world.items()
    print(key + "--" + str(value))

# Dictionaries are inherently unordered
# key and values are arbitrary names

#2D numpy arrays

import numpy as np
np_height = np.array([1.73, 1.68, 1.71, 1.89, 1.79])
np_weight = np.array([65.4, 59.2, 63.6, 88.4, 68.7])
meas = np.array([np_height, np_weight])

for val in np.nditer(meas) :
    print(val)

# Dictionary -- for key, val in my_dict.items() :
# Numpy array -- for val in np.nditer(my_array) :


# Loop datastructures part 2 - Pandas DF

for val in brics :
    print(val)
# Will only print column headers

for lab, row in brics.iterrows():
    print(lab)
    print(row)
# Prints entire series

for lab, row in brics.iterrows() :
    print(lab + ": " + row["capital"])

# To add additional column with length of country name
for lab, row in brics.iterrows() :
    # Creating series on every iteration
    brics.loc[lab, "name length"] = len(row["country"])
# Can be inefficient as creating for each row

# Apply is more efficient
    brics["name_length"] = brics["country"].apply(len) # --> creates new array
    print(brics)

# Random numbers - import numpy
# np.random.rand()  ---> #Pseudo random numbers

np.random.seed(123)
np.random.rand() # Ensures 'reproduceability

# Coin toss
import numpy as np
np.random.seed(123)
coin = np.random.randint(0,2) # Randomly generates 0 or 1
print (coin)

if coin == 0 :
    print("heads")
else :
    print("tails")

# Random walk - succession of random steps, e.g. path of molecules
# Building list based on outcomes

import numpy as np
np.random.seed(123)
outcomes = [] #Initialise an empty list
for x in range(10) :
    coin = np.random.randint(0,2) # Random coin toss
    if coin == 0 :
        outcomes.append("heads")
    else :
        outcomes.append("tails")
    print(outcomes)

# Generating a random walk

import numpy as np
np.random.seed(123)
tails = [0]
for x in range(10) :
    coin = np.random.randint(0,2)
    tails.append(tails[x] + coin)
print(tails)

# Distribution of random walks

import numpy as np
np.random.seed(123)
tails = [0]
for x in range(10):
    coin = np.random.randint(0, 2)
    tails.append(tails[x] + coin)

#100 runs

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(123)
final_tails = []

for x in range(100000) :
    tails = [0]
    for x in range(10) :
        coin = np.random.randint(0,2)
        tails.append(tails[x] + coin)
    final_tails.append(tails[-1])
plt.hist(final_tails,bins=10)
plt.show()