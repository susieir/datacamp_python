# Data science toolbox part 2
# Introduction to iterators
# Iterating with a for loop
# Can iterate over characters in a string with a for loop

for letter in 'Datacamp':
    print(letter)

# Iterating over a range object

for i in range(4):
    print(i)

# Iterators vs. iterables
# Iterable: Lists, strings, dictionaries, file connections - an object with an associated iter() method
# Applying iter() to an iterable creates an iterator

# Iterator - has associated next method
# produces next value with next()

# Iterating over iterables
word = 'Da'
it = iter(word)
next(it)  # Calls first value
next(it)  # Calls next value
next(it)  # And so on...

# Iterating at once with * (splat operator)
word = 'Data'
it = iter(word)
print(*it)  # Unpacks all elements of an itertor
# Once done, there are no more values to go through!

# Iterating over dictionaries
# Applying items method unpacks them

pythonistas = {'susie': 'irons', 'david': 'irons'}
for key, value in pythonistas.items():
    print(key, value)

# Iterating over file connections

file = open('file.txt')
it = iter(file)
print(next(it))  # Prints the first line of a file

# Playing with iterators
# enumerate() - add a counter to any iterable
# takes any iterable as argument, e.g. list
# returns special enumerate object

avengers = ['hawkeye', 'iron man', 'thor', 'quicksilver']
e = enumerate(avengers)
print(type(e))

# Consists of pairs of elements from the original iterable, along with index within iterable

e_list = list(e)  # Turns into list of tuples
print(e_list)

# Enumerate() and unpac
# Enumerate itself is also an iterable
# Enumerate is an index-value pair

for index, value in enumerate(avengers):
    print(index, value)

# Default behaviour is to begin indexing at 0
# Can alter with a second argument, start

for index, value in enumerate(avengers, start=10):
    print(index, value)

# Using zip() - accepts arbitrary number of iterables and returns with tuples

names = ['barton', 'stark', 'odinson', 'maxmioff']

# Zip object - iterator of tuples

z = zip(avengers, names)
print(type(z))

# Can turn into a list and print the list

z_list = list(z)
print(z_list)

# Returns tuple, containing first element of each lists
# Alternatively, could use a for loop to loop over the two objects and print the tuples

for z1, z2 in zip(avengers, names):
    print(z1, z2)

# Could have used the splat operator (*) to print all the elements

print(*z)

# Using iterators to load large files into memory
# Loading data in chunks - there can be too much to hold in memory
# Solution: load and process the data in chunks!
# Can use pandas function read_csv
# Specify chunk - chunk size

import pandas as pd

result = []  # Holds the result of the iteration (could also have used a 0 value int)
for chunk in pd.read_csv('data.csv', chunksize=1000):  # Each chunk is a dataframe
    result.append(sum(chunk['x']))  # Computes the sum of the column of interest and append to result
total = sum(result)
print(total)

# List comprehensions

# Populate a list with a for loop

nums = [12, 8, 21, 3, 16]
new_nums = []

for num in nums:
    new_nums.append(num + 1)
print(new_nums)

# But for loops are inefficient - both computationally and in terms of coding time

new_nums = [num + 1 for num in nums]
print(new_nums)

# Can write a list comprehension over any iterable, e.g. range object

result = [num for num in range(11)]
print(result)

# List comprehensions collapse loops for building lists into a single line
# Components
## Iterable
## Iterator variable (represent members of iterable)
## Output expression

# Nested loops (1)

pairs_1 = []
for num1 in range(0, 2):
    for num2 in range(6, 8):
        pairs_1.append(num1, num2)
print(pairs_1)

# Nested loops (2)

pairs_2 = [(num1, num2) for num1 in range(0, 2) for num2 in range(6, 8)]
print(pairs_2)

# Tradeoff - readability

# Advanced comprehensions
# Conditionals in comprehensions
# Conditionals on the iterable

ans = [num ** 2 for num in range(10) if num % 2 == 0]
print(list(ans))

# Conditions on the output expression

ans = [num ** 2 if num % 2 == 0 else 0 for num in range(10)]

# Dict comprehensions - create new dictionaries from iterables
# Use curly braces instead of square brackets

pos_neg = {num: -num for num in range(9)}

# Introduction to generator expressions
# Like list comprehension, but using () instead of []
# Creates a generator object
# Like a list comprehension but does not store the list in memory
# Does not construct the list, but is an object that can be iterated over, to produce elements as required

new_num = (2 * num for num in range(10))
print(new_num)

# Looping over a generator expression produces elements of the analagous list

result = (num for num in range(6))
for num in result:
    print(num)


# Can pass the generator to a function list to generate the list
# Can pass to the function next to iterate over its elements - 'lazy evaluation'
# Evaluation delayed until its answer is needed

# Generator functions
# Produce generator objects when called
# Defined like a regular function 'def'
# Yield a sequence of values instead of a single value
# Generates a value with a yield keyword

def num_sequence(n):
    """Generate values from = to n"""
    i = 0
    while i < n:
        yield i
        i += 1


result = num_sequence(5)
print(type(result))

for item in result:
    print(item)

# Summary - list comprehensions
# Basic: [output expression for iterator variable in iterable]
# Adanced: [output expression + conditional on output for iterator variable in iterator + conditional on iterable]

# Recaps:
# Zip - accepts arbitrary number of iterables and returns an iterator of tuples