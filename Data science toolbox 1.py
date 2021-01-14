# User-defined functions

# Built in functions
# str()  # Accepts object, returns string object


# Defining a function


def square():  # <- Function header
    new_value = 4 ** 2  # <- Function body
    print(new_value)


print(2 * 6)

square()


# Function parameters


def square(value):
    new_value = value ** 2
    print(new_value)


square(4)


# Return a value from a function


def square(value):
    new_value = value ** 2
    return new_value


num = square(4)

print(num)


# Docstrings - used to describe what your function does, serve as documentation for function
# Placed in the immediate line after the function header
# In between triple double quotes """ """

def square(value):
    """Return the square of a value."""
    new_value = value ** 2
    return new_value


# Return lets you return values from functions - print() has NoneType

# Multiple parameters and return values
# Multiple function parameters


def raise_to_power(value1, value2):
    """Raise value1 to the power of value2"""
    new_value = value1 ** value2
    return new_value


# Call function: # of arguments = # of parameters


result = raise_to_power(2, 3)

print(result)

# Making functions return multiple values - tuples!
# Tuples - like a list, can contain multiple values
# Immutable - can't modify values! (unlike a list)
# Constructed using parentheses () (unlike a list - [])

even_nums = (2, 4, 6)
a, b, c = even_nums  # Assigns the values in the oder they appear

# Accessing individual elements - as with lists

print(even_nums[1])

second_num = even_nums[1]
print(second_num)


# Tuples use zero indexing

# Returning multiple values


def raise_both(value1, value2):
    """Raise value1 to the power of value2 and vice versa"""

    new_value1 = value1 ** value2
    new_value2 = value2 ** value1

    new_tuple = (new_value1, new_value2)

    return new_tuple


# Scope and user defined functions
# Not all objects are accessible everywhere in a script
# Scope - part of program where an object or name may be accessed
# Global scope - defined in main body of the script
# Local scope - defined in a function
# Built-in scope - names in the pre-defined built-ins module provided by Python (e.g. sum())

# Global vs. local scope
# If python cannot find the name in the local scope, it will then look in the global scope

# If want to alter the value of a global name within a function, use global function


def square(value):
    """Returns square of a number"""
    global new_val  # Allows us to access and alter glboal variable
    new_val = new_val ** 2
    return new_val


# Nested functions


def mod2plus5(x1, x2, x3):
    """Returns the remainder plus 5 of three values"""

    def inner(x):
        """Returns the remainder plus 5 of a value"""
        return x % 2 + 5

    return (inner(x1), inner(x2), inner(x3))


print(mod2plus5(1, 2, 3))


# Returning functions


def raise_val(n):
    """Return the inner function"""

    def inner(x):
        """Raise x to the power of n"""
        raised = x ** n
        return raised

    return inner


square = raise_val(2)
cube = raise_val(3)
print(square(2), cube(4))


# Using nonlocal

def outer():
    """Prints the value of n."""
    n = 1

    def inner():
        nonlocal n
        n = 2
        print(n)

    inner()
    print(n)

outer()

# Scopes searches: local, enclosing functions, global, built-in [LEGB rule]

# Default and flexible arguments
# Some parameters have arguments used when not specified otherwise
# Flexible arguments - allow to pass any number of arguments to a function


def power(number, pow=1):  # Default function followed with = and default value
    """Raise number to power of pow"""
    new_value = number ** pow
    return new_value

power(9, 2)

power(9)  # Assumes pow=1

# Flexible arguments


def add_all(*args):  # Turns all args into a tuple called args
    """Sum all values in *args together."""

    # Initialise sum
    sum_all = 0

    # Accumulate the sum
    for num in args:  # Loops over the tuple args
        sum_all += num  # Adds each element successively to sum_all

    return sum_all

add_all(5, 10, 15, 20)

# Flexible arguments: **kwargs - keyword arguments, args preceded by identifiers


def print_all(**kwargs):  # Turns identifier keyword pairs into dictionary within function body
    """Print out key-value pairs in **kwargs"""

    # Print out the key value pairs
    for key, value in kwargs.items():
        print(key + \": \" + value)  # Prints key value pairs stored in the dictionary **kwargs

print_all(name = "Susannah Irons", employer = "SAS")

# Lambda functions - use keyword lambda

raise_to_power = lambda x, y: x ** y  # Name of args, then ":", followed by expression

raise_to_power(2, 3)

# Anonymous functions
# Function map that takes two arguments map(func, seq)
# map() applies the function to ALL elements in sequence

nums = [48, 6, 9, 21, 1]

square_all = map(lambda num : num ** 2, nums)

print(square_all)  # Printing reveals it is a map object

print(list(square_all))  # Need to use list to print as list


# Introduction to error handling
# Exceptions - caught during execution
# Catch exceptions with try-except clause
# Runs the code following try
# If there's an exception, runs the code following except

def sqrt(x):
    """Returns the square root of a number"""
    try:
        return x ** 0.5
    except:
        print('x must be an int or float')


# If only want to capture type erros

def sqrt(x):
    """Returns the square root of a number"""
    try:
        return x ** 0.5
    except TypeError:
        print('x must be an int or float')

# If want to raise an error, e.g. if don't want complex numbers

def sqrt(x):
    """Returns the square root of a number"""
    if x < 0:
        raise ValueError('x must be non-negative')
    try:
        return x ** 0.5
    except TypeError:
        print('x must be an int or float')


