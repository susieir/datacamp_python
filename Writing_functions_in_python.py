"""Chapter 1 - Best Practices"""

# Video 1: Docstrings
# Python best practice - easier to read, use and maintain
# String - written as firstline of a function
# Usually span multiple lines
# 5 key pieces of info:
import contextlib

"""
Description of what the function does
Description of the arguments, if any
Description of the return values, if any
Descrpition of errors raised, if any
Optional extra notes, or examples of usage"""

# Docstring formats
# Google style - most popular
# Numpydoc - most popular
# reStructured Text
# EpyText

# Google style
""" Description of what the function does, using imperative language

Args:
    arg_1 (str): Description of arg_1 that can break onto the next line
        if needed
    arg_2 (int, optional): Write optional when an argument has a default value

Returns:
    bool: Optional description of the return value
    Extra lines are not indented.
    
Raises:
    ValueError: Include any error types that the function intentionally
        raises.

Notes:
    See https://wwww.datacamp.com/community/tutorials/docstrings-python
    for more info.
"""

# Numpydoc - most common in scientific community

"""
Description of what the function does.
  
Parameters
----------
arg_1 : expected type of arg_1
    Description of arg_1
arg_2 : int, optional
    Write optional when an argument has a default value
    Default=42.
    
Returns
-------
The type of return value
    Can include a description of the return value.
    Replaces "returns" with "yields" if the function is a generator.
"""

# Sometimes useful for the code to access the contents of the functions doc string
# Every function in python comes with __doc__ attribute that holds this information
# e.g. print(the_answer.__doc__) - contains the raw docstring
# for a cleaner version - can use getdoc() .function from the inspect module
# e.g. print(inspect.getdoc(the_answer))

# DRY - Don't Repeat Yourself and "Do One Thing"

# Copying and pasting code
# Easy to make mistakes
# If you have to change something - you have to do it in multiple places
# Use functions to avoid repetition and make it easy to change things

# Another software engineering principle - "Do One Thing"
# Every function should have a single resonsibility
# Makes code more flexible
# Easier for other developers to understand
# Simpler to test
# Simpler to debug

# If not - creates code smells
# Refactoring - improving code by changing a little at a time

# Pass by assignment
# In python - integers are immutable - they can't be changed

# Immutable:
#  - int
#  - float
#  - bool
#  - string
#  - bytes
#  - tuple
#  - frozenset
#  - None

# Mutable:
#  - list
#  - dict
#  - set
#  - bytearray
#  - objects
#  - functions
#  - almost everything else!

# Mutable default arguments are dangerous!

"""Chapter 2 - Context Managers"""

# A context manager:
# - Sets up a context
# - Runs your code
# - Removes the context

# Caterers are like a context manager
# open() is a context manager:
"""
with open(my_file.txt) as my_file:
    text = my_file.read()
    length = len(text)
# Closes the file before running the print statement
print('The file is {} characters long'.format(length))
"""
# with - lets python know entering a context
# <context-manager>(<args>):
# - call a function, any function built to work as a context manager
# - it can take args
# - finish with colon
# as <variable-name> - assigns returned value to variable name

# Compound statements - have indented text (e.g. if loops etc.)
# Code inside the context manager needs to be indented

# Writing context managers

# Two ways to write a context manager
#  - Class-based
#  - Function-based  -> focus for this class

# 1. Define a function
# 2. Optional - add any setup code your function needs
# 3. Use the yield keyword
# 4. Optional - add any teardown code your context needs to clean up
# 5. Add the '@contextlib.contextmanager' decorator

@contextlib.contextmanager
def my_context():
    # Add any setup code you need
    print('hello')
    yield 42
    # Add any teardown code you need
    print('goodbye')

with my_context() as foo:
    print('foo is {}'.format(foo))

# Yield - going to return a value, but expect to finish the rest of the function at some point in the future

# Context manager function - technically a generator that yields a single value

# Context manager - allows developed to hide things such as connecting and disconnecting from a db
# Makes it simpler to just perform operations on the database

# Example 1
# Write context manager
@contextlib.contextmanager
def database(url):
    # set up database connection
    db = postgres.connect(url)

    yield db

    # tear down connection
    db.disconnect
# Use context manager
url = 'http://datacamp.com/data'
with database(url) as my_db:
    course_list = my_db.execute(
        'SELECT * FROM courses'
    )

# Example 2
# Write context manager
@contextlib.contextmanager
def in_dir(path):
    # save current working directory
    old_dir = os.getcwd()

    # switch to new working directory
    os.chdir(path)

    yield

    # change back to previous working directory
    os.chdir(old_dir)

# Use context manager
with in_dir('/data/project_1/'):
    project_files = os.listdir()

# Some do not need to return anything with its yield statement

# Advanced topics
# Nested contexts

def copy(src, dst):
    """Copy the contents of one file to another.

    Args:
        src (str): File name of the file to be copied
        dst (str): Where to write the new file.
       """
    # Open the source file and read in file contents
    with open(src) as f_src:
        contents = f_src.read()

    # Open the destination file and write out the contents
    with open(dst, 'w') as f_dst:
        f_dst.write(contents)

# This works well, until the file becomes too large to be stored in memory
# It would be better if could open both files at once and copy one line at a time

    with open('my_file.txt') as my_file:
        for line in my_file:  # Reads content one line at a time to end of file
            # Do something


# Improved version:
def copy(src, dst):
    """Copy the contents of one file to another.

    Args:
        src (str): File name of the file to be copied
        dst (str): Where to write the new file.
       """
    # Open both files
    with open(src) as f_src:
        with open(dst, 'w') as f_dst:
            # Read and write each line, one at a time
            for line in f_src:
                f_dst.write(line)

# Nested context manager means both objects can be accessed within the code

# Handling errors
# Watch out for errors - may prevent the context manager from disconnecting / performing teardown code

# Can use:
# try: -- Code that might raise error
# except: -- do something about the error
# finally: -- this code runs no matter what

def get_printer(ip):
    p = connect_to_printer(ip)

    try:
        yield
    finally:  # Ensures disconnect is called even if an error is raised
        p.disconnect()
        print('disconnected from printer')

# Context manager patterns:
# Open / Close
# Lock / Release
# Change / Reset
# Enter / Exit
# Start / Stop
# Setup / Teardown
# Connect / Disconnect

"""Chapter 3 - Decorators"""
# Functions as objects
# Functions are just another type of object
# Can do anything to or with functions that can be done to any other type of object
# Can assign to a variable

def my_function():
    print('Hello')
x = my_function
type(x)
x()  # Can call x instead of my_function

# E.g. can assign print to printy_mcprintface

# Lists and dictionaries of functions
list_of_functions = [my_function, open, print]
list_of_functions[2]('I am printing with an element of a list!')
# Can call an element of the list and pass in arguments

# Can do the same for dicts
dict_of_functions = {
    'func1' : my_function,
    'func2' : open,
    'func3' : print
}

dict_of_functions['func3']('I am printing with an element of a list!')

# When assigning - do not include the parentheses after the function name
# Without the parentheses - references the function itself
# With the parentheses - calling the function

# Can pass a function as an argument to another function
# Functions can be defined inside other functions - nested functions / inner functions / helper functions / child functions
# Can use functions as return values

# Scope
# Determines which variables can be accessed at each point in your code
# Local scope - args, variables defined inside function
# Non-local - in the case of nested functions, e.g. in parent function
# Global scope - args, variables defined outside function
# Builtin scope - if not in local or global scope, always available. E.g. print function

# Try to avoid using global variables if possible - it can make testing and debugging harder

# Closures
# Tuple of variables that are no longer in scope but that a function needs in order to run
# Stored in __closure__ attribute of function

def foo():
    a = 5
    def bar():
        print(a)
    return bar

func = foo()

func()

# a is non-local, shouldn't have been observable in bar, but was stored as closure

func.__closure__[0].cell_contents  # Allows you to access the contents

x = 25

def foo(value):
    def bar():
        print(value)
    return bar

my_func = foo(x)
my_func  # Returns 25

del(x)
my_func()  # Still returns 25, stored in closure

my_func.__closure__[0].cell_contents  # Returns 25
len(my_func.__closure__)

# Nested function - function defined inside another function - parent / child
# Non-local variable - variable defined in the parent function that gets used by the child function
# Closure - python's way of attaching non-local variables to a returned function, so that the function can operate
    # even when it is called outside of it's parent's scope

# If del(x), and re-run then 25 is stored in closure

# Decorators
# A wrapper that you can place around a function, that changes that functions behaviour
# Can modify inputs, outputs or change the behaviour of the function itself

@double_args  # Modifies behaviour of the multiply function
def multiply(a, b):
    return a * b

# Creating the construct

def multiply(a, b):
    return a * b
def double_args(func):
    # Define a new function we can modify
    def wrapper(a, b):
        # For now, just call modified function
        return func(a, b)
    # Return the new function
    return wrapper
new_multiply = double_args(multiply)

# Modifying the function
def multiply(a, b):
    return a * b
def double_args(func):
    # Define a new function we can modify
    def wrapper(a, b):
        # Call the passed in function, but double each argument
        return func(a * 2, b * 2)
    return wrapper
new_multiply = double_args(multiply)
new_multiply(1, 5)  # Returns 20

# Can instead overwrite the multiply variable
multiply = double_args(multiply)
# Original multiple function stored in the functions closure

# Alternative way of doing this:
def double_args(func):
    def wrapper(a, b):
        return func(a * 2, b * 2)
    return wrapper

@double_args
def multiply(a, b):
    return a * b

multiply(1, 5)

"""Chapter 4 - More decorators"""
# Real world examples
# Note - all decorators take and return a single function
# Time a function

import time

def timer(func):
    """ A decorator that prints how long a function took to run.

    Args:
        func (callable): The function being decorated.

    Returns:
        callable: The decorated function
        """
    # Define the wrapper function to return
    def wrapper(*args, **kwargs):  # Takes any number of positional and keyword arguments, so can be used to decorate any func
        # When wrapper() is called, get the current time.
        t_start = time.time()
        # Call the decorated function and store the result
        result = func(*args, **kwargs)
        # Get the total time it took to run, and print it
        t_total = time.time() - t_start
        print('{} took {}s'.format(func.__name__, t_total))
        return result
    return wrapper

@timer
def sleep_n_seconds(n):
    time.sleep(n)

sleep_n_seconds

def memoize(func):
    """Store the results of the decorated function for fast lookup
    """
    # Store results in a dict that maps arguments to results
    cache = {}
    # Define the wrapper function to return
    def wrapper(*args, **kwargs):
        # If these arguments haven't been seen before,
        if (args, kwargs) not in cache:
            # Call func and store the result.
            cache[(args, kwargs)] = func(*args, **kwargs)
        return cache[(args, kwargs)]
    return wrapper

# When to use decorators
#  - Add common behaviour to multiple functions

# Decorators and metadata

def sleep_n_seconds(n=10):
    """Pause processing for n seconds.

    Args:
        n (int): The number of seconds to pause for:
    """
    time.sleep(n)
print(sleep_n_seconds.__doc__)  # Displays docstring
print(sleep_n_seconds.__name__)  # Displays name
print(sleep_n_seconds.__defaults__)  # Displays default args

# When decorated e.g. with @timer, obscures metadata

# Wraps tool - decorator you use when defining a decorator (wrapper function)

from functools import wraps
def timer(func):
    """A decorator that prints how long a function took to run"""

    @wraps(func)  # Takes func decorating as arg
    def wrapper(*args, **kwargs):
        t_start = time.time()
        result = func(*args, **kwargs)
        t_total = time.time() - t_start
        print('{} took {}s'.format(func.__name__, t_total))
        return result
    return wrapper

# Original function can be accessed via .__wrapped__

# Decorators that take arguments

def run_three_times(func):  # Runs any function three times
    def wrapper(*args, **kwargs):
        for i in range(3):
            func(*args, **kwargs)
    return wrapper

# Want to pass n as an argument
# Need to turn it into a function that returns a decorator - a decorator factory!

def run_n_times(n):
    """Define and return a decorator"""
    def decorator(func):  # Function that will be acting as our decorator
        def wrapper(*args, **kwargs):
            for i in range(n):
                func(*args, **kwargs)
        return wrapper
    return decorator

@run_n_times(3)  # Calling run n times and decorating with the result of the function
def print_sum(a, b):
    print(a + b)

# @ must be reference to decorator
# Can use a decorator function
# Or can call a function that returns a decorator

# Timeout
# Want a timeout which will raise an error if a function runs for longer than expected

import signal
def raise_timout(*args, **kwargs):
    raise TimeoutError()
# When an 'alarm' signal goes off, call raise_timout()
signal.signal(signalnum=signal.SIGALRM, handler=raise_timeout)  # When see signal, whose num is signalnum, call handler func
# Calls raise_timeout, whenever sees alrm signal
# Set off an alarm n 5 seconds
signal.alarm(5)
# Cancel the alarm
signal.alarm(0)


def foo():
    time.sleep(10)
    print('foo!')

def timeout(n_seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
        # Set an alarm for n seconds
            signal.alarm(n_seconds)
            try:
                # Call the decorated func
                return func(*args, **kwargs)
            finally:
                # Cancel alarm
                signal.alarm(0)
        return wrapper
    return decorator


