# Introduction to Seaborn
# Python data visualisation library
# Allows data exploration and communication of results
"Advantages: Easy to use, works well with pandas, built on matplotlib"
import seaborn as sns
# Samuel Norman Seaborn (West Wing)
import matplotlib.pyplot as plt

height = [62, 64, 69, 75, 66, 68, 65, 71, 76, 73]
weight = [120, 136, 148, 175, 137, 165, 154, 172, 200, 187]
sns.scatterplot(x=height, y=weight)
plt.show()

# Count plot
# Shows number of list entries per category
import seaborn as sns
import matplotlib.pyplot as plt

gender = ["Female", "Female", "Female", "Female", "Male", "Male", "Male", "Male", "Male", "Male"]
sns.countplot(x=gender)

# Using pandas with seaborn
# Easily read files, data analysis library
# Dataframe - most common object

import pandas as pd

df = pd.read_csv("masculinity.csv")
df.head()  # Shows first 5 rows
# Using dataframes with countplot

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.read_csv("masculinity.csv")
sns.countplot(x="how_masculine", data=df)  # Column name automatically added as x-axis
plt.show()

# Sns only works with tidy data - each obs has its own row, each variable has its own column
# Untidy data needs transforming to work

# Adding a third variable with hue
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

hue_colors = {"Yes": "black", "No": "red"}
tips = sns.load_dataset("tips")
sns.scatterplot(x="total_bill", y="tip", data=tips, hue="smoker", hue_order=["Yes", "No"], palette=hue_colors)
plt.show()
# Can use html colour codes - put in quotes with # at beginning
# Can use hue in multiple types of plots

import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x="smoker", data=tips, hue="sex")

plt.show()

# Introduction to relational plots and subplots
# Visualises relationship between two quantitative variables
# Relplot() - create scatter or line plots, can create subplots in a single figure

import seaborn as sns
import matplotlib.pyplot as plt

sns.relplot(x="total_bill",
            y="tip",
            data=tips,
            kind="scatter",
            col="smoker",  # Get separate columns - two plots one for smoker and one for non-smoker
            row="time")  # Can use row/col separately or together
plt.show()

# Using colwrap

import seaborn as sns
import matplotlib.pyplot as plt

sns.relplot(x="total_bill",
            y="tip",
            data=tips,
            kind="scatter",
            col="day",  # Get separate columns - two plots one for smoker and one for non-smoker
            col_wrap=2,  # Max 2 per col/row?
            col_order=["Thur", "Fri", "Sat", "Sun"])  # Give list of ordered values
plt.show()

# Customising scatter plots
# Subgroups with point size

import seaborn as sns
import matplotlib.pyplot as plt

sns.relplot(x="total_bill",
            y="tip",
            data=tips,
            kind="scatter",
            size="size",  # Best if qualitative variable or categories of something
            hue="size")  # Makes easier to read by using in combination with size - uses shades

plt.show()

# Point style

import seaborn as sns
import matplotlib.pyplot as plt

sns.relplot(x="total_bill",
            y="tip",
            data=tips,
            kind="scatter",
            hue="smoker",
            style="smoker")

plt.show()

# Changing point transparency

import seaborn as sns
import matplotlib.pyplot as plt

sns.relplot(x="total_bill",
            y="tip",
            data=tips,
            kind="scatter",
            alpha=0.4)  # Varies transparency

plt.show()

# Introduction to line plots
# Scatter plot - each point an independent observation
# Line plot - tracking the same thing over time

import matplotlib.pyplot as plt
import seaborn as sns

sns.relplot(x="hour",
            y="NO_2_mean",
            data=air_df_mean,
            kind="line")

plt.show()

# Subgroups by location

import matplotlib.pyplot as plt
import seaborn as sns

sns.relplot(x="hour",
            y="NO_2_mean",
            data=air_df_mean,
            kind="line",
            style="location",
            hue="location",
            markers=True,  # Shows marker for each point
            dashes=False)  # Don't want line style to vary by subgroup

plt.show()

# Multiple observations per x-value
# If given multiple observations per value it will aggregate into a singular summary measure
# Default - display mean
# Automatically calc confidence interval for mean - shaded region (95%) - indicate uncertainty in estimate
# Assumes dataset is a random sample

import matplotlib.pyplot as plt
import seaborn as sns

sns.relplot(x="hour",
            y="NO_2_mean",
            data=air_df_mean,
            kind="line")

plt.show()

# Replacing confidence intervals with standard deviation
# Shows spread of distribution of observations at each x value

import matplotlib.pyplot as plt
import seaborn as sns

sns.relplot(x="hour",
            y="NO_2",
            data=air_df,
            kind="line",
            ci="sd")  # Replaces CI with STDDEV

plt.show()

# Turning off confidence interval

import matplotlib.pyplot as plt
import seaborn as sns

sns.relplot(x="hour",
            y="NO_2",
            data=air_df,
            kind="line",
            ci=None)  # Turns off CI

plt.show()

# Visualising a categorical and quantitative variable
# Count plots and bar plots
# Categorical plots - involve categorical variable, fixed, typically small number of categories
# Commonly used for comparisons between different groups
# Count plots - number of obs within each category
# catplot() - creates types of categorical plots - same flex as relplot(), easier to create subplots

import matplotlib.pyplot as plt
import seaborn as sns
category_order=["No answer",
                "Not at all",
                "Not very",
                "Somewhat",
                "Very"]

sns.catplot(x="how_masculine",
            data=masculinity_data,
            kind="count",
            order=category_order)  # Works for all categorical plots

plt.show()

# Bar plots - display mean of quant variable per category

import matplotlib.pyplot as plt
import seaborn as sns

sns.catplot(x="day",
            y="total_bill",
            data=tips,
            kind="bar",
            ci=None)  # Turns off ci

plt.show()

# Automatically shows confidence intervals for means (as for line plots)
# Can change orientation by switching x and y parameters
# When y variable is true/false - bar plots will show the percentage of True responses

# Creating a box plot
# SHows distribution of quant data
# Box - 25th to 75th percentile
# Line in middle - median
# Whiskers - spread
# Floating points - outliers
# Facilitates comparison between groups

import matplotlib.pyplot as plt
import seaborn as sns

sns.catplot(x="time",
            y="total_bill",
            data=tips,
            kind="box",
            order=["Dinner", "Lunch"],
            sym="")  # Omits outliers, can also be used to change appearance of outliers

plt.show()

# Whiskers - by default - 1.5x IQR
# whis=2.0 # Changes to 2x IQR
# whis=[5,95] # 5th to 95th percentile
# whis=[0,100] # Shows min and max values

# Point plots
# Points show mean of quantitative variable
# Vertical bars - 95% CI for mean
# Line plots = relational
# Point plot = categorical
# Point plot = easier to compare subgroups and differences between categories

import matplotlib.pyplot as plt
import seaborn as sns

sns.catplot(x="age",
            y="masculinity_important",
            data=masculinity_data,
            hue="feel_masculine",
            kind="point",
            join=False)  # Removes lines connecting each category

plt.show()

# Displaying the median

import matplotlib.pyplot as plt
import seaborn as sns
from numpy import median

sns.catplot(x="smoker",
            y="total_bill",
            data=tips,
            kind="point",
            capsize=0.2,  # Adds caps to end of CIs
            estimator=median)  # Numpy median function, useful if lots of outliers


plt.show()

# Changing plot style and color
# 5 pre-set figure styles
# Preset:
# "white",
# "dark" (grey background),
# "whitegrid" (gridlines)
# "darkgrid" (grey background with gridlines), "ticks" (small tickmarks to x and y axes)
# Setting global style for all plots
sns.set_style()

# Figure palette changes the color of the main elements of the plot
sns.set_palette()
# Use preset or create custom
# Preset "diverging" palettes - useful for diverging scales, or two ends opposite with neutral in middle
# "RdBu", "PRGn", "RdBu_r", "PRGn_r" - _r reverses palette

# Sequential palettes - moving from light to dark values
# "Greys", "Blues", "PuRd", "GnBu"
# Useful for variable on a continuous scale

custom_palette = ["red", "green", "orange", "blue", "yellow", "purple"]  # Can also use HEX colors
sns.set_palette(custom_palette)

# Figure "context" changes scale of plot elements and labels
sns.set_context()
# Smallest to largest: "paper", "notebook", "talk", "poster"
# Default - "paper"
# "Talk" better for presentations

# Adding titles and labels
# Seaborn plots create two different types of objects: FacetGrid and AxesSubplot
# To work out which you're dealing with assign the plot output to a variable, g usually used

g = sns.scatterplot(x="height", y="weight", data=df)
type(g)

# FacetGrid - supports subplots (relplot(), catplot())
# AxesSubplot - supports single plot (scatterplot(), countplot() etc.)

# Adding title to FacetGrid

g = sns.catplot(x="Region",
                y="Birthrate",
                data=gdp_data,
                kind="box")
g.fig.suptitle("New Title",  # Title for figure as whole
                y=1.03)  # y parameter adjusts height, default is 1 (sometimes low by default)

plt.show()

# Adding titles and labels part 2
# Adding titles to an AxesSubplot

g = sns.boxplot(x="Region",
                y="Birthrate",
                data=gdp_data)

g.set_title("New Title",
            y=1.03)

# Titles for subplots

g = sns.catplot(x="Region",
                y="Birthrate",
                data=gdp_data,
                kind="box",
                col="Group")

g.fig.suptitle("New Title",  # Title for figure as whole
                y=1.03)  # y parameter adjusts height, default is 1 (sometimes low by default)

g.set_titles("This is {col_name}")  # Sets titles for subplots using col name

# Adding axis labels

g = sns.catplot(x="Region",
                y="Birthrate",
                data=gdp_data,
                kind="box")

g.set(xlabel="New X Label", ylabel="New Y Label")  # Works with both FacetGrid and AxesSubplot

# Rotating x axis tick labels

g = sns.catplot(x="Region",
                y="Birthrate",
                data=gdp_data,
                kind="box")
plt.xticks(rotation=90)
plt.show()

# Putting it all together
