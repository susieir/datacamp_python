# Intermediate data visualisation with Seaborn

# Seaborn uses matplotlib to generate statistical visualisations
# Panda is a foundational library for analysing data, it also supports basic plotting capability
# Seaborn supports complex visualisations of data - it is build on matplotlib and works best with pandas' dataframes
# The distplot is similar to a histogram - by default generates a Guassian Kernel Density Estimate (KDE)

import seaborn as sns
sns.distplot(df['alcohol''])

# Displot has multiple optional arguments
# In order to plot a simple histogram, you can disable the KDE and specify the number of bins to use
# E.g. Creating a simple histogram

sns.distplot(df['alcohol']), kde=False, bins=10)

# Alternative data distributions
# A rug plot is an alternative way to view the distribution of data
# A kde plot and rug plot can be combined

sns.distplot(df['alcohol'], hist=False, rug=True])

# It is possible to further customise a plot by passing arguments to the underlying function

sns.distplot((df['alcohol'], hist=False, rug=True, kde_kws={'shade':True})) # Passing kde kewords dictionary - used to shade

# Regression plots in Seaborn
# Histogram - univariate analysis
# Regression - bi-variate analysis
# regplot function generates a scatterplot with a regression line
# Usage is similar to distplot
# data and x and y variables must be defined

sns.regplot(x="alcohol", y="pH", data=df)

# lmplot() builds on top of the base regplot()
# regplot() - low level
# lmplot() - high level

# lmplot faceting
# Organise data by hue

sns.lmplot(x="quality", y="alcohol", data=df, hue="type")

# Organise data by col

sns.lmplot(x="quality", y="alcohol", data=df, col="type")

# Using seaborn styles
# Seaborn has default configurations that can be applied with sns.set()
# These styles can override matplotlib and pandas plots as well

sns.set()  # Sets default theme - also called dark grid
df['Tuition'].plot.hist()

# Theme examples with sns.set_style()

for style in ['white', 'dark', 'whitegrid', 'darkgrid', 'ticks']:
    sns.set_style(style)
    sns.distplot(df['Tuition'])
    plt.show()

# Removing axes with despine()

sns.set_style('white')
sns.distplot(df['Tuition'])
sns.despine(left=True)

# Colors in seaborn
# Seaborn supports assigning colors using matplotlib color codes

sns.set(color_codes=True)
sns.distplot(df['Tuition'], color=g)

# Palettes
# Seaborn uses the set_palette() function to define a palette
# Can set a palette of colours that can be cycled through in a plot

for p in sns.palettes.SEABORN_PALETTES:
    sns.set_palette(p)
    sns.distplot(df['Tuition'])

# sns.palplot() function displays a palette
# sns.color_palette() function returns the current palette

for p in sns.palettes.SEABORN_PALETTES:
    sns.set_palette(p)
    sns.palplot(sns.color_palette())
    plt.show()

# Displays a palette in swatches in a jupyter notebook

# Defining custom palettes
# Circular colors = when the data is not ordered
sns.palplot(sns.color_palette("Paired", 12))
# Sequential colors = when the data has a consistent range from high to low
sns.palplot(sns.color_palette("Blues", 12))
# Diverging colors = when both the high and low values are interesting
sns.palplot(sns.color_palette("BrBG", 12))

# Customizing with matplotlib
# Most customisation is available through matplotlib Axes objects
# Axes can be passed to seaborn functions

import matplotlib as plt

fig, ax = plt.subplots()
sns.distplot(df['Tuition'], ax=ax)
ax.set(xlabel="Tuition 2013-14")  # Customises x data

fig, ax = plt.subplots()
sns.distplot(df['Tuition'], ax=ax)
ax.set(xlabel="Tuition 2013-14",
       ylabel="Distribution",
       xlim=(0, 5000),
       title="2013-14 Tuition and Fees Distribution")

# It is possible to combine and configure multiple plots

fix, (ax0, ax1) = plot.subplots(
    nrows=1, ncols=2, sharey=True, figsize=(7,4))

sns.distplot(df['Tuition'], ax=ax0)
sns.distplot(df.query("State == 'MN'")['Tuition'], ax=ax1)  # Only plots data for the state of MN

ax1.set(xlabel = "Tuition (MN)", xlim=(0, 7000))
ax1.axvline(x=20000, label="My Budget", linestyle="--")  # Shows the max amount that can be budgeted for tuition
ax1.legend()

# Categorical plot types
# Categorical data = takes on a limited and fixed number of values
# Normally combined with numeric data
# Examples include - geography, gender, ethnicity, blood type, eye color

# Plot types
# Show each observation - Stripplot and swarmplot
# Abstract representations - Boxplot, violinplot, lvplot
# Statistical estimates - barplot, pointplot, countplot

# Stripplot - shows every observation in the dataset, can be difficult to see individual datapoints

sns.stripplot(data=df, y='DRG Definition',
              x='Average Covered Charges',
              jitter=True)

# Swarmplot - more sophisticated visualisation of all the data

sns.swarmplot(data=df, y="DRG Definition",
              x="Average Covered Charges")  # Places observations in a non-overlapping manner
# Does not scale well to large datasets

# Boxplot - used to show several measures related to dist of data incl median, upper and lower quartiles and outliers

sns.boxplot(data=df, y="DRG Definition",
            x="Average Covered Charges")

# Violinplot - combination of kernel density plot and boxplot, suitable for providing an alternative view of the dist of data

sns.violinplot(data=df, y="DRG Definition",
               x="Average Covered Charges")
# As uses a kernel density function, does not show all datapoints
# Useful for large datasets, can be computationally intensive to create

# lvplot - letter value plot

sns.lvplot(data=df, y="DRG Definition",
           x="Average Covered Charges")

# API same as boxplot and violin plot
# Hybrid between boxplot and violin plot
# Relatively quick to render and easy to interpret

# Barplot - shows estimate of value and confidence interval

sns.barplot(data=df, y="DRG Definition",
            x="Average Covered Charges",
            hue="Region")

# Pointplot - similar to barplot, shows summary measure and confidence interval
# Can be useful for observing how values change across categorical values

sns.pointplot(data=df, y="DRG Definition",
              x="Average Covered Charges",
              hue="Region")

# Countplot - displays the number of instances of each variable

sns.countplot(data=df, y="DRG_Code", hue="Region")


# Regression plots
# PLotting with regplot()

sns.regplot(data=df, x='temp', y='total_rentals', marker='+')

# Evaluating regression with residplot()
# Useful for evaluating fit of a model

sns.residplot(data=df, x='temp', y='total_rentals')
# Ideally residual markers should be plotted randomly across the horizontal line
# If curved - may suggest that a non-linear model might be appropriate

# Polynomial regression - using order parameters

sns.regplot(data=df, x='temp', y='total_rentals', order=2)  # Attempts polynomial fit using numpy functions

# Residual plot with polynomial regression

sns.regplot(data=df, x='temp', y='total_rentals', order=2)

# Regression plots with categorical variables

sns.regplot(data=df, x='mnth',
            y='total_rentals', x_jitter=.1, order=2)  # x_jitter makes it easier to see values for each month

# In some cases x_estimator can be useful for highlighting trends

sns.regplot(data=df, x='mnth',
            y='total_rentals', x_estimator=np.mean, order=2)  # Uses an estimator for x value (shows mean and CI)

# Binning the data - x_bins can be used to divide the data into discrete bins
# The regression line is still fit against all the data
# Useful for a quick read on continuous data

sns.regplot(data=df, x='temp', y='total_rentals', x_bins=4)

# Matrix plots - heatmap most common type
# heatmap() function requires data to be in a grid format
# pandas crosstab() is frequently used to manipulate data

pd.crosstab(df['mnth'], df['weekday'],
            values=df["total_rentals"],aggfunc='mean').round(0)

# Build a heatmap

sns.heatmap(pd.crosstab(df['mnth'], df['weekday'],
            values=df["total_rentals"],aggfunc='mean'))

# Customise a heatmap

sns.heatmap(df_crosstab, annot=True,  # Turns on annotations in the individual cells
            fmt="d",  # Ensures results are displayed as integers
            cmap="YlGnBu",  # Changes the shading used
            cbar=False,  # Color bar is not displayed
            linewidths=.5)  # Lines between cells

# Centering the heatmap color scheme on a specific value

sns.heatmap(df_crosstab, annot=True,
            fmt="d",
            cmap="YlGnBu",
            cbar=True,
                center=df_crosstab.loc[9.6])  # Centered around saturdays in June

# Plotting a correlation matrix

# Pandas corr function calculates correlations between columns in a dataframe
# The output can be converted to a heatmap with seaborn

sns.heatmap(df.corr())

# Using FacetGrid, factorplot and lmplot
# Can combine multiple smaller plots into a larger visualisation
# Using small multiples is helpful for comparing trends across multiple variables - trellis or lattice plot
# Also frequently called faceting
# Seaborn's grid plots requires "tidy" format - one observation per row of data

# FacetGrid
# Fondational for many data aware grid
# Allows the user to control how data is distributed across columns, rows and hue
# Once a FacetGrid is created, its plot type must be mapped to the grid
# Must use two step process - defining the facets and mapping the plot type

# FacetGrid categorical example
g = sns.FacetGrid(df, col="HIGHDEG")  # Set up FacetGrid with col has highest degree awarded
g.map(sns.boxplot, 'Tuition',  # Plot a boxplot of the tuition values
      order=['1','2','3','4'])  # Specifies the order

# Factorplot simpler way of using FacetGrid for categorical data
# Combines facetting and mapping into 1 function

sns.factorplot(x="Tuition", data=df, col="HIGHDEG", kind="box")

# FacetGrid for scatter or regression plots

g = sns.FacetGrid(df, col="HIGHDEG")
g.map(plt.scatter, 'Tuition', 'SAT_AVG_ALL')

# lmplot plots scatter and regression plots on a FacetGrid

sns.lmplot(data=df, x="Tuition", y="SAT_AVG_ALL",
           col="HIGHDEG", fit_reg=False)  # Disabled regression lines


sns.lmplot(data=df, x="Tuition", y="SAT_AVG_ALL",
           col="HIGHDEG", row="REGION")  # Row used to filter data by region

# Using PairGrid and pairplot
# Also allow us to see interactions across different columns of data
# Only define the columns of data we want to compare

# Pairwise relationships
# PairGrid shows pairwise relationships between data elements
# Diagonals contain histograms
# Contains scatter plot alternating which variable is on the x and y axis
# Similar API to FacetGrid, but do not define the row and column parameters

g = sns.PairGrid(df, vars=["Fair_Mrkt_Rent", "Median_Income"])  # Define variables - dataframe cols we want to look at
g = g.map(plt.scatter)

# Customising the PairGrid diagonals

g = sns.PairGrid(df, vars=["Fair_Mrkt_Rent", "Median_Income"])]
g = g.map_diag(plt.hist)
g = g.map_offdiag(plt.scatter)

# Pairplot is a shortcut for PairGrid

sns.pairplot(df, vars=["Fair_Mrkt_Rent", "Median_Income"],
    kind=reg,  # Plots regression line
    diag_kind='hist')

sns.pairplot(df.query('BEDRMS < 3'), vars=["Fair_Mrkt_Rent", "Median_Income", "UTILITY"]\
             ,hue='BDRMS', palette='husl', plot_kws={'alpha': 0.5})
# 3 variables results in 9 plots

# Using JointGrid and jointplot
# Compares the distribution of data between two variables

# Input - x and y variable
# Centre contains scatter plot of two variables
# Plots along x and y axis show the distribution of data for each variable
# Can configure by specifying the types of joint plots and marginal plots

g = sns.JointGrid(data=df, x="Tuition",
                  y="ADM_RATE_ALL")  # Define grid
g.plot(sns.regplot, sns.distplot)  # Map plots onto grid

# Advanced JointGrid

g = sns.JointGrid(data=df, x="Tuition", y="ADM_RATE_ALL")
g = g.plot_joint(sns.kdeplot)  # Specifies KDE plot in center
g = g.plot_marginals(sns.kdeplot, shade=True)
g = g.annotate(stats.pearsonr)  # Provides additional information about the relationship of the variables (pearson correl value)

# Jointplot - easier to use, less available customisations

sns.jointplot(data=df, x="Tuition", y="ADM_RATE_ALL", kind="hex")  # Specifies hex plot

# Customising a jointplot - shows the pearson r by default for a regplot

g = (sns.jointplot(x="Tuition",
                   y="ADM_RATE_ALL", kind="scatter",
                   xlim=(0,25000),  # Set limits for x axis
                   marginal_kws=dict(bins=15, rug=True),  # Pass keywords to marginal plot to control structure of hist
                   data=df.query('UG < 2500 & Ownership == "Public"'))  # Filters data
     .plot_joint(sns.kdeplot))  # KDE plot is overlaid on scatter plot
# Supports adding overlay plots to enhance the final output

# Selecting seaborn plots

# distplot() is a good place to start for dist analysis
# rugplot() and kdeplot() can be useful alternatives

# For two variables
# lmplot() performs regression analysis and supports facetting
# Good for determining linear relationships between data

# Explore data with the categorical plots and facet with FacetGrid

# Pairplot and Jointplot - more useful after preliminary analysis completed
# Good for regression analysis with lmplot
# Good for analysing distributions with distplot









