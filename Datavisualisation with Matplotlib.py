''' Matplotlib - complete control over properties of your plot
pyplot submodule'''
import matplotlib.pyplot as plt

# Subplots creates figure object and axes object
# Figure - container, holds everything on page
# Axes - holds the data (the canvas)
# Adding data to axes
fig, ax = plt.subplots()
ax.plot(seattle_weather['MONTH'], seattle_weather['MLY-TAVG-NORMAL'])
ax.plot(austin_weather['MONTH'], austin_weather['MLY-TAVG-NORMAL'])
plt.show()
# Can plot multiple plots on one axes

# Customising your plots
# Adding markers
ax.plot(austin_weather['MONTH'], austin_weather['MLY-TAVG-NORMAL'], marker="o")
plt.show()
# Also: "v", check library for more
# Setting linestyle
ax.plot(austin_weather['MONTH'], austin_weather['MLY-TAVG-NORMAL'], marker="o", linestyle="--")
plt.show()
# linestyle="None" - no line
# Chosing colour
ax.plot(austin_weather['MONTH'], austin_weather['MLY-TAVG-NORMAL'], marker="o", linestyle="--", color="r")
plt.show()
# Axes object has several methods that starts with "set" - methods can be used to change properties of object before show
# Customising axis labels
ax.set_xlabel("Time (months)")
ax.set_ylabel("Average temperature (Farenheit degrees)")
ax.set_title("Weather in Seattle")
plt.show()

#Small multiples
#Multiple small plots that show similar data across different conditions
fig, ax=plt.subplots() #Creates one subplot
# Typically arranged as a grid with rows and columns
fig, ax=plt.subplots(3, 2)
# Axes - now an array of axes objects with shape 3x2
ax.shape #3,2
# Now need to call the plot method on an element of the array
# Special case for only one row or column of plots
fig, ax=plt.subplots(2, 1, sharey="True") #sharey ensures y-axis range is fixed the same for both
ax[0].plot(seattle_weather["MONTH"], seattle_weather["MLY-PRCP-NORMAL"]), color="b")
ax[0].plot(seattle_weather["MONTH"], seattle_weather["MLY-PRCP-25PCTL"]), linestyle="--", color="b")
ax[0].plot(seattle_weather["MONTH"], seattle_weather["MLY-PRCP-75PCTL"]), linestyle="--", color="b")
ax[1].plot(austin_weather["MONTH"], austin_weather["MLY-PRCP-NORMAL"]), color="r")
ax[1].plot(austin_weather["MONTH"], austin_weather["MLY-PRCP-25PCTL"]), linestyle="--", color="r")
ax[1].plot(austin_weather["MONTH"], austin_weather["MLY-PRCP-75PCTL"]), linestyle="--", color="r")
ax[0].set_ylabel("Precipitation")
ax[1].set_ylabel("Precipitation")
ax[1].set_xlabel("Time (months)") # Only add x-axis label to bottom plot
plt.show()

# PLotting time series data

import matplotlib.pyplot as plt
fig, ax=plt.subplots()
ax.plot(climate_change.index, climate_change["co2"])
ax.set_xlabel('Time')
ax.set_ylabel("CO2 (ppm)")
plt.show()
# Zooming in on a decade
sixties = climate_change["1960-01-01":"1969-12-31"]
fig, ax=plt.subplots()
ax.plot(sixties.index, sixties["co2"])
ax.set_xlabel('Time')
ax.set_ylabel("CO2 (ppm)")
plt.show()

# Plotting time series with different variables
# Using twin axes
import matplotlib.pyplot as plt
fig, ax=plt.subplots()
ax.plot(climate_change.index, climate_change["CO2"], color="blue")
ax.set_xlabel('Time')
ax.set_ylabel('CO2 (ppm)', color="blue")
ax.tick_params('y',colors='blue') #Sets tick color to blue
ax2 = ax.twinx() #Share the same x axis, but y axis separate
ax2.plot(climate_change.index, climate_change["relative_temp"], color='red')
ax2.set_ylabel('Relative temperature (Celsius)', color='red')
ax2.tick_params('y', color='red')
plt.show()

# A function that plots time series
def plot_timeseries(axes, x, y, color, xlabel, ylabel) :
    axes.plot(x, y, color=color)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel, color=color)
    axes.tick_params('y', colors=color)

# Using our function
fig, ax=plt.subplots()
plot_timeseries(ax, climate_change.index, climate_change['CO2'], 'blue', 'Time', 'CO2 (ppm)')
ax2 = ax.twinx()
plot_timeseries(ax, climate_change.index, climate_change['relative_temp'], 'red', 'Time', 'Relative temperature (Celsius)')
plt.show()

# Adding annotations to time series data
fig, ax=plt.subplots()
plot_timeseries(ax, climate_change.index, climate_change['CO2'], 'blue', 'Time', 'CO2 (ppm)')
ax2 = ax.twinx()
plot_timeseries(ax, climate_change.index, climate_change['relative_temp'], 'red', 'Time', 'Relative temperature (Celsius)')
ax2.annotate(">1 degree",
             xy=(pd.TimeStamp("2015-10-06"),1),
             xytext=(pd.TimeStamp('2008-10-06'),-0.2),
             arrowprops={"arrowstyle":"->","color":"grey"})
#Pandas timestamp obj used to define date
# xytext object allows positioning of text
# arrowprops - connects text to data, empty dictionary - default properties
# can define the properties of arrows - customising annotations, matplot lib documentation

plt.show()

# Quantitative comparisons - bar charts
medals = pd.read_csv('medals_by_country_2016.csv', index_col=0)
fig, ax=plt.subplots()
ax.bar(medals.index, medals["Gold"])
ax.set_xticklabels(medals.index, rotation = 90) # Rotates tick labels
ax.set_ylabel("Number of medals") #Set ylabels
plt.show()

# Creating a stacked bar chart
fig, ax=plt.subplots()
ax.bar(medals.index, medals["Gold"], label="Gold") #Labels allow for legend
ax.bar(medals.index, medals["Silver"], bottom=medals["Gold"], label="Silver")
ax.bar(medals.index, medals["Bronze"], bottom=medals["Gold"]+medals["Silver"], label="Bronze")
ax.set_xticklabels(medals.index, rotation=90)
ax.set_ylabel("Number of medals")
ax.legend()
plt.show()

# Histograms - shows distribution of values within a variable
fig, ax=plt.subplots()
ax.hist(mens_rowing["Height"], label="Rowing", bins=5, histtype="step")
ax.hist(mens_gymnastic["Height"], label="Gymnastics", bins=5, histtype="step")
ax.set_xlabel("Height (cm)")
ax.set_ylabel("# of observations")
ax.legend()
plt.show()

# Bins - default = 10
# Can set a sequence of values - sets boundaries between the bins, e.g
bins=[150, 160, 170, 180, 190, 200, 210]
# Transparency histtype="step"

#Statistical plotting
#Adding error bars to bar charts - additional markers that tell us something sbout the dist of data
fig, ax=plt.subplots()

ax.bar("Rowing",
       mens_rowing["Height"].mean(),
       yerr=mens_rowing["Height"].std())

ax.bar("Gymnastics",
       mens_gymnastics["Height"].mean(),
       yerr=mens_gymnastics["Height"].std())

ax.set_ylabel("Height (cm)")

plt.show()


# Adding error bars to line plots
fig, ax = plt.subplots()
ax.errorbar(seattle_weather['MONTH'] #Sequence of x values
            seattle_weather['MLY-TAVG-NORMAL'], #Sequence of y values
            yerr=seattle_weather['MLY-TAVG-STDEV']) #yerr keyword, takes std devs of avg monthly temps

ax.errorbar(austin_weather['MONTH'],
            austin_weather['MLY-TAVG-NORMAL'],
            yerr=austin_weather['MLY-TAVG-STDEV'])

ax.set_ylabel("Temperature (Farenheit)")
plt.show()

#Adding boxplots
fig, ax=plt.subplots()
ax.boxplot([mens_rowing["Height"],
            mens_gymnastics["Height"]])
ax.set_xticklabels(["Rowing", "Gymnastics"])
ax.set_ylabel("Height (cm)")
plt.show()

# Line = mean
# Box = IQR
# Whiskers = 1.5x IQR
# SHould be around 99% of the distribution, if normal
# Points beyond whiskers - outliers

# Quant comparisons, scatter plots
# Bivariate comparison - scatter is standard visualisation
fig, ax=plt.subplots()
ax.scatter(climate_change["co2"], climate_change["relative_temp"])
ax.set_ylabel("Relative temperature (Celcius)")
ax.set_xlabel("CO2 (ppm)")
plt.show()

#Customising scatter plots
eighties = climate_change["1980-01-01":"1989-12-31"]
nineties = climate_change["1990-01-01":"1999-12-31"]
fig, ax=plt.subplots()
ax.scatter(eighties["co2"], eighties["relative_temp"], color="red", label="eighties")
ax.scatter(nineties["co2"], eighties["relative_temp"], color="blue", label="nineties")
ax.legend()
ax.set_xlabel("CO2")
ax.set_ylabel("Relative temperature (Celsius)")
plt.show()

#Encoding a third variable by color
fig, ax=plt.subplots()
ax.scatter(climate_change["co2"], climate_change["relative_temp"],
        c=climate_change.index) #Variable becomes encoded as color (different to color keyword)
ax.set_xlabel("CO2")
ax.set_ylabel("Relative temperature (Celsius)")
plt.show()

# Preparing your figures to share with others
# Changing plot style
plt.style.use("ggplot") #Emulates the style of the R library ggplot, changes multiple elements
ax.plot(austin_weather['MONTH'], austin_weather['MLY-TAVG-NORMAL'], marker="o", linestyle="--", color="r")
ax.set_xlabel("Time (months)")
ax.set_ylabel("Average temperature (Farenheit degrees)")
ax.set_title("Weather in Seattle")
plt.show()

plt.style.use("default") #Go back to default
# Several different styles available at matplotlib documentation
# e.g. bmh, seaborn-colorblind
# Seaborn software library for statistical visualisation based on matplotlib
# (some styles adopted back by matplotlib)

# Dark backgrounds less visible - discouraged
# Consider colourblind friendly styled, e.g. "tableau-colorblind10"
# If printing required - use less ink, avoid coloured backgrounds
# If printing b&w, use grayscale style

#Sharing visualisations with others
fig, ax=plt.subplots()
ax.bar=(medals.index, medals["Gold"])
ax.set_xticklabels(medals.index, rotation=90)
ax.set_ylabel("Number of medals")
fig.savefig("goldmedals.png") #Call to figure objects to save, provide file name. Won't appear on screen, but as a file
ls #unix function gives list of files in working directory
#png - lossless compression of image - image retains highquality, but takes up relatively large space/bandwidth
#jpg - if part of website, uses lossycompression, takes up less diskspace/bandwidth, can control deg of loss of qual
fig.savefig("godmedals.png", quality=50) #Controls degree of loss of quality. Avoid above 95, comp no longer affected
fig.savefig("goldmedals.svg") #Vector graphics file. Diff elements can be edited in detail by advanced graphics software (good if need to edit later)
fig.savefig("goldmedals.png", dpi=300) #dots per inch - higher number -> more densely rendered image. 300 rel high qual. Larger filesize
fig.set_size_inches([5,3]) #Allows control of size of figure - width x height (determines aspect ratio)

# Automating figures from data
# Allows you to write programmes that automatically adjust what you are doing based on the input data
" Why automate?" \
"- Ease and speed" \
"- Flexibility" \
"- Robustness" \
"- Reproducibility"

# Column - panda series object
sports = summer_2016_medals["Sport"].unique() #Creates a list of distinct sports

fig ax=plt.subplots()

for sport in sports:
    sport_df = summer_2016_medals[summer_2016_medals["Sport"] == sport]
    ax.bar(sport, sport_df["Height"].mean(),
           yerr=sport_df["Height"].std())
ax.set_ylabel("Height (cm)")
ax.set_xticklabels(sports, rotation=90)
plt.show()

#Matplotlib gallery - lots of examples
# Can also plot in 3D, e.g. parametric curve through 3D space
# Visualising data from images, using pseudo colour - each value in image translated into a color
# Animations - uses time to vary display through animation
# Geospatial data - other packages e.g. catopy extends matplotlib using maps. Also seaborn
# Pandas + Matplotlib = Seaborn


