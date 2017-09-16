# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 17:03:05 2017

@author: sahebsingh
"""

""" Import Libraries """

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from bokeh.models import HoverTool
from bokeh.plotting import figure
from bokeh.io import output_notebook,show
from haversine import haversine
from geopy.distance import great_circle
from scipy.spatial.distance import euclidean , cityblock

""" Load Data, remove obvious outliers and convert to sensible data. """

train = pd.read_csv('/Users/sahebsingh/Documents/Projects/cabthing1/train.csv')
test = pd.read_csv('/Users/sahebsingh/Documents/Projects/cabthing1/test.csv')
#print(train.head(2))

def topics(reader):
    topics = []
    for row in reader:
        topics.append(row)
    print(topics)
#topics(train)

# Remove rides to and fro from far areas.

xlim = [-74.03, -73.77]
ylim = [40.63, 40.85]

train = train[(train.pickup_longitude> xlim[0]) & (train.pickup_longitude < xlim[1])]
train = train[(train.dropoff_longitude> xlim[0]) & (train.dropoff_longitude < xlim[1])]
train = train[(train.pickup_latitude> ylim[0]) & (train.pickup_latitude < ylim[1])]
train = train[(train.dropoff_latitude> ylim[0]) & (train.dropoff_latitude < ylim[1])]


# Plot Rides.

longitude = list(train['pickup_longitude']) + list(train['dropoff_longitude'])
latitude = list(train['pickup_latitude']) + list(train['dropoff_latitude'])
plt.figure(figsize = (10,10))
plt.plot(longitude,latitude,'.', alpha = 0.4, markersize = 0.05)
plt.show()


# Cluster

def task2():
    lonlat = pd.DataFrame()
    lonlat['latitude'] = latitude
    lonlat['longitude'] = longitude

    kmeans = KMeans(n_clusters = 15, random_state = 2, n_init = 10).fit(lonlat)
    lonlat['label'] = kmeans.labels_

    lonlat = lonlat.sample(200000)
    plt.figure(figsize = (10, 10))
    for label in lonlat.label.unique():
        plt.plot(lonlat.longitude[lonlat.label == label], lonlat.latitude[lonlat.label == label], '.'
                 ,alpha = 0.3, markersize = 0.3)

    plt.title("Clusters of New York")
    plt.show()


""" Analyse Variables. """

#pd.set_option('display.float_format', lambda x: '%.2f' % x)
#print(train.describe())

plt.scatter(train.trip_duration, train.index, color = "gold")
plt.title('Trip Duration')
plt.show()

# Checking for trip_duration variable.

train['log_trip_duration'] = np.log1p(train['trip_duration'].values)

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,8))
fig.suptitle('Train trip duration and log of trip duration')
ax1.legend(loc=0)
ax1.set_ylabel('count')
ax1.set_xlabel('trip duration')
ax2.set_xlabel('log(trip duration)')
ax2.legend(loc=0)
ax1.hist(train.trip_duration,color='black',bins=7)
ax2.hist(train.log_trip_duration,bins=70,color='gold');
plt.show()

# Famous Vendor.

train["vendor_id"].value_counts().plot(kind='bar', color=["black", "gold"])
plt.title('Vendor')
plt.xlabel('Vendor Ids')
plt.ylabel('Count for each Vendor')
plt.show()

# Passengers Travelling Together.

train['passenger_count'].value_counts().plot(kind='bar', color = ['black', 'gold'])
plt.xlabel("Count of Passengers")
plt.ylabel("Number of Passengers")
plt.show()

# Dates and Timings of the Trips.

train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
test['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
train['dropoff_datetime'] = pd.to_datetime(train['dropoff_datetime'])
test['dropoff_datetime'] = pd.to_datetime(train['dropoff_datetime'])

# Dates

train['pickup_date'] = train['pickup_datetime'].dt.date
test['pickup_date'] = test['pickup_datetime'].dt.date

# Day of month 1 to 30/31

train['pickup_day'] = train['pickup_datetime'].dt.day
test['pickup_day'] = train['pickup_datetime'].dt.day

# Month of year 1 to 12.

train['pickup_month'] = train['pickup_datetime'].dt.month
test['pickup_month'] = test['pickup_datetime'].dt.month

# Weekday 0 to 6.

train['pickup_weekday'] = train['pickup_datetime'].dt.weekday
test['pickup_weekday'] = test['pickup_datetime'].dt.weekday

# Week of Year.

train['pickup_weekofyear'] = train['pickup_datetime'].dt.weekofyear
train['pickup_weekofyear'] = test['pickup_datetime'].dt.weekofyear

# Hour.

train['pickup_hour'] = train['pickup_datetime'].dt.hour
test['pickup_hour'] = test['pickup_datetime'].dt.hour

# Minute of Hour.

train['pickup_minute'] = train['pickup_datetime'].dt.minute
test['pickup_minute'] = test['pickup_datetime'].dt.minute

# Day of Year.

train['pickup_dayofyear'] = train['pickup_datetime'].dt.dayofyear
train['pickup_week_hour'] = train['pickup_weekday'] * 24 + train['pickup_hour']
test['pickup_dayofyear'] = test['pickup_datetime'].dt.dayofyear
test['pickup_week_hour'] = test['pickup_weekday'] * 24 + train['pickup_hour']

# Analysing New Variables.

pd.set_option('display.float_format', lambda x: '%.2f' % x)
#print(train.describe())

# Looking at Duration for Each Month.

plt.figure(figsize=(10,6))
train['pickup_month'].value_counts().plot(kind='bar', color=['black', 'gold'],
     align='center', width=0.3)
plt.title('Trips Each Month')
plt.xlabel("Months")
plt.ylabel("Number of Trips")
plt.show()
""" This shows we have only 6 months of data. """

# Looking at Taxi Pickup by Dates.

tripsByDate = train['pickup_date'].value_counts()
plot = figure( x_axis_type="datetime", tools="",
              toolbar_location=None, x_axis_label='Dates',
            y_axis_label='Taxi trip counts', title='Hover over points to see taxi trips')

x,y= tripsByDate.index, tripsByDate.values
plot.line(x,y, line_dash="4 4", line_width=1, color='gold')

cr = plot.circle(x, y, size=20,
                fill_color="gold", hover_fill_color="black",
                fill_alpha=0.05, hover_alpha=0.5,
                line_color=None, hover_line_color="black")
plot.left[0].formatter.use_scientific = False

plot.add_tools(HoverTool(tooltips=None, renderers=[cr], mode='hline'))
#show(plot)

# Looking at Number of Pickup on each Day.

plt.figure(figsize=(10,6))
train.pickup_day.value_counts().plot(kind='bar', color = ['black', 'gold'],
     align = 'center', width = 0.3)
plt.title('Trips Each Day')
plt.xlabel('Days')
plt.ylabel('Number of Trips')
plt.show()

""" This shows highest number of pickups were on the date 16th. """

# Taxi Pickup on Weekdays.

plt.figure(figsize=(10,6))
train['pickup_weekday'].value_counts().plot(kind='bar', color = ['black', 'gold'],
    align = 'center', width = 0.3)
plt.xlabel("WeekDay")
plt.ylabel('Number of Pickup')
plt.title('Trips each day of the week.')
plt.show()
""" Highest Number of Trips Happen on Thursday and Lowest on Friday. """

# Taxi By Pickup Hour. 

train['pickup_hour'].value_counts().plot(kind='bar', color=['black', 'gold'],
     align = 'center', width = 0.3)
plt.xlabel("Hour")
plt.ylabel('Number of Pickups')
plt.title('Number of Pickup on Each Hour')
plt.show()
""" Highest Number of Pickups are on 6pm and 7pm """

# Longitude and Latitude Coordinates.

fig, ax = plt.subplots(ncols=2, nrows=2,figsize=(10, 12), sharex=False, sharey = False)
ax[0,0].hist(train.pickup_latitude.values,bins=40,color="gold")
ax[0,1].hist(train.pickup_longitude.values,bins=35,color="black")
ax[1,0].hist(train.dropoff_latitude.values,bins=40,color="gold")
ax[1,1].hist(train.dropoff_longitude.values,bins=35,color="black")
ax[0,0].set_xlabel('Pickup Latitude')
ax[0,1].set_xlabel('Pickup Longitude')
ax[1,0].set_xlabel('Dropoff Latitude')
ax[1,1].set_xlabel('Dropoff Longitude')
plt.show()


""" Calculating Distance and Speed of Rides """

train['lat_diff'] = train['pickup_latitude'] - train['dropoff_latitude']
train['lon_diff'] = train['pickup_longitude'] - train['dropoff_longitude']
test['lat_diff'] = test['pickup_latitude'] - test['dropoff_latitude']
test['lon_diff'] = test['pickup_longitude'] - test['dropoff_longitude']

# Calcualte Haverine Distance

train['haversine_distance'] = train.apply(lambda row: haversine( (row['pickup_latitude'], row['pickup_longitude']),
     (row['dropoff_latitude'], row['dropoff_longitude']) ), axis=1)
train['log_haversine_distance'] = np.log1p(train['haversine_distance'])

plt.scatter(train['log_haversine_distance'], train['log_trip_duration'],color=['gold', 'black'],s=5)
plt.title("log(Haversine Distance) Vs log(Trip Duration)")
plt.show()

# Calculate Great_circle Distance.

train['great_circle_distance'] = train.apply(lambda row: great_circle((row['pickup_latitude'],
                        row['pickup_longitude']),(row['dropoff_latitude'], row['dropoff_longitude'])).miles, axis = 1)
train['log_great_circle_distance'] = np.log1p(train['great_circle_distance'])
plt.scatter(train['great_circle_distance'], train['log_great_circle_distance'], color=['gold', 'black'], s=5)
plt.title("Great Circle Distance VS Log(Great Circle Distance)")
plt.show()

# Calculate Euclidean Distance
 
train['euclidean_distance'] = train.apply(lambda row: euclidean((row['pickup_latitude'], row['pickup_longitude']),
     (row['dropoff_latitude'], row['dropoff_longitude'])), axis = 1)
train['log1p_euclidean_distance'] = np.log1p(train['euclidean_distance'])
plt.scatter(train['euclidean_distance'], train['log1p_euclidean_distance'], color=['gold', 'black'], s=5)
plt.title('log1p(Euclidean Distance) vs Euclidean Distance')
plt.show()


""" Calculate Average Speed of The Trip. """

train['average_speed_h'] = 1000 * train['haversine_distance'] / train['trip_duration']
train['average_speed_e'] = 1000 * train['euclidean_distance'] / train['trip_duration']
train['average_speed_c'] = 1000 * train['great_circle_distance'] / train['trip_duration']

# Plotting Cityblock Distance.

train['cityblock_distance'] = train.apply(lambda row: cityblock((row['pickup_latitude'], row['pickup_longitude']),
     (row['dropoff_latitude'], row['dropoff_longitude'])), axis = 1)
train['log1p_cityblock_distance'] = np.log1p(train['cityblock_distance'])
plt.scatter(train['cityblock_distance'], train['log1p_cityblock_distance'], color = ['gold', 'black'], s=5)
plt.title('Cityblock Distance vs log1p (Cityblock Distance)')
plt.show()

# Plotting Average Distance.

fig, ax = plt.subplots(ncols = 2, sharey = True)
ax[0].plot(train.groupby('pickup_hour').mean()['average_speed_e'], '^', lw=2, alpha=0.7,color='yellow')
ax[1].plot(train.groupby('pickup_weekday').mean()['average_speed_h'], 's', lw=2, alpha=0.7, color='black')
ax[0].set_xlabel('hour')
ax[1].set_xlabel('weekday')
fig.suptitle('Rush Hour Average Speed')
plt.show()

#print(train.info())
#print(train.dtypes)


""" Model And Prediction """

target = train['log_trip_duration'].values

train = train.drop(['id', 'pickup_datetime', 'haversine_distance', 'cityblock_distance',
                    'euclidean_distance','average_speed_c','average_speed_e',
                    'average_speed_h','dropoff_datetime', 'trip_duration',
                    'log_trip_duration','pickup_date', 'store_and_fwd_flag'], axis=1)


print(train.dtypes)

Id = test.id.values
test = test.drop(['id', 'store_and_fwd_flag', 'pickup_datetime'], axis = 1)

# Random Forest.

rf_model = RandomForestRegressor(n_estimators = 25, min_samples_leaf = 25, min_samples_split = 25)

# Fitting The Model and Predicting The Data.

rf_model.fit(train.values, target)
predictions=rf_model.predict(test.values)
print(predictions[:5])

# Submission

test['trip_duration'] = np.exp(predictions) - 1
test['id']=Id
test[['id', 'trip_duration']].to_csv('poonam.csv.gz', index=False, compression='gzip')
print(test['trip_duration'][:5])
print(rf_model)























