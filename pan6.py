# -*- coding: utf-8 -*-
import pandas as pd
ds=pd.read_csv('Game_medal.csv',encoding="ISO-8859-1")
#encoding="ISO-8859-1"
ds.head()#default first five row
ds.tail()#default last five row
ds.describe()#BEsic Summary dataset
ds.shape#Nuber of row and Colums
ds.shift#Default shift as it is database otherwise leftshif or right shift
ds.info()#Database Information
ds.NOC#Colum name NOC displayed
ds.plot()#default INteger value accept x no of record y year

import matplotlib.pyplot as plt
plt.plot(ds.Edition)#graph for edition
plt.plot(ds.Edition, label="Year of Event")#lebel shown
plt.legend(loc='upper left')#default first five row
plt.plot(ds.Edition, color="blue",label="Year of Event")
plt.plot(ds.Edition,linewidth=2.5, color="red",label="Year of Event")
fig = plt.gcf() # get current figure
ds.plot()
fig.savefig('my_figure.png')



import pandas as pd
ds1=pd.read_csv('movie_metadata.csv',encoding="ISO-8859-1")
ds1.head()
ds1.shape
ds1.info()
ds1.imdb_score.describe()
ds1['movie_title']   # display movie_title column
ds1['duration'][:10]  # First 10 records of duration column
ds1[['budget','gross']] # select multiple columns
ds1[ds1['duration'] > 120] # select movie more than 2 hr duration

### Remove incomplete rows ####
ds1.country=ds1.country.fillna('') # Will put empty string
ds1.duration=ds1.duration.fillna(ds1.duration.mean()) # will put mean value of duration table
ds1.dropna() # Dropping all rows with any NA values 
ds1.dropna(how='all') # drop rows that have all NA values
ds1.dropna(thresh=5) # at least 5 non-null values, and greater than that will be deleted
ds1.dropna(subset=['title_year'])  # don’t want to include any movie that doesn’t have information on when the movie came out

### Deal with error-prone columns ###
### axis=1 for Column and axis=0 for row it is default

ds1.dropna(axis=1, how='all') # drop the columns with that are all NA values
ds1.dropna(axis=1, how='any') # Drop all columns with any NA values
#ds1= pd.read_csv('movie_metadata.csv', dtype={'duration':int})
#ds1=pd.read_csv('movie_metadata.csv', dtype={'title_year':str}) # Normalize data types
## Change casing
ds1['movie_title'].str.upper()
ds1['movie_title'].str.strip() # Remove trailing white spaces

# Rename column

ds1.rename(columns = {'title_yds1.dropna(axis=1, how='any')ear':'release_date', 'movie_facebook_likes':'facebook_likes'})

# Save your result

ds1.to_csv('cleanfile.csv', encoding='utf-8')
