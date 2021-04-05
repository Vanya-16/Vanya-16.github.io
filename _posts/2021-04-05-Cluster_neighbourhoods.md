---
title: "Determine neighbourhood to open new restaurant using clustering"
date: 2021-04-05
tags: [data science, clustering, machine learning algorithms, unsupervised learning, kmeans, web scraping, BeautifulSoup]
excerpt: "Data Science, machine learning algorithms, unsupervised, k means, web scraping"
mathjax: "true"
---


### Objective: Determine the neighbourhood to open a new restaurant in order to expand business.

_We want to open a new restaurant in New York similar to one we have in San Francisco. Firstly, we need to shortlist a few places where we can open up our new restaurant. We'll perform K-Means Clustering in order to determine the place closest to our current location in terms of nearby venues. We'll be using FourSquare API data and some web scraping to get details on the list of neighbourhoods in New York City._

#### The idea behind this analysis can be extended to many other items like opening a new office, play center, buying a house etc.


```python
#importing libraries to be used
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#view plots in jupyter notebook
%matplotlib inline
sns.set_style('whitegrid') #setting style for plots, optional

#Libraries for Gepgraphical identification of location
from geopy.geocoders import Nominatim

# Library for using Kmeans method for clustering
from sklearn.cluster import KMeans

# Libraries to handle requests
import requests
from pandas.io.json import json_normalize

# Libraries to plot and visualize locations on maps and also plotting other kmeans related data
import matplotlib.cm as cm
import matplotlib.colors as colors
import folium

# Liraries to import data from website - Web Scraping
import seaborn as sns
from bs4 import BeautifulSoup as BS
```

### Putting in details of current restaurant location in San Francisco


```python
SF_restaurant = "Octavia St, San Francisco, CA 94102, United States"

# Getting the Lat-Lon of the office
geolocator = Nominatim(user_agent="USA_explorer")
SF_res_location = geolocator.geocode(SF_restaurant)
SF_latitude = SF_res_location.latitude
SF_longitude = SF_res_location.longitude
print('The geograpical coordinate are {}, {}.'.format(SF_latitude, SF_longitude))
```

    The geograpical coordinate are 37.7780777, -122.424924.


### Populating New York City Neighbourhood information from: https://www.baruch.cuny.edu/nycdata/population-geography/neighborhoods.htm


```python
URL = "https://www.baruch.cuny.edu/nycdata/population-geography/neighborhoods.htm"
r = requests.get(URL, verify = False)

soup = BS(r.text, "html.parser")
data = soup.find_all("tr")

start_index = 0
for i in range (len(data)):
    td = data[i].find_all("td")
    for j in range (len(td)):
        if td[j].text == "Brooklyn":
            start_index = i
            break
    if start_index != 0:
        break

end_index = 0
for i in range (len(data)-1,0,-1):
    td = data[i].find_all("td")
    for j in range (len(td)):
        if td[j].text == "Woodside":
            end_index = i
            break
    if end_index != 0:
        break

list1 = []
list2 = []
list3 = []
list4 = []
list5 = []
for i in range (start_index,end_index+1):
    td = data[i].find_all("td")
    list1.append(td[1].text)
    list2.append(td[2].text)
    list3.append(td[3].text)
    list4.append(td[4].text)
    list5.append(td[5].text)

final = []
final.append(list1)
final.append(list2)
final.append(list3)
final.append(list4)
final.append(list5)

df = pd.DataFrame(final)

df = df.transpose()


final_df = pd.DataFrame(columns=['Borough','Neighbourhood'])

for i in range (5):
    d = {}
    d = {'Borough':df[i][0]}
    for j in range (1,len(df)):
        if df[i][j]=='\xa0':
            break
        else:
            d['Neighbourhood'] = df[i][j]
            final_df = final_df.append(d,ignore_index=True)
final_df
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>Neighbourhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Brooklyn</td>
      <td>Bath Beach</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Brooklyn</td>
      <td>Bay Ridge</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brooklyn</td>
      <td>Bedford Stuyvesant</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brooklyn</td>
      <td>Bensonhurst</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brooklyn</td>
      <td>Bergen Beach</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>324</th>
      <td>Staten Island</td>
      <td>Ward Hill</td>
    </tr>
    <tr>
      <th>325</th>
      <td>Staten Island</td>
      <td>West Brighton</td>
    </tr>
    <tr>
      <th>326</th>
      <td>Staten Island</td>
      <td>Westerleigh</td>
    </tr>
    <tr>
      <th>327</th>
      <td>Staten Island</td>
      <td>Willowbrook</td>
    </tr>
    <tr>
      <th>328</th>
      <td>Staten Island</td>
      <td>Woodrow</td>
    </tr>
  </tbody>
</table>
<p>329 rows × 2 columns</p>
</div>



#### Adding Lattitude and Longitude information for each Neighbourhoods


```python
final_df['Latitude']=""
final_df['Longitude']=""

for i in range(len(final_df)):
    nyadd=str(final_df['Neighbourhood'][i])+', '+str(final_df['Borough'][i])+', New York'

    geolocator = Nominatim(user_agent="USA_explorer")
    location = geolocator.geocode(nyadd)
    try:
        latitude = location.latitude
        longitude = location.longitude
    except:
        latitude=1000 # For those neighbourhoods whose latitude and longitude could not be fetched
        longitude=1000
    final_df['Latitude'][i]=latitude
    final_df['Longitude'][i]=longitude

final_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>Neighbourhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Brooklyn</td>
      <td>Bath Beach</td>
      <td>40.6018</td>
      <td>-74.0005</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Brooklyn</td>
      <td>Bay Ridge</td>
      <td>40.634</td>
      <td>-74.0146</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brooklyn</td>
      <td>Bedford Stuyvesant</td>
      <td>40.6834</td>
      <td>-73.9412</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brooklyn</td>
      <td>Bensonhurst</td>
      <td>40.605</td>
      <td>-73.9934</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brooklyn</td>
      <td>Bergen Beach</td>
      <td>40.6204</td>
      <td>-73.9068</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>324</th>
      <td>Staten Island</td>
      <td>Ward Hill</td>
      <td>40.6329</td>
      <td>-74.0829</td>
    </tr>
    <tr>
      <th>325</th>
      <td>Staten Island</td>
      <td>West Brighton</td>
      <td>1000</td>
      <td>1000</td>
    </tr>
    <tr>
      <th>326</th>
      <td>Staten Island</td>
      <td>Westerleigh</td>
      <td>40.6212</td>
      <td>-74.1318</td>
    </tr>
    <tr>
      <th>327</th>
      <td>Staten Island</td>
      <td>Willowbrook</td>
      <td>40.6032</td>
      <td>-74.1385</td>
    </tr>
    <tr>
      <th>328</th>
      <td>Staten Island</td>
      <td>Woodrow</td>
      <td>40.5434</td>
      <td>-74.1976</td>
    </tr>
  </tbody>
</table>
<p>329 rows × 4 columns</p>
</div>



#### Cleaning the dataset fetched from the external URL


```python
final_df=final_df[final_df.Latitude!=1000]
final_df.reset_index(inplace=True)
final_df.drop('index',axis=1,inplace=True)
final_df
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>Neighbourhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Brooklyn</td>
      <td>Bath Beach</td>
      <td>40.6018</td>
      <td>-74.0005</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Brooklyn</td>
      <td>Bay Ridge</td>
      <td>40.634</td>
      <td>-74.0146</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brooklyn</td>
      <td>Bedford Stuyvesant</td>
      <td>40.6834</td>
      <td>-73.9412</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brooklyn</td>
      <td>Bensonhurst</td>
      <td>40.605</td>
      <td>-73.9934</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brooklyn</td>
      <td>Bergen Beach</td>
      <td>40.6204</td>
      <td>-73.9068</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>311</th>
      <td>Staten Island</td>
      <td>Travis</td>
      <td>40.5932</td>
      <td>-74.1879</td>
    </tr>
    <tr>
      <th>312</th>
      <td>Staten Island</td>
      <td>Ward Hill</td>
      <td>40.6329</td>
      <td>-74.0829</td>
    </tr>
    <tr>
      <th>313</th>
      <td>Staten Island</td>
      <td>Westerleigh</td>
      <td>40.6212</td>
      <td>-74.1318</td>
    </tr>
    <tr>
      <th>314</th>
      <td>Staten Island</td>
      <td>Willowbrook</td>
      <td>40.6032</td>
      <td>-74.1385</td>
    </tr>
    <tr>
      <th>315</th>
      <td>Staten Island</td>
      <td>Woodrow</td>
      <td>40.5434</td>
      <td>-74.1976</td>
    </tr>
  </tbody>
</table>
<p>316 rows × 4 columns</p>
</div>



#### Adding in the location of the current restaurant (in San Francisco) so that it is also used in clustering along with NYC neighbourhoods


```python
SF_rest_add={'Borough': 'Hayes Valley, SF','Neighbourhood':'Hayes Valley','Latitude':SF_latitude,'Longitude':SF_longitude}
final_df=final_df.append(SF_rest_add,ignore_index=True)
final_df.iloc[[-1]]
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>Neighbourhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>316</th>
      <td>Hayes Valley, SF</td>
      <td>Hayes Valley</td>
      <td>37.7781</td>
      <td>-122.425</td>
    </tr>
  </tbody>
</table>
</div>



### Clustering the neighbourhoods of New York including the neighbourhood of San Francisco

#### Defining FourSquare credentials


```python
CLIENT_ID = '*****************************************'
CLIENT_SECRET = '**************************************'
VERSION = '20180605'
LIMIT = 100
```

#### Defining a function to get the venues from all neighbourhoods


```python
def getNearbyVenues(borough, names, latitudes, longitudes, radius=500):

    venues_list=[]
    for borough, name, lat, lng in zip(borough, names, latitudes, longitudes):

        # API request URL creation
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID,
            CLIENT_SECRET,
            VERSION,
            lat,
            lng,
            radius,
            LIMIT)

        # making requests for the URL
        results = requests.get(url).json()["response"]['groups'][0]['items']

        # Returning only relevant information for each nearby venue
        venues_list.append([(
            borough,
            name,
            lat,
            lng,
            v['venue']['name'],
            v['venue']['location']['lat'],
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Borough','Neighborhood',
                  'Neighborhood Latitude',
                  'Neighborhood Longitude',
                  'Venue',
                  'Venue Latitude',
                  'Venue Longitude',
                  'Venue Category']

    return(nearby_venues)
```


```python
# Getting the venues for each neighbourhoods
NewYork_venues = getNearbyVenues(borough=final_df['Borough'],names=final_df['Neighbourhood'],
                                   latitudes=final_df['Latitude'],
                                   longitudes=final_df['Longitude']
                                  )
```


```python
# Looking at the data received from FourSquare
NewYork_venues.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10720 entries, 0 to 10719
    Data columns (total 8 columns):
     #   Column                  Non-Null Count  Dtype  
    ---  ------                  --------------  -----  
     0   Borough                 10720 non-null  object
     1   Neighborhood            10720 non-null  object
     2   Neighborhood Latitude   10720 non-null  float64
     3   Neighborhood Longitude  10720 non-null  float64
     4   Venue                   10720 non-null  object
     5   Venue Latitude          10720 non-null  float64
     6   Venue Longitude         10720 non-null  float64
     7   Venue Category          10720 non-null  object
    dtypes: float64(4), object(4)
    memory usage: 670.1+ KB



```python
NewYork_venues.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Neighborhood Latitude</th>
      <th>Neighborhood Longitude</th>
      <th>Venue</th>
      <th>Venue Latitude</th>
      <th>Venue Longitude</th>
      <th>Venue Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Brooklyn</td>
      <td>Bath Beach</td>
      <td>40.60185</td>
      <td>-74.000501</td>
      <td>Lenny's Pizza</td>
      <td>40.604908</td>
      <td>-73.998713</td>
      <td>Pizza Place</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Brooklyn</td>
      <td>Bath Beach</td>
      <td>40.60185</td>
      <td>-74.000501</td>
      <td>King's Kitchen</td>
      <td>40.603844</td>
      <td>-73.996960</td>
      <td>Cantonese Restaurant</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brooklyn</td>
      <td>Bath Beach</td>
      <td>40.60185</td>
      <td>-74.000501</td>
      <td>Delacqua</td>
      <td>40.604216</td>
      <td>-73.997452</td>
      <td>Spa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brooklyn</td>
      <td>Bath Beach</td>
      <td>40.60185</td>
      <td>-74.000501</td>
      <td>Lutzina Bar&amp;Lounge</td>
      <td>40.600807</td>
      <td>-74.000578</td>
      <td>Hookah Bar</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brooklyn</td>
      <td>Bath Beach</td>
      <td>40.60185</td>
      <td>-74.000501</td>
      <td>Planet Fitness</td>
      <td>40.604567</td>
      <td>-73.997861</td>
      <td>Gym / Fitness Center</td>
    </tr>
  </tbody>
</table>
</div>




```python
NewYork_venues["Venue Category"].unique()
```


    array(['Pizza Place', 'Cantonese Restaurant', 'Spa', 'Hookah Bar',
           'Gym / Fitness Center', 'Dessert Shop', 'Chinese Restaurant',
           'Bakery', 'Italian Restaurant', 'Coffee Shop', 'Restaurant',
           'Japanese Restaurant', 'Supplement Shop',
           'Eastern European Restaurant', 'Dim Sum Restaurant', 'Tea Room',
           'Ice Cream Shop', 'Peruvian Restaurant', 'Sandwich Place', 'Bank',
           'American Restaurant', 'Shanghai Restaurant', 'Mobile Phone Shop',
           'Kids Store', 'Gas Station', 'Middle Eastern Restaurant',
           'Seafood Restaurant', 'Tennis Court', 'Vietnamese Restaurant',
           'Noodle House', 'Rental Car Location', 'Park',
           'Fried Chicken Joint', 'Hotpot Restaurant', 'Gift Shop',
           'Irish Pub', 'Malay Restaurant', 'Bar', 'Playground', 'Donut Shop',
           'Bubble Tea Shop', 'Nightclub', 'Cocktail Bar',
           'New American Restaurant', 'Wine Shop', 'Boutique', 'Tiki Bar',
           'Café', 'Taco Place', 'Mexican Restaurant', 'Gym', 'Wine Bar',
           'Gourmet Shop', 'Bagel Shop', 'Lounge', 'Food', 'Deli / Bodega',
           'Thrift / Vintage Store', 'Garden',
           'Southern / Soul Food Restaurant', 'Caribbean Restaurant',
           'Discount Store', 'Pharmacy', 'Farmers Market', 'Cosmetics Shop',
           'Turkish Restaurant', 'Sushi Restaurant', 'Fast Food Restaurant',
           'Salon / Barbershop', 'Video Game Store', 'Frozen Yogurt Shop',
           'Asian Restaurant', 'Shoe Store', 'Clothing Store',
           "Women's Store", 'Optical Shop', 'Accessories Store',
           'Supermarket', 'Bus Station', 'Furniture / Home Store',
           'Concert Hall', 'Antique Shop', 'Yoga Studio',
           'Martial Arts School', 'Grocery Store', 'Indian Restaurant',
           'Athletics & Sports', 'Burrito Place', 'Music Venue',
           'Jewelry Store', 'French Restaurant', 'Thai Restaurant',
           'Flower Shop', 'Dance Studio', 'Bookstore', "Men's Store",
           'Theater', 'Korean Restaurant', 'Music Store',
           'Health & Beauty Service', 'Cajun / Creole Restaurant',
           'Garden Center', 'Arts & Crafts Store', 'Boxing Gym',
           'Electronics Store', 'Dry Cleaner', 'Gastropub', 'Historic Site',
           'Juice Bar', 'Burger Joint', 'Convenience Store',
           'Bed & Breakfast', 'Hotel', 'Bistro', 'Bike Shop', 'Neighborhood',
           'Russian Restaurant', 'Mediterranean Restaurant',
           'Food & Drink Shop', 'Other Great Outdoors', 'Food Truck',
           'Non-Profit', 'Diner', 'Varenyky restaurant', 'Karaoke Bar',
           'Pool', 'Bus Line', 'Tunnel', 'Recording Studio', 'History Museum',
           'Pet Store', 'Scenic Lookout', 'Beach', 'Falafel Restaurant',
           'Pier', 'Indie Theater', 'Pilates Studio', 'Ramen Restaurant',
           'Pub', 'Plaza', 'Chocolate Shop', 'Mattress Store',
           'Spanish Restaurant', 'Moving Target', 'Vape Store',
           'Fruit & Vegetable Store', 'Liquor Store', 'Lawyer',
           'Metro Station', 'Bus Stop', 'Greek Restaurant', 'Record Shop',
           'Beer Garden', 'Butcher', 'Event Space', 'Gaming Cafe',
           'Herbs & Spices Store', 'Church', 'Filipino Restaurant',
           'Latin American Restaurant', 'Wings Joint', 'Breakfast Spot',
           'Brewery', 'Fish Market', 'Art Gallery', 'African Restaurant',
           'Photography Studio', 'Sculpture Garden',
           'Vegetarian / Vegan Restaurant', 'Pie Shop', 'Market',
           'Waterfront', 'Ethiopian Restaurant', 'Yemeni Restaurant',
           'Dumpling Restaurant', 'Indie Movie Theater',
           'Sporting Goods Shop', 'Toy / Game Store', 'Speakeasy', 'Dive Bar',
           'Harbor / Marina', 'Basketball Court', 'Candy Store', 'Museum',
           'Drugstore', 'Department Store', 'Gun Range', 'Tibetan Restaurant',
           'Health Food Store', 'Nail Salon', 'Tapas Restaurant',
           'Salad Place', 'Climbing Gym', 'Theme Park Ride / Attraction',
           'Roof Deck', 'Dog Run', 'Food Court', 'Trail', 'Boat or Ferry',
           'Performing Arts Venue', 'Entertainment Service', 'Hotel Bar',
           'Intersection', 'Poke Place', 'Moroccan Restaurant',
           'Massage Studio', 'Flea Market', 'Cycle Studio', 'Perfume Shop',
           'Residential Building (Apartment / Condo)', 'Whisky Bar', 'Winery',
           'Factory', 'Miscellaneous Shop', 'Hobby Shop', 'High School',
           'Rental Service', 'BBQ Joint', 'Israeli Restaurant', 'Opera House',
           'German Restaurant', 'Beer Bar', 'Cupcake Shop', 'Steakhouse',
           'Shipping Store', 'Board Shop', 'Bridge', 'Shopping Mall',
           'Child Care Service', 'Skate Park', 'Soccer Field',
           'Cuban Restaurant', 'Indoor Play Area', 'Baseball Field',
           "Doctor's Office", 'Jewish Restaurant', 'Polish Restaurant',
           'Sports Bar', 'Track', 'Cheese Shop', 'Bowling Alley',
           'Laundromat', 'Austrian Restaurant', 'Organic Grocery', 'Farm',
           'Gymnastics Gym', 'Halal Restaurant', 'Big Box Store',
           'Kosher Restaurant', 'Tourist Information Center', 'Film Studio',
           'IT Services', 'School', 'Comic Shop', 'Gym Pool',
           'Colombian Restaurant', 'Soup Place', 'Used Bookstore',
           'Business Service', 'North Indian Restaurant', 'Other Nightlife',
           'Public Art', 'Field', 'Picnic Shelter', 'Waterfall',
           'Amphitheater', 'Bike Trail', 'Hill', 'Snack Place', 'Sports Club',
           'Video Store', 'Paper / Office Supplies Store', 'Lake',
           'General Travel', 'Comfort Food Restaurant', 'Creperie',
           'Szechuan Restaurant', 'Stadium', 'Community Center',
           'Arepa Restaurant', 'Brazilian Restaurant', 'Football Stadium',
           'Laundry Service', 'Theme Park', 'Aquarium', 'Exhibit', 'Arcade',
           'Movie Theater', 'Beer Store', 'Udon Restaurant',
           'South American Restaurant', 'Hardware Store', 'Gay Bar',
           'Outdoor Gym', 'Picnic Area', 'Storage Facility', 'Tattoo Parlor',
           'Smoke Shop', 'Piano Bar', 'Train Station', 'Print Shop',
           'Pool Hall', 'Zoo', 'Zoo Exhibit', 'Souvenir Shop',
           'Warehouse Store', 'Check Cashing Service', 'Post Office',
           'Jazz Club', 'Puerto Rican Restaurant', 'Eye Doctor', 'River',
           'Outlet Store', 'Waste Facility', 'Tennis Stadium', 'Canal',
           'Recreation Center', 'Social Club', 'Library', 'Shop & Service',
           'Distillery', 'Home Service', 'Auto Dealership',
           'Construction & Landscaping', 'Outdoors & Recreation', 'Building',
           'Cooking School', 'Memorial Site', 'Auditorium', 'Tree',
           'Lingerie Store', 'Monument / Landmark', 'Paella Restaurant',
           'Japanese Curry Restaurant', 'Lebanese Restaurant',
           'Peking Duck Restaurant', 'Art Museum', 'Smoothie Shop',
           'Argentinian Restaurant', 'Comedy Club', 'Cha Chaan Teng',
           'Taiwanese Restaurant', 'Sake Bar', 'Food Stand', 'Animal Shelter',
           'Molecular Gastronomy Restaurant', 'Medical Center',
           'Golf Driving Range', 'Outdoor Sculpture', 'Ukrainian Restaurant',
           'Soba Restaurant', 'Shabu-Shabu Restaurant',
           'Australian Restaurant', 'Coworking Space', 'Kebab Restaurant',
           'General Entertainment', 'Office', 'Tex-Mex Restaurant',
           'Fountain', 'Stationery Store', 'Adult Boutique',
           'Leather Goods Store', 'Golf Course', 'Fondue Restaurant',
           'Theme Restaurant', 'Veterinarian', 'Empanada Restaurant',
           'College Academic Building', 'Czech Restaurant', 'Club House',
           'Bridal Shop', 'Shoe Repair', 'College Arts Building', 'Circus',
           'College Bookstore', 'Kitchen Supply Store', 'Newsstand',
           'Pet Service', 'Hostel', 'Hawaiian Restaurant', 'College Theater',
           'Churrascaria', 'Skating Rink', 'Luggage Store',
           'College Cafeteria', 'Cultural Center', 'Resort', 'Watch Shop',
           'Outdoor Supply Store', 'Street Art', 'Duty-free Shop',
           'Scandinavian Restaurant', 'Pet Café', 'Swiss Restaurant',
           'Tram Station', 'Persian Restaurant', 'Bike Rental / Bike Share',
           'Tailor Shop', 'Pedestrian Plaza', 'Hot Dog Joint', 'Daycare',
           'Tanning Salon', 'Train', 'Surf Spot', 'Parking',
           'Indonesian Restaurant', 'Venezuelan Restaurant',
           'Imported Food Shop', 'Bath House', 'Fish & Chips Shop',
           'Afghan Restaurant', 'Automotive Shop', 'Beach Bar', 'Pop-Up Shop',
           'Sri Lankan Restaurant', 'Portuguese Restaurant', 'Rest Area',
           'Rock Club', 'Costume Shop', 'Government Building',
           'Airport Lounge', 'Airport Terminal', 'Airport Food Court',
           'Plane', 'Motorcycle Shop', 'Rock Climbing Spot', 'Cafeteria',
           'Auto Garage', 'Romanian Restaurant', 'Go Kart Track',
           'Professional & Other Places', 'Racetrack', 'Fishing Spot',
           'Lighthouse', 'Nightlife Spot', 'Weight Loss Center', 'Buffet',
           'Toll Plaza', 'Botanical Garden', 'Baseball Stadium',
           'Outlet Mall', 'Souvlaki Shop', 'Camera Store'], dtype=object)




```python
NewYork_venues["Venue Category"].nunique()
```




    443



_We are getting 443 unique venues from the FourSquaredata_


```python
NewYork_venues=NewYork_venues[NewYork_venues['Venue Category']!='Neighborhood'] # Code adjusted to remove Neighborhood
```


```python
# One - hot encoding to handle categorical data for clustering
NY_onehot = pd.get_dummies(data=NewYork_venues[['Borough','Neighborhood','Venue Category']],columns=['Venue Category'],drop_first=True,prefix="", prefix_sep="")
NY_onehot.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Adult Boutique</th>
      <th>Afghan Restaurant</th>
      <th>African Restaurant</th>
      <th>Airport Food Court</th>
      <th>Airport Lounge</th>
      <th>Airport Terminal</th>
      <th>American Restaurant</th>
      <th>Amphitheater</th>
      <th>...</th>
      <th>Whisky Bar</th>
      <th>Wine Bar</th>
      <th>Wine Shop</th>
      <th>Winery</th>
      <th>Wings Joint</th>
      <th>Women's Store</th>
      <th>Yemeni Restaurant</th>
      <th>Yoga Studio</th>
      <th>Zoo</th>
      <th>Zoo Exhibit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Brooklyn</td>
      <td>Bath Beach</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Brooklyn</td>
      <td>Bath Beach</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brooklyn</td>
      <td>Bath Beach</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brooklyn</td>
      <td>Bath Beach</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brooklyn</td>
      <td>Bath Beach</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 443 columns</p>
</div>



#### Getting scored of each category based on Mean of the frequency of their occurences. This will help to determine the similarities between neighbourhoods


```python
NY_grouped = NY_onehot.groupby(['Borough','Neighborhood']).mean().reset_index()
NY_grouped.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Adult Boutique</th>
      <th>Afghan Restaurant</th>
      <th>African Restaurant</th>
      <th>Airport Food Court</th>
      <th>Airport Lounge</th>
      <th>Airport Terminal</th>
      <th>American Restaurant</th>
      <th>Amphitheater</th>
      <th>...</th>
      <th>Whisky Bar</th>
      <th>Wine Bar</th>
      <th>Wine Shop</th>
      <th>Winery</th>
      <th>Wings Joint</th>
      <th>Women's Store</th>
      <th>Yemeni Restaurant</th>
      <th>Yoga Studio</th>
      <th>Zoo</th>
      <th>Zoo Exhibit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bronx</td>
      <td>Allerton</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bronx</td>
      <td>Bathgate</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.010000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bronx</td>
      <td>Baychester</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bronx</td>
      <td>Bedford Park</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bronx</td>
      <td>Belmont</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.017241</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 443 columns</p>
</div>



#### Function to determine most common venues, we are cleaning and reducing the venues we are doing our analysis on to reduce noise from the data and to get more precise results


```python
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[2:]
    row_categories_sorted = row_categories.sort_values(ascending=False)

    return row_categories_sorted.index.values[0:num_top_venues]
```


```python
num_top_venues = 15 # selecting top 15 venues for our analysis

indicators = ['st', 'nd', 'rd']

# creating columns according to number of top venues
columns = ['Borough','Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# new dataframe to hold the top 10 venues for each of the neighbourhoods
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Borough']=NY_grouped['Borough']
neighborhoods_venues_sorted['Neighborhood'] = NY_grouped['Neighborhood']

# calling the function to get the top 10 venues for each neighbourhood
for ind in np.arange(NY_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 2:] = return_most_common_venues(NY_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
      <th>11th Most Common Venue</th>
      <th>12th Most Common Venue</th>
      <th>13th Most Common Venue</th>
      <th>14th Most Common Venue</th>
      <th>15th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bronx</td>
      <td>Allerton</td>
      <td>Discount Store</td>
      <td>Sandwich Place</td>
      <td>Fast Food Restaurant</td>
      <td>Pizza Place</td>
      <td>Pharmacy</td>
      <td>Donut Shop</td>
      <td>Storage Facility</td>
      <td>Bike Trail</td>
      <td>Soccer Field</td>
      <td>Seafood Restaurant</td>
      <td>Bar</td>
      <td>Bank</td>
      <td>Clothing Store</td>
      <td>Mobile Phone Shop</td>
      <td>Trail</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bronx</td>
      <td>Bathgate</td>
      <td>Italian Restaurant</td>
      <td>Pizza Place</td>
      <td>Deli / Bodega</td>
      <td>Spanish Restaurant</td>
      <td>Liquor Store</td>
      <td>Bank</td>
      <td>Bakery</td>
      <td>Grocery Store</td>
      <td>Dessert Shop</td>
      <td>Mexican Restaurant</td>
      <td>Sandwich Place</td>
      <td>Food &amp; Drink Shop</td>
      <td>Coffee Shop</td>
      <td>Shoe Store</td>
      <td>Donut Shop</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bronx</td>
      <td>Baychester</td>
      <td>Pharmacy</td>
      <td>Italian Restaurant</td>
      <td>Grocery Store</td>
      <td>Bike Trail</td>
      <td>Liquor Store</td>
      <td>Historic Site</td>
      <td>Pizza Place</td>
      <td>Sandwich Place</td>
      <td>Donut Shop</td>
      <td>Print Shop</td>
      <td>Mobile Phone Shop</td>
      <td>Bus Station</td>
      <td>Bus Line</td>
      <td>Deli / Bodega</td>
      <td>Playground</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bronx</td>
      <td>Bedford Park</td>
      <td>Diner</td>
      <td>Pizza Place</td>
      <td>Deli / Bodega</td>
      <td>Mexican Restaurant</td>
      <td>Supermarket</td>
      <td>Pharmacy</td>
      <td>Chinese Restaurant</td>
      <td>Sandwich Place</td>
      <td>Spanish Restaurant</td>
      <td>Bus Station</td>
      <td>Grocery Store</td>
      <td>Food Truck</td>
      <td>Smoke Shop</td>
      <td>Baseball Field</td>
      <td>Bakery</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bronx</td>
      <td>Belmont</td>
      <td>Italian Restaurant</td>
      <td>Pizza Place</td>
      <td>Bakery</td>
      <td>Deli / Bodega</td>
      <td>Dessert Shop</td>
      <td>Restaurant</td>
      <td>Fish Market</td>
      <td>Food &amp; Drink Shop</td>
      <td>Cheese Shop</td>
      <td>Chinese Restaurant</td>
      <td>Tattoo Parlor</td>
      <td>Grocery Store</td>
      <td>Mexican Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>Mediterranean Restaurant</td>
    </tr>
  </tbody>
</table>
</div>



#### Determining the K value (using elbow method)


```python
K=range(1,25)

NY_grouped_clustering = NY_grouped.drop(['Borough','Neighborhood'], 1)
WCSS=[] # Model performance indicator --- Within Cluster Sum of Squares

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0).fit(NY_grouped_clustering)
    WCSS.append(kmeans.inertia_)

print (WCSS)
```

    [30.047043953503767, 28.142337991544935, 26.84013025212907, 25.956055283198904, 24.8308555411957, 24.19646209527639, 23.195158562596642, 22.935235144735433, 22.330608681361483, 22.097341077068755, 21.759179057646183, 21.06650654574184, 20.813200614316383, 20.438676231526742, 19.913500964265413, 19.51207544053955, 19.414458087213852, 19.1960151544852, 18.751570710040756, 18.449575691630375, 18.142105128663783, 17.802439692856836, 17.593415639791107, 17.267414917310226]


_We have used the Within cluster sum of squares value (intertia) in order to determine the best possible value of k to be used in our analysis_


```python
#Plotting the graph of K vs WCSS to determine "k"
plt.figure(figsize=(20,10))
plt.plot(K,WCSS)
plt.xlabel("k")
plt.ylabel("Sum of Squares")
plt.title("Determining K-value")
plt.show()
```

<img src="{{ site.url }}{{ site.baseurl }}/images/Cluster_neighbourhoods/Cluster_neighbourhoods_36_0.png">


_We'll be using the value of k as 7 as per the above graph. This seems to be the closest point of deflection, though there's no clear cut point for our analysis._

#### Running the clustering algorithm


```python
k = 7
kmeans = KMeans(n_clusters=k, random_state=0).fit(NY_grouped_clustering)
```


```python
# adding clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

NY_clustered = final_df

# Adding latitude/longitude for each neighborhood with the cluster labels
NY_clustered = NY_clustered.merge(neighborhoods_venues_sorted.set_index(['Borough','Neighborhood']), left_on=['Borough','Neighbourhood'],right_on=['Borough','Neighborhood'])

NY_clustered.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>Neighbourhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
      <th>11th Most Common Venue</th>
      <th>12th Most Common Venue</th>
      <th>13th Most Common Venue</th>
      <th>14th Most Common Venue</th>
      <th>15th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Brooklyn</td>
      <td>Bath Beach</td>
      <td>40.6018</td>
      <td>-74.0005</td>
      <td>1</td>
      <td>Chinese Restaurant</td>
      <td>Cantonese Restaurant</td>
      <td>Supplement Shop</td>
      <td>Pizza Place</td>
      <td>Bank</td>
      <td>Italian Restaurant</td>
      <td>Japanese Restaurant</td>
      <td>Dessert Shop</td>
      <td>Gas Station</td>
      <td>Bakery</td>
      <td>Tea Room</td>
      <td>Sandwich Place</td>
      <td>Eastern European Restaurant</td>
      <td>Peruvian Restaurant</td>
      <td>Middle Eastern Restaurant</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Brooklyn</td>
      <td>Bay Ridge</td>
      <td>40.634</td>
      <td>-74.0146</td>
      <td>2</td>
      <td>Chinese Restaurant</td>
      <td>Dessert Shop</td>
      <td>Seafood Restaurant</td>
      <td>Playground</td>
      <td>Irish Pub</td>
      <td>Vietnamese Restaurant</td>
      <td>Bubble Tea Shop</td>
      <td>Noodle House</td>
      <td>Nightclub</td>
      <td>Tea Room</td>
      <td>Tennis Court</td>
      <td>Gift Shop</td>
      <td>Park</td>
      <td>Fried Chicken Joint</td>
      <td>Malay Restaurant</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brooklyn</td>
      <td>Bedford Stuyvesant</td>
      <td>40.6834</td>
      <td>-73.9412</td>
      <td>2</td>
      <td>Coffee Shop</td>
      <td>Pizza Place</td>
      <td>Café</td>
      <td>Bar</td>
      <td>Fried Chicken Joint</td>
      <td>Deli / Bodega</td>
      <td>Playground</td>
      <td>Gym</td>
      <td>Lounge</td>
      <td>Gym / Fitness Center</td>
      <td>Cocktail Bar</td>
      <td>Tiki Bar</td>
      <td>Seafood Restaurant</td>
      <td>Thrift / Vintage Store</td>
      <td>Gourmet Shop</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brooklyn</td>
      <td>Bensonhurst</td>
      <td>40.605</td>
      <td>-73.9934</td>
      <td>1</td>
      <td>Chinese Restaurant</td>
      <td>Bakery</td>
      <td>Bank</td>
      <td>Cantonese Restaurant</td>
      <td>Japanese Restaurant</td>
      <td>Mobile Phone Shop</td>
      <td>Bubble Tea Shop</td>
      <td>Pizza Place</td>
      <td>Supplement Shop</td>
      <td>Kids Store</td>
      <td>Gourmet Shop</td>
      <td>Coffee Shop</td>
      <td>Pharmacy</td>
      <td>Turkish Restaurant</td>
      <td>Sushi Restaurant</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brooklyn</td>
      <td>Bergen Beach</td>
      <td>40.6204</td>
      <td>-73.9068</td>
      <td>1</td>
      <td>Chinese Restaurant</td>
      <td>American Restaurant</td>
      <td>Donut Shop</td>
      <td>Pizza Place</td>
      <td>Bus Station</td>
      <td>Gym</td>
      <td>Peruvian Restaurant</td>
      <td>Sushi Restaurant</td>
      <td>Supermarket</td>
      <td>Deli / Bodega</td>
      <td>Italian Restaurant</td>
      <td>Field</td>
      <td>Flea Market</td>
      <td>Filipino Restaurant</td>
      <td>Film Studio</td>
    </tr>
  </tbody>
</table>
</div>



#### Here's the fun part, visualize the clusters in New York City!


```python
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(k)
ys = [i + x + (i*x)**2 for i in range(k)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(NY_clustered['Latitude'], NY_clustered['Longitude'], NY_clustered['Neighbourhood'], NY_clustered['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)

map_clusters
```

<img src="{{ site.url }}{{ site.baseurl }}/images/Cluster_neighbourhoods/Cluster1.jpeg">
<img src="{{ site.url }}{{ site.baseurl }}/images/Cluster_neighbourhoods/Cluster2.jpeg">
<img src="{{ site.url }}{{ site.baseurl }}/images/Cluster_neighbourhoods/Cluster3.jpeg">


#### Getting the cluster for our current restaurant


```python
NY_clustered.loc[NY_clustered['Neighbourhood'] == "Hayes Valley"]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>Neighbourhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
      <th>11th Most Common Venue</th>
      <th>12th Most Common Venue</th>
      <th>13th Most Common Venue</th>
      <th>14th Most Common Venue</th>
      <th>15th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>315</th>
      <td>Hayes Valley, SF</td>
      <td>Hayes Valley</td>
      <td>37.7781</td>
      <td>-122.425</td>
      <td>2</td>
      <td>Clothing Store</td>
      <td>Wine Bar</td>
      <td>French Restaurant</td>
      <td>Boutique</td>
      <td>Mexican Restaurant</td>
      <td>Pizza Place</td>
      <td>Performing Arts Venue</td>
      <td>Cocktail Bar</td>
      <td>Sushi Restaurant</td>
      <td>Coffee Shop</td>
      <td>Park</td>
      <td>Bakery</td>
      <td>Optical Shop</td>
      <td>Café</td>
      <td>Juice Bar</td>
    </tr>
  </tbody>
</table>
</div>




```python
curr_rest_cluster = NY_clustered.loc[NY_clustered['Neighbourhood'] == "Hayes Valley","Cluster Labels"].item()
print(curr_rest_cluster)
```

    2



```python
# Extracting all the neighbourhoods falling under the cluster 2 - same as that of our restaurant in San Francisco
Cluster0=NY_clustered[NY_clustered['Cluster Labels']==curr_rest_cluster]
Cluster0.shape
```




    (161, 20)



_We have 160 options to open our new restaurant. This basically means that 160 neighbourhoods in New York City are similar to our current locality._
_Though this gives us a lot of options to choose from, let's try to narrow down our choices by doing another round of clustering in these 160 locations!_


```python
NY1_grouped = NY_grouped

# Adding cluster labels to our original dataframe on which the Kmeans clustering was done
NY1_grouped = NY1_grouped.merge(neighborhoods_venues_sorted[['Borough','Neighborhood','Cluster Labels']].set_index(['Borough','Neighborhood']), left_on=['Borough','Neighborhood'],right_on=['Borough','Neighborhood'])
NY1_grouped.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Adult Boutique</th>
      <th>Afghan Restaurant</th>
      <th>African Restaurant</th>
      <th>Airport Food Court</th>
      <th>Airport Lounge</th>
      <th>Airport Terminal</th>
      <th>American Restaurant</th>
      <th>Amphitheater</th>
      <th>...</th>
      <th>Wine Bar</th>
      <th>Wine Shop</th>
      <th>Winery</th>
      <th>Wings Joint</th>
      <th>Women's Store</th>
      <th>Yemeni Restaurant</th>
      <th>Yoga Studio</th>
      <th>Zoo</th>
      <th>Zoo Exhibit</th>
      <th>Cluster Labels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bronx</td>
      <td>Allerton</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bronx</td>
      <td>Bathgate</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.010000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bronx</td>
      <td>Baychester</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bronx</td>
      <td>Bedford Park</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bronx</td>
      <td>Belmont</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.017241</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 444 columns</p>
</div>




```python
# Extracting data for the neighbourhoods which has the same cluster as that of our restaurant
SF_cluster=NY1_grouped[NY1_grouped['Cluster Labels']==curr_rest_cluster]
SF_cluster.shape
```




    (161, 444)



_Preparing the revised dataframe for reclustering to identify the neighbourhood which most similar to our current neighbourhod_


```python
SF_grouped_clustering = SF_cluster.drop(['Borough','Neighborhood','Cluster Labels'], 1)
```

_Determining which neighbourhood is most similar to our office's neighbourhood by increasing the K value._
_Using K value in this manner is indirect way to calculate the distance between the neighbourhoods from a particular neighbourhood._


```python
k=75
# Kmeans clustering
kmeans = KMeans(n_clusters=k, random_state=0).fit(SF_grouped_clustering)
```


```python
# Inserting the revised clusters in the extracted dataframe
SF1_cluster=SF_cluster
SF1_cluster.drop('Cluster Labels',inplace=True,axis=1)
SF1_cluster.insert(0, 'Cluster Labels', kmeans.labels_)
SF1_cluster
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cluster Labels</th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Adult Boutique</th>
      <th>Afghan Restaurant</th>
      <th>African Restaurant</th>
      <th>Airport Food Court</th>
      <th>Airport Lounge</th>
      <th>Airport Terminal</th>
      <th>American Restaurant</th>
      <th>...</th>
      <th>Whisky Bar</th>
      <th>Wine Bar</th>
      <th>Wine Shop</th>
      <th>Winery</th>
      <th>Wings Joint</th>
      <th>Women's Store</th>
      <th>Yemeni Restaurant</th>
      <th>Yoga Studio</th>
      <th>Zoo</th>
      <th>Zoo Exhibit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>38</td>
      <td>Bronx</td>
      <td>Bronx Park South</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.035714</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.107143</td>
      <td>0.250000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>38</td>
      <td>Bronx</td>
      <td>Bronx River</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.037037</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.111111</td>
      <td>0.259259</td>
    </tr>
    <tr>
      <th>9</th>
      <td>50</td>
      <td>Bronx</td>
      <td>City Island</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.037037</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>45</td>
      <td>Bronx</td>
      <td>Clason Point</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.083333</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>40</td>
      <td>Bronx</td>
      <td>Concourse</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>294</th>
      <td>68</td>
      <td>Staten Island</td>
      <td>Pleasant Plains</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>297</th>
      <td>58</td>
      <td>Staten Island</td>
      <td>Randall Manor</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>298</th>
      <td>36</td>
      <td>Staten Island</td>
      <td>Richmond Town</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.166667</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>304</th>
      <td>43</td>
      <td>Staten Island</td>
      <td>South Beach</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>305</th>
      <td>11</td>
      <td>Staten Island</td>
      <td>St. George</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.051282</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.025641</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>161 rows × 444 columns</p>
</div>




```python
# Identifying the new cluster label
SF1_cluster[SF1_cluster['Neighborhood']=='Hayes Valley']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cluster Labels</th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Adult Boutique</th>
      <th>Afghan Restaurant</th>
      <th>African Restaurant</th>
      <th>Airport Food Court</th>
      <th>Airport Lounge</th>
      <th>Airport Terminal</th>
      <th>American Restaurant</th>
      <th>...</th>
      <th>Whisky Bar</th>
      <th>Wine Bar</th>
      <th>Wine Shop</th>
      <th>Winery</th>
      <th>Wings Joint</th>
      <th>Women's Store</th>
      <th>Yemeni Restaurant</th>
      <th>Yoga Studio</th>
      <th>Zoo</th>
      <th>Zoo Exhibit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>130</th>
      <td>11</td>
      <td>Hayes Valley, SF</td>
      <td>Hayes Valley</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.05</td>
      <td>0.01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.01</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 444 columns</p>
</div>




```python
# Extracting the details of the new cluster(=11)
Cluster_Label1=SF1_cluster.loc[SF1_cluster['Neighborhood']=='Hayes Valley','Cluster Labels'].item()
SF1_cluster[SF1_cluster['Cluster Labels']==Cluster_Label1]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cluster Labels</th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Adult Boutique</th>
      <th>Afghan Restaurant</th>
      <th>African Restaurant</th>
      <th>Airport Food Court</th>
      <th>Airport Lounge</th>
      <th>Airport Terminal</th>
      <th>American Restaurant</th>
      <th>...</th>
      <th>Whisky Bar</th>
      <th>Wine Bar</th>
      <th>Wine Shop</th>
      <th>Winery</th>
      <th>Wings Joint</th>
      <th>Women's Store</th>
      <th>Yemeni Restaurant</th>
      <th>Yoga Studio</th>
      <th>Zoo</th>
      <th>Zoo Exhibit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21</th>
      <td>11</td>
      <td>Bronx</td>
      <td>Fordham</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>11</td>
      <td>Bronx</td>
      <td>Kingsbridge</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.025641</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.025641</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>63</th>
      <td>11</td>
      <td>Brooklyn</td>
      <td>Brighton Beach</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>95</th>
      <td>11</td>
      <td>Brooklyn</td>
      <td>Homecrest</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.025000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.025000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>130</th>
      <td>11</td>
      <td>Hayes Valley, SF</td>
      <td>Hayes Valley</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.05</td>
      <td>0.010000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.010000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>145</th>
      <td>11</td>
      <td>Manhattan</td>
      <td>Harlem (Central)</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.03</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.010000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.01</td>
      <td>0.010000</td>
      <td>0.0</td>
      <td>0.01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.010000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>154</th>
      <td>11</td>
      <td>Manhattan</td>
      <td>Manhattanville</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.033333</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.033333</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>179</th>
      <td>11</td>
      <td>Queens</td>
      <td>Auburndale</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.037037</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>217</th>
      <td>11</td>
      <td>Queens</td>
      <td>Kew Gardens</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.020833</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>305</th>
      <td>11</td>
      <td>Staten Island</td>
      <td>St. George</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.051282</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.025641</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 444 columns</p>
</div>



#### The above 9 options are the closest to our current restaurant locality and can be used to open our new restaurant and help in business expansion!

### Finalizing the output


```python
SF2_cluster=SF1_cluster[SF1_cluster['Cluster Labels']==Cluster_Label1].copy()
SF2_cluster.drop("Cluster Labels",inplace=True,axis=1)
SF2_cluster
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Adult Boutique</th>
      <th>Afghan Restaurant</th>
      <th>African Restaurant</th>
      <th>Airport Food Court</th>
      <th>Airport Lounge</th>
      <th>Airport Terminal</th>
      <th>American Restaurant</th>
      <th>Amphitheater</th>
      <th>...</th>
      <th>Whisky Bar</th>
      <th>Wine Bar</th>
      <th>Wine Shop</th>
      <th>Winery</th>
      <th>Wings Joint</th>
      <th>Women's Store</th>
      <th>Yemeni Restaurant</th>
      <th>Yoga Studio</th>
      <th>Zoo</th>
      <th>Zoo Exhibit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21</th>
      <td>Bronx</td>
      <td>Fordham</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Bronx</td>
      <td>Kingsbridge</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.025641</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.025641</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>63</th>
      <td>Brooklyn</td>
      <td>Brighton Beach</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>95</th>
      <td>Brooklyn</td>
      <td>Homecrest</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.025000</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.025000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>130</th>
      <td>Hayes Valley, SF</td>
      <td>Hayes Valley</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.05</td>
      <td>0.010000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.010000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>145</th>
      <td>Manhattan</td>
      <td>Harlem (Central)</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.03</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.010000</td>
      <td>0.01</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.01</td>
      <td>0.010000</td>
      <td>0.0</td>
      <td>0.01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.010000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>154</th>
      <td>Manhattan</td>
      <td>Manhattanville</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.033333</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.033333</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>179</th>
      <td>Queens</td>
      <td>Auburndale</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.037037</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>217</th>
      <td>Queens</td>
      <td>Kew Gardens</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.020833</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>305</th>
      <td>Staten Island</td>
      <td>St. George</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.051282</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.025641</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 443 columns</p>
</div>




```python
columns1 = ['Borough','Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns1.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns1.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
Final_venues_sorted = pd.DataFrame(columns=columns1)
Final_venues_sorted['Borough']=SF2_cluster['Borough']
Final_venues_sorted['Neighborhood'] = SF2_cluster['Neighborhood']

for ind in np.arange(SF2_cluster.shape[0]):
    Final_venues_sorted.iloc[ind, 2:] = return_most_common_venues(SF2_cluster.iloc[ind, :], num_top_venues)

Final_venues_sorted
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
      <th>11th Most Common Venue</th>
      <th>12th Most Common Venue</th>
      <th>13th Most Common Venue</th>
      <th>14th Most Common Venue</th>
      <th>15th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21</th>
      <td>Bronx</td>
      <td>Fordham</td>
      <td>Shoe Store</td>
      <td>Fast Food Restaurant</td>
      <td>Coffee Shop</td>
      <td>Sandwich Place</td>
      <td>Clothing Store</td>
      <td>Spanish Restaurant</td>
      <td>Supplement Shop</td>
      <td>Bank</td>
      <td>Pharmacy</td>
      <td>Gym / Fitness Center</td>
      <td>Pizza Place</td>
      <td>Café</td>
      <td>Deli / Bodega</td>
      <td>Miscellaneous Shop</td>
      <td>Mobile Phone Shop</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Bronx</td>
      <td>Kingsbridge</td>
      <td>Supermarket</td>
      <td>Café</td>
      <td>Gym</td>
      <td>Pizza Place</td>
      <td>Donut Shop</td>
      <td>Mexican Restaurant</td>
      <td>Sandwich Place</td>
      <td>Spanish Restaurant</td>
      <td>Grocery Store</td>
      <td>Thrift / Vintage Store</td>
      <td>Gourmet Shop</td>
      <td>Breakfast Spot</td>
      <td>Supplement Shop</td>
      <td>Steakhouse</td>
      <td>Burger Joint</td>
    </tr>
    <tr>
      <th>63</th>
      <td>Brooklyn</td>
      <td>Brighton Beach</td>
      <td>Supermarket</td>
      <td>Bakery</td>
      <td>Eastern European Restaurant</td>
      <td>Health &amp; Beauty Service</td>
      <td>Grocery Store</td>
      <td>Donut Shop</td>
      <td>Theater</td>
      <td>Mobile Phone Shop</td>
      <td>Café</td>
      <td>Flower Shop</td>
      <td>Gourmet Shop</td>
      <td>Bar</td>
      <td>Food Truck</td>
      <td>Lounge</td>
      <td>Bus Line</td>
    </tr>
    <tr>
      <th>95</th>
      <td>Brooklyn</td>
      <td>Homecrest</td>
      <td>Sushi Restaurant</td>
      <td>Café</td>
      <td>Pizza Place</td>
      <td>Bagel Shop</td>
      <td>Mobile Phone Shop</td>
      <td>Ice Cream Shop</td>
      <td>Market</td>
      <td>Seafood Restaurant</td>
      <td>Bank</td>
      <td>Bar</td>
      <td>Sandwich Place</td>
      <td>Gym</td>
      <td>Coffee Shop</td>
      <td>Mediterranean Restaurant</td>
      <td>Eastern European Restaurant</td>
    </tr>
    <tr>
      <th>130</th>
      <td>Hayes Valley, SF</td>
      <td>Hayes Valley</td>
      <td>Clothing Store</td>
      <td>Wine Bar</td>
      <td>French Restaurant</td>
      <td>Boutique</td>
      <td>Mexican Restaurant</td>
      <td>Pizza Place</td>
      <td>Performing Arts Venue</td>
      <td>Cocktail Bar</td>
      <td>Sushi Restaurant</td>
      <td>Coffee Shop</td>
      <td>Park</td>
      <td>Bakery</td>
      <td>Optical Shop</td>
      <td>Café</td>
      <td>Juice Bar</td>
    </tr>
    <tr>
      <th>145</th>
      <td>Manhattan</td>
      <td>Harlem (Central)</td>
      <td>Southern / Soul Food Restaurant</td>
      <td>Mobile Phone Shop</td>
      <td>Clothing Store</td>
      <td>Theater</td>
      <td>Pizza Place</td>
      <td>Burger Joint</td>
      <td>African Restaurant</td>
      <td>Cosmetics Shop</td>
      <td>Sandwich Place</td>
      <td>Café</td>
      <td>Arts &amp; Crafts Store</td>
      <td>Mexican Restaurant</td>
      <td>Japanese Restaurant</td>
      <td>Jazz Club</td>
      <td>Deli / Bodega</td>
    </tr>
    <tr>
      <th>154</th>
      <td>Manhattan</td>
      <td>Manhattanville</td>
      <td>Art Gallery</td>
      <td>Fried Chicken Joint</td>
      <td>Coffee Shop</td>
      <td>Boutique</td>
      <td>Seafood Restaurant</td>
      <td>Chinese Restaurant</td>
      <td>Sandwich Place</td>
      <td>Lounge</td>
      <td>Ethiopian Restaurant</td>
      <td>Bank</td>
      <td>Public Art</td>
      <td>College Theater</td>
      <td>Park</td>
      <td>Spa</td>
      <td>Food &amp; Drink Shop</td>
    </tr>
    <tr>
      <th>179</th>
      <td>Queens</td>
      <td>Auburndale</td>
      <td>Pharmacy</td>
      <td>Café</td>
      <td>Hookah Bar</td>
      <td>Sandwich Place</td>
      <td>Train Station</td>
      <td>Train</td>
      <td>Lounge</td>
      <td>Pizza Place</td>
      <td>Miscellaneous Shop</td>
      <td>Greek Restaurant</td>
      <td>Athletics &amp; Sports</td>
      <td>Toy / Game Store</td>
      <td>Donut Shop</td>
      <td>Korean Restaurant</td>
      <td>Vietnamese Restaurant</td>
    </tr>
    <tr>
      <th>217</th>
      <td>Queens</td>
      <td>Kew Gardens</td>
      <td>Pizza Place</td>
      <td>Metro Station</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>Mediterranean Restaurant</td>
      <td>Donut Shop</td>
      <td>Cosmetics Shop</td>
      <td>Deli / Bodega</td>
      <td>Nail Salon</td>
      <td>Bagel Shop</td>
      <td>Supplement Shop</td>
      <td>Bank</td>
      <td>Train Station</td>
      <td>Gym / Fitness Center</td>
      <td>Grocery Store</td>
    </tr>
    <tr>
      <th>305</th>
      <td>Staten Island</td>
      <td>St. George</td>
      <td>Clothing Store</td>
      <td>Italian Restaurant</td>
      <td>Sporting Goods Shop</td>
      <td>American Restaurant</td>
      <td>Bar</td>
      <td>Bakery</td>
      <td>Museum</td>
      <td>Monument / Landmark</td>
      <td>Deli / Bodega</td>
      <td>Tapas Restaurant</td>
      <td>Pharmacy</td>
      <td>Theater</td>
      <td>Farmers Market</td>
      <td>Bus Stop</td>
      <td>Seafood Restaurant</td>
    </tr>
  </tbody>
</table>
</div>



### Result: We were able to identify 9 closest neighbourhoods in New York City which are ideal to open up our new restaurant and expand our business.
#### There are a few things than can be further developed, like a better value of "k" since elbow method didn't provide a clear option. We could use Average silhouette method or Gap statistic method in order to do so.
#### The final decision needs to be made basis a lot more information like availability of space, taxes involved, easy of transportation etc.
