---
title: "Predict House price"
date: 2020-11-19
tags: [data science, deep learning, linear regression, neural network ]
excerpt: "Data Science, Deep Learning, Neural Network"
mathjax: "true"
---


### Objective: To determine the price of the house using deep neural network. We've been provided with historical data and fetaures of the house.  
[Source:](https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/) Udemy | Python for Data Science and Machine Learning Bootcamp  
Data used in the below analysis: [Housing data from Kaggle](https://www.kaggle.com/harlfoxem/housesalesprediction).


```python
#Importing libraries as required
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
%matplotlib inline
```


```python
sns.set_style('whitegrid')
```


```python
data = pd.read_csv('DATA/kc_house_data.csv') #read dataset
```

### Starting with some EDA! (Explanatory Data Analysis)


```python
data.isnull().sum() #checking for null data
```




    id               0
    date             0
    price            0
    bedrooms         0
    bathrooms        0
    sqft_living      0
    sqft_lot         0
    floors           0
    waterfront       0
    view             0
    condition        0
    grade            0
    sqft_above       0
    sqft_basement    0
    yr_built         0
    yr_renovated     0
    zipcode          0
    lat              0
    long             0
    sqft_living15    0
    sqft_lot15       0
    dtype: int64



_The data has no null values_


```python
#viewing some basic info about the dataset
data.describe().transpose()
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>id</th>
      <td>21597.0</td>
      <td>4.580474e+09</td>
      <td>2.876736e+09</td>
      <td>1.000102e+06</td>
      <td>2.123049e+09</td>
      <td>3.904930e+09</td>
      <td>7.308900e+09</td>
      <td>9.900000e+09</td>
    </tr>
    <tr>
      <th>price</th>
      <td>21597.0</td>
      <td>5.402966e+05</td>
      <td>3.673681e+05</td>
      <td>7.800000e+04</td>
      <td>3.220000e+05</td>
      <td>4.500000e+05</td>
      <td>6.450000e+05</td>
      <td>7.700000e+06</td>
    </tr>
    <tr>
      <th>bedrooms</th>
      <td>21597.0</td>
      <td>3.373200e+00</td>
      <td>9.262989e-01</td>
      <td>1.000000e+00</td>
      <td>3.000000e+00</td>
      <td>3.000000e+00</td>
      <td>4.000000e+00</td>
      <td>3.300000e+01</td>
    </tr>
    <tr>
      <th>bathrooms</th>
      <td>21597.0</td>
      <td>2.115826e+00</td>
      <td>7.689843e-01</td>
      <td>5.000000e-01</td>
      <td>1.750000e+00</td>
      <td>2.250000e+00</td>
      <td>2.500000e+00</td>
      <td>8.000000e+00</td>
    </tr>
    <tr>
      <th>sqft_living</th>
      <td>21597.0</td>
      <td>2.080322e+03</td>
      <td>9.181061e+02</td>
      <td>3.700000e+02</td>
      <td>1.430000e+03</td>
      <td>1.910000e+03</td>
      <td>2.550000e+03</td>
      <td>1.354000e+04</td>
    </tr>
    <tr>
      <th>sqft_lot</th>
      <td>21597.0</td>
      <td>1.509941e+04</td>
      <td>4.141264e+04</td>
      <td>5.200000e+02</td>
      <td>5.040000e+03</td>
      <td>7.618000e+03</td>
      <td>1.068500e+04</td>
      <td>1.651359e+06</td>
    </tr>
    <tr>
      <th>floors</th>
      <td>21597.0</td>
      <td>1.494096e+00</td>
      <td>5.396828e-01</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.500000e+00</td>
      <td>2.000000e+00</td>
      <td>3.500000e+00</td>
    </tr>
    <tr>
      <th>waterfront</th>
      <td>21597.0</td>
      <td>7.547345e-03</td>
      <td>8.654900e-02</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>view</th>
      <td>21597.0</td>
      <td>2.342918e-01</td>
      <td>7.663898e-01</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>4.000000e+00</td>
    </tr>
    <tr>
      <th>condition</th>
      <td>21597.0</td>
      <td>3.409825e+00</td>
      <td>6.505456e-01</td>
      <td>1.000000e+00</td>
      <td>3.000000e+00</td>
      <td>3.000000e+00</td>
      <td>4.000000e+00</td>
      <td>5.000000e+00</td>
    </tr>
    <tr>
      <th>grade</th>
      <td>21597.0</td>
      <td>7.657915e+00</td>
      <td>1.173200e+00</td>
      <td>3.000000e+00</td>
      <td>7.000000e+00</td>
      <td>7.000000e+00</td>
      <td>8.000000e+00</td>
      <td>1.300000e+01</td>
    </tr>
    <tr>
      <th>sqft_above</th>
      <td>21597.0</td>
      <td>1.788597e+03</td>
      <td>8.277598e+02</td>
      <td>3.700000e+02</td>
      <td>1.190000e+03</td>
      <td>1.560000e+03</td>
      <td>2.210000e+03</td>
      <td>9.410000e+03</td>
    </tr>
    <tr>
      <th>sqft_basement</th>
      <td>21597.0</td>
      <td>2.917250e+02</td>
      <td>4.426678e+02</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>5.600000e+02</td>
      <td>4.820000e+03</td>
    </tr>
    <tr>
      <th>yr_built</th>
      <td>21597.0</td>
      <td>1.971000e+03</td>
      <td>2.937523e+01</td>
      <td>1.900000e+03</td>
      <td>1.951000e+03</td>
      <td>1.975000e+03</td>
      <td>1.997000e+03</td>
      <td>2.015000e+03</td>
    </tr>
    <tr>
      <th>yr_renovated</th>
      <td>21597.0</td>
      <td>8.446479e+01</td>
      <td>4.018214e+02</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.015000e+03</td>
    </tr>
    <tr>
      <th>zipcode</th>
      <td>21597.0</td>
      <td>9.807795e+04</td>
      <td>5.351307e+01</td>
      <td>9.800100e+04</td>
      <td>9.803300e+04</td>
      <td>9.806500e+04</td>
      <td>9.811800e+04</td>
      <td>9.819900e+04</td>
    </tr>
    <tr>
      <th>lat</th>
      <td>21597.0</td>
      <td>4.756009e+01</td>
      <td>1.385518e-01</td>
      <td>4.715590e+01</td>
      <td>4.747110e+01</td>
      <td>4.757180e+01</td>
      <td>4.767800e+01</td>
      <td>4.777760e+01</td>
    </tr>
    <tr>
      <th>long</th>
      <td>21597.0</td>
      <td>-1.222140e+02</td>
      <td>1.407235e-01</td>
      <td>-1.225190e+02</td>
      <td>-1.223280e+02</td>
      <td>-1.222310e+02</td>
      <td>-1.221250e+02</td>
      <td>-1.213150e+02</td>
    </tr>
    <tr>
      <th>sqft_living15</th>
      <td>21597.0</td>
      <td>1.986620e+03</td>
      <td>6.852305e+02</td>
      <td>3.990000e+02</td>
      <td>1.490000e+03</td>
      <td>1.840000e+03</td>
      <td>2.360000e+03</td>
      <td>6.210000e+03</td>
    </tr>
    <tr>
      <th>sqft_lot15</th>
      <td>21597.0</td>
      <td>1.275828e+04</td>
      <td>2.727444e+04</td>
      <td>6.510000e+02</td>
      <td>5.100000e+03</td>
      <td>7.620000e+03</td>
      <td>1.008300e+04</td>
      <td>8.712000e+05</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.head()
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
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>...</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7129300520</td>
      <td>10/13/2014</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>1180</td>
      <td>0</td>
      <td>1955</td>
      <td>0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>5650</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6414100192</td>
      <td>12/9/2014</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>2170</td>
      <td>400</td>
      <td>1951</td>
      <td>1991</td>
      <td>98125</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>7639</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5631500400</td>
      <td>2/25/2015</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>6</td>
      <td>770</td>
      <td>0</td>
      <td>1933</td>
      <td>0</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>8062</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2487200875</td>
      <td>12/9/2014</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>1050</td>
      <td>910</td>
      <td>1965</td>
      <td>0</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360</td>
      <td>5000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1954400510</td>
      <td>2/18/2015</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>8</td>
      <td>1680</td>
      <td>0</td>
      <td>1987</td>
      <td>0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>7503</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
#looking at the price range, the feature to be predicted
plt.figure(figsize=(12,8))
sns.distplot(data['price'])
```


<img src="{{ site.url }}{{ site.baseurl }}/images/House Predictions/dist1_keras.png">



_The data is mostly concentrated around 1,000,000 - 2,000,000 with few houses at 3,000,000 and even at around 7,500,000_


```python
sns.countplot(data['bedrooms'])
```


<img src="{{ site.url }}{{ site.baseurl }}/images/House Predictions/count1_keras.png">


_The data mostly contains houses with 3 bedrooms_


```python
#Checking the correlation of price with other features!
data.corr()['price'].sort_values()
```




    zipcode         -0.053402
    id              -0.016772
    long             0.022036
    condition        0.036056
    yr_built         0.053953
    sqft_lot15       0.082845
    sqft_lot         0.089876
    yr_renovated     0.126424
    floors           0.256804
    waterfront       0.266398
    lat              0.306692
    bedrooms         0.308787
    sqft_basement    0.323799
    view             0.397370
    bathrooms        0.525906
    sqft_living15    0.585241
    sqft_above       0.605368
    grade            0.667951
    sqft_living      0.701917
    price            1.000000
    Name: price, dtype: float64



_The most correlated is the Square footage of the apartments interior living space with 0.71. We can see that in the below plot as well!_


```python
plt.figure(figsize=(12,8))
sns.scatterplot(x='price',y='sqft_living',data=data)
```


<img src="{{ site.url }}{{ site.baseurl }}/images/House Predictions/scatter1_keras.png">



```python
#Num of bedrooms vs price of the house
plt.figure(figsize=(15,10))
sns.boxplot(x='bedrooms',y='price',data=data)
```


<img src="{{ site.url }}{{ site.baseurl }}/images/House Predictions/box_keras.png">



```python
#price vs longitude
plt.figure(figsize=(12,8))
sns.scatterplot(x='price',y='long',data=data)
```

<img src="{{ site.url }}{{ site.baseurl }}/images/House Predictions/scatter2_keras.png">



```python
#price vs latitude
plt.figure(figsize=(12,8))
sns.scatterplot(x='price',y='lat',data=data)
```



<img src="{{ site.url }}{{ site.baseurl }}/images/House Predictions/scatter3_keras.png">



```python
#plotting price with longitude and langitude can give us expensive areas in the given region
plt.figure(figsize=(12,8))
sns.scatterplot(x='long',y='lat',data=data,hue='price')
```


<img src="{{ site.url }}{{ site.baseurl }}/images/House Predictions/scatter4_keras.png">


_We can clean data to get a better plot of house prices_


```python
#getting the price outliers - most priced houses
data.sort_values('price',ascending=False).head(20)
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
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>...</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7245</th>
      <td>6762700020</td>
      <td>10/13/2014</td>
      <td>7700000.0</td>
      <td>6</td>
      <td>8.00</td>
      <td>12050</td>
      <td>27600</td>
      <td>2.5</td>
      <td>0</td>
      <td>3</td>
      <td>...</td>
      <td>13</td>
      <td>8570</td>
      <td>3480</td>
      <td>1910</td>
      <td>1987</td>
      <td>98102</td>
      <td>47.6298</td>
      <td>-122.323</td>
      <td>3940</td>
      <td>8800</td>
    </tr>
    <tr>
      <th>3910</th>
      <td>9808700762</td>
      <td>6/11/2014</td>
      <td>7060000.0</td>
      <td>5</td>
      <td>4.50</td>
      <td>10040</td>
      <td>37325</td>
      <td>2.0</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>11</td>
      <td>7680</td>
      <td>2360</td>
      <td>1940</td>
      <td>2001</td>
      <td>98004</td>
      <td>47.6500</td>
      <td>-122.214</td>
      <td>3930</td>
      <td>25449</td>
    </tr>
    <tr>
      <th>9245</th>
      <td>9208900037</td>
      <td>9/19/2014</td>
      <td>6890000.0</td>
      <td>6</td>
      <td>7.75</td>
      <td>9890</td>
      <td>31374</td>
      <td>2.0</td>
      <td>0</td>
      <td>4</td>
      <td>...</td>
      <td>13</td>
      <td>8860</td>
      <td>1030</td>
      <td>2001</td>
      <td>0</td>
      <td>98039</td>
      <td>47.6305</td>
      <td>-122.240</td>
      <td>4540</td>
      <td>42730</td>
    </tr>
    <tr>
      <th>4407</th>
      <td>2470100110</td>
      <td>8/4/2014</td>
      <td>5570000.0</td>
      <td>5</td>
      <td>5.75</td>
      <td>9200</td>
      <td>35069</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>13</td>
      <td>6200</td>
      <td>3000</td>
      <td>2001</td>
      <td>0</td>
      <td>98039</td>
      <td>47.6289</td>
      <td>-122.233</td>
      <td>3560</td>
      <td>24345</td>
    </tr>
    <tr>
      <th>1446</th>
      <td>8907500070</td>
      <td>4/13/2015</td>
      <td>5350000.0</td>
      <td>5</td>
      <td>5.00</td>
      <td>8000</td>
      <td>23985</td>
      <td>2.0</td>
      <td>0</td>
      <td>4</td>
      <td>...</td>
      <td>12</td>
      <td>6720</td>
      <td>1280</td>
      <td>2009</td>
      <td>0</td>
      <td>98004</td>
      <td>47.6232</td>
      <td>-122.220</td>
      <td>4600</td>
      <td>21750</td>
    </tr>
    <tr>
      <th>1313</th>
      <td>7558700030</td>
      <td>4/13/2015</td>
      <td>5300000.0</td>
      <td>6</td>
      <td>6.00</td>
      <td>7390</td>
      <td>24829</td>
      <td>2.0</td>
      <td>1</td>
      <td>4</td>
      <td>...</td>
      <td>12</td>
      <td>5000</td>
      <td>2390</td>
      <td>1991</td>
      <td>0</td>
      <td>98040</td>
      <td>47.5631</td>
      <td>-122.210</td>
      <td>4320</td>
      <td>24619</td>
    </tr>
    <tr>
      <th>1162</th>
      <td>1247600105</td>
      <td>10/20/2014</td>
      <td>5110000.0</td>
      <td>5</td>
      <td>5.25</td>
      <td>8010</td>
      <td>45517</td>
      <td>2.0</td>
      <td>1</td>
      <td>4</td>
      <td>...</td>
      <td>12</td>
      <td>5990</td>
      <td>2020</td>
      <td>1999</td>
      <td>0</td>
      <td>98033</td>
      <td>47.6767</td>
      <td>-122.211</td>
      <td>3430</td>
      <td>26788</td>
    </tr>
    <tr>
      <th>8085</th>
      <td>1924059029</td>
      <td>6/17/2014</td>
      <td>4670000.0</td>
      <td>5</td>
      <td>6.75</td>
      <td>9640</td>
      <td>13068</td>
      <td>1.0</td>
      <td>1</td>
      <td>4</td>
      <td>...</td>
      <td>12</td>
      <td>4820</td>
      <td>4820</td>
      <td>1983</td>
      <td>2009</td>
      <td>98040</td>
      <td>47.5570</td>
      <td>-122.210</td>
      <td>3270</td>
      <td>10454</td>
    </tr>
    <tr>
      <th>2624</th>
      <td>7738500731</td>
      <td>8/15/2014</td>
      <td>4500000.0</td>
      <td>5</td>
      <td>5.50</td>
      <td>6640</td>
      <td>40014</td>
      <td>2.0</td>
      <td>1</td>
      <td>4</td>
      <td>...</td>
      <td>12</td>
      <td>6350</td>
      <td>290</td>
      <td>2004</td>
      <td>0</td>
      <td>98155</td>
      <td>47.7493</td>
      <td>-122.280</td>
      <td>3030</td>
      <td>23408</td>
    </tr>
    <tr>
      <th>8629</th>
      <td>3835500195</td>
      <td>6/18/2014</td>
      <td>4490000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>6430</td>
      <td>27517</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>12</td>
      <td>6430</td>
      <td>0</td>
      <td>2001</td>
      <td>0</td>
      <td>98004</td>
      <td>47.6208</td>
      <td>-122.219</td>
      <td>3720</td>
      <td>14592</td>
    </tr>
    <tr>
      <th>12358</th>
      <td>6065300370</td>
      <td>5/6/2015</td>
      <td>4210000.0</td>
      <td>5</td>
      <td>6.00</td>
      <td>7440</td>
      <td>21540</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>12</td>
      <td>5550</td>
      <td>1890</td>
      <td>2003</td>
      <td>0</td>
      <td>98006</td>
      <td>47.5692</td>
      <td>-122.189</td>
      <td>4740</td>
      <td>19329</td>
    </tr>
    <tr>
      <th>4145</th>
      <td>6447300265</td>
      <td>10/14/2014</td>
      <td>4000000.0</td>
      <td>4</td>
      <td>5.50</td>
      <td>7080</td>
      <td>16573</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>12</td>
      <td>5760</td>
      <td>1320</td>
      <td>2008</td>
      <td>0</td>
      <td>98039</td>
      <td>47.6151</td>
      <td>-122.224</td>
      <td>3140</td>
      <td>15996</td>
    </tr>
    <tr>
      <th>2083</th>
      <td>8106100105</td>
      <td>11/14/2014</td>
      <td>3850000.0</td>
      <td>4</td>
      <td>4.25</td>
      <td>5770</td>
      <td>21300</td>
      <td>2.0</td>
      <td>1</td>
      <td>4</td>
      <td>...</td>
      <td>11</td>
      <td>5770</td>
      <td>0</td>
      <td>1980</td>
      <td>0</td>
      <td>98040</td>
      <td>47.5850</td>
      <td>-122.222</td>
      <td>4620</td>
      <td>22748</td>
    </tr>
    <tr>
      <th>7028</th>
      <td>853200010</td>
      <td>7/1/2014</td>
      <td>3800000.0</td>
      <td>5</td>
      <td>5.50</td>
      <td>7050</td>
      <td>42840</td>
      <td>1.0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>13</td>
      <td>4320</td>
      <td>2730</td>
      <td>1978</td>
      <td>0</td>
      <td>98004</td>
      <td>47.6229</td>
      <td>-122.220</td>
      <td>5070</td>
      <td>20570</td>
    </tr>
    <tr>
      <th>19002</th>
      <td>2303900100</td>
      <td>9/11/2014</td>
      <td>3800000.0</td>
      <td>3</td>
      <td>4.25</td>
      <td>5510</td>
      <td>35000</td>
      <td>2.0</td>
      <td>0</td>
      <td>4</td>
      <td>...</td>
      <td>13</td>
      <td>4910</td>
      <td>600</td>
      <td>1997</td>
      <td>0</td>
      <td>98177</td>
      <td>47.7296</td>
      <td>-122.370</td>
      <td>3430</td>
      <td>45302</td>
    </tr>
    <tr>
      <th>16288</th>
      <td>7397300170</td>
      <td>5/30/2014</td>
      <td>3710000.0</td>
      <td>4</td>
      <td>3.50</td>
      <td>5550</td>
      <td>28078</td>
      <td>2.0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>12</td>
      <td>3350</td>
      <td>2200</td>
      <td>2000</td>
      <td>0</td>
      <td>98039</td>
      <td>47.6395</td>
      <td>-122.234</td>
      <td>2980</td>
      <td>19602</td>
    </tr>
    <tr>
      <th>18467</th>
      <td>4389201095</td>
      <td>5/11/2015</td>
      <td>3650000.0</td>
      <td>5</td>
      <td>3.75</td>
      <td>5020</td>
      <td>8694</td>
      <td>2.0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>12</td>
      <td>3970</td>
      <td>1050</td>
      <td>2007</td>
      <td>0</td>
      <td>98004</td>
      <td>47.6146</td>
      <td>-122.213</td>
      <td>4190</td>
      <td>11275</td>
    </tr>
    <tr>
      <th>6502</th>
      <td>4217402115</td>
      <td>4/21/2015</td>
      <td>3650000.0</td>
      <td>6</td>
      <td>4.75</td>
      <td>5480</td>
      <td>19401</td>
      <td>1.5</td>
      <td>1</td>
      <td>4</td>
      <td>...</td>
      <td>11</td>
      <td>3910</td>
      <td>1570</td>
      <td>1936</td>
      <td>0</td>
      <td>98105</td>
      <td>47.6515</td>
      <td>-122.277</td>
      <td>3510</td>
      <td>15810</td>
    </tr>
    <tr>
      <th>15241</th>
      <td>2425049063</td>
      <td>9/11/2014</td>
      <td>3640000.0</td>
      <td>4</td>
      <td>3.25</td>
      <td>4830</td>
      <td>22257</td>
      <td>2.0</td>
      <td>1</td>
      <td>4</td>
      <td>...</td>
      <td>11</td>
      <td>4830</td>
      <td>0</td>
      <td>1990</td>
      <td>0</td>
      <td>98039</td>
      <td>47.6409</td>
      <td>-122.241</td>
      <td>3820</td>
      <td>25582</td>
    </tr>
    <tr>
      <th>19133</th>
      <td>3625049042</td>
      <td>10/11/2014</td>
      <td>3640000.0</td>
      <td>5</td>
      <td>6.00</td>
      <td>5490</td>
      <td>19897</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>12</td>
      <td>5490</td>
      <td>0</td>
      <td>2005</td>
      <td>0</td>
      <td>98039</td>
      <td>47.6165</td>
      <td>-122.236</td>
      <td>2910</td>
      <td>17600</td>
    </tr>
  </tbody>
</table>
<p>20 rows × 21 columns</p>
</div>



_We can disregard a % of entries to remove outliers and get better model predictions!_


```python
len(data)*0.01
```




    215.97




```python
non_top_1_percent = data.sort_values('price',ascending=False).iloc[216:]
```


```python
plt.figure(figsize=(12,8))
sns.scatterplot(x='long',y='lat',data=non_top_1_percent,
                edgecolor=None, alpha=0.2, palette = 'RdYlGn', hue='price')
```


<img src="{{ site.url }}{{ site.baseurl }}/images/House Predictions/scatter5_keras.png">


_We can clearly see the most expensive housing areas now!_


```python
#Checking the price of houses on the waterfront
#As per the above plot, they seem to be more expensive
sns.boxplot(x='waterfront',y='price',data=data)
```

<img src="{{ site.url }}{{ site.baseurl }}/images/House Predictions/box2_keras.png">


### Start with feature engineering! We need to clean data for model to give better results!


```python
#Dropping unnecessary fields!
data.drop('id',axis=1,inplace=True)
```


```python
#converting date into something usefull
data['date'] = pd.to_datetime(data['date'])
data['year'] = data['date'].apply(lambda x : x.year)
data['month'] = data['date'].apply(lambda x : x.month)
```


```python
#checking month vs price of house
plt.figure(figsize=(12,8))
sns.boxplot(x='month',y='price',data=data)
```



<img src="{{ site.url }}{{ site.baseurl }}/images/House Predictions/box3_keras.png">



```python
data.groupby(by='month').mean()['price'].plot()
```


<img src="{{ site.url }}{{ site.baseurl }}/images/House Predictions/plot1_keras.png">



```python
#price vs year
data.groupby(by='year').mean()['price'].plot()
```


<img src="{{ site.url }}{{ site.baseurl }}/images/House Predictions/plot2_keras.png">


_There seems to an expected relationship with year and mean prices of the house_


```python
data.drop('date',axis=1,inplace=True) #dropping date as we already have extracted its features
```


```python
data['zipcode'].value_counts() #checking unique values to get the idea of the data
```




    98103    602
    98038    589
    98115    583
    98052    574
    98117    553
            ...
    98102    104
    98010    100
    98024     80
    98148     57
    98039     50
    Name: zipcode, Length: 70, dtype: int64




```python
#dropping as it don't seem to affect price much
#Seen before, has -ve correlation
data.drop('zipcode',axis=1,inplace=True)
```


```python
data['yr_renovated'].value_counts()
```




    0       20683
    2014       91
    2013       37
    2003       36
    2000       35
            ...  
    1934        1
    1959        1
    1951        1
    1948        1
    1944        1
    Name: yr_renovated, Length: 70, dtype: int64



_We can convert the above data into two options, renovated or non renovated for our price predictions. We are lucky we have years and renovated already in high correlation with price, intuitively we can use the data as is_


```python
data['sqft_basement'].value_counts()
```




    0       13110
    600       221
    700       218
    500       214
    800       206
            ...  
    792         1
    2590        1
    935         1
    2390        1
    248         1
    Name: sqft_basement, Length: 306, dtype: int64



_Same case with the basement area!_

### Training the model


```python
#train test split
X = data.drop('price', axis=1).values
y = data['price'].values
```


```python
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size = 0.3,
                                                random_state=101)
```


```python
#Scaling the data
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
#fit and transform together
Xtrain = scale.fit_transform(Xtrain)
#we don't fit test data so that no info leak is there while training data
Xtest = scale.transform(Xtest)
```


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
```


```python
model = Sequential()

model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')
```


```python
model.fit(x=Xtrain,y=ytrain,
          validation_data=(Xtest,ytest),
          batch_size = 128, epochs=400)
```

    Epoch 1/400
    119/119 [==============================] - 0s 2ms/step - loss: 430241349632.0000 - val_loss: 418922692608.0000
    Epoch 2/400
    119/119 [==============================] - 0s 1ms/step - loss: 429360381952.0000 - val_loss: 415949946880.0000
    Epoch 3/400
    119/119 [==============================] - 0s 1ms/step - loss: 417961672704.0000 - val_loss: 390411911168.0000
    Epoch 4/400
    119/119 [==============================] - 0s 1ms/step - loss: 362967072768.0000 - val_loss: 298817191936.0000
    Epoch 5/400
    119/119 [==============================] - 0s 1ms/step - loss: 234846666752.0000 - val_loss: 154322370560.0000
    Epoch 6/400
    119/119 [==============================] - 0s 1ms/step - loss: 121333202944.0000 - val_loss: 97280843776.0000
    Epoch 7/400
    119/119 [==============================] - 0s 1ms/step - loss: 98753257472.0000 - val_loss: 94313840640.0000
    .
    .
    .
    Epoch 400/400
    119/119 [==============================] - 0s 2ms/step - loss: 29929684992.0000 - val_loss: 27626838016.0000






```python
losses = pd.DataFrame(model.history.history)
```


```python
losses.plot()
```



<img src="{{ site.url }}{{ site.baseurl }}/images/House Predictions/plot3_keras.png">


_Expected behaviour of our model with loss vs Val_loss. We can train at more epochs as loss is still over val_loss and it's not yet overfitting_

### Evaluating the model performance


```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score
```


```python
predictions = model.predict(Xtest)
```


```python
print("Mean_absolute_error:", mean_absolute_error(ytest,predictions))
print("Mean_squared_error:",  mean_squared_error(ytest,predictions))
print("Root_Mean_Squared_error:", np.sqrt(mean_squared_error(ytest,predictions)))
```

    Mean_absolute_error: 103103.7446638696
    Mean_squared_error: 27626834030.167973
    Root_Mean_Squared_error: 166213.2185783308



```python
data['price'].describe()['mean'] #checking the mean price
```




    540296.5735055795



_Our model is off by about 19.08% (Mean absolute error) of the mean value of price. It is not too bad but not great as such_


```python
#this tells how much variance our model can explain
#lower value is worse, best value is 1.0
explained_variance_score(ytest,predictions)
```




    0.7918666773104263




```python
plt.figure(figsize=(12,6))
plt.scatter(ytest,predictions)
# Perfect predictions
plt.plot(ytest,ytest,'r')
```


<img src="{{ site.url }}{{ site.baseurl }}/images/House Predictions/scatter6_keras.png">


_The plot of predicted vs true values seem fine except for some outliers_

### Predicting a new value


```python
single_house = data.drop('price',axis=1).iloc[0] #creating a entry to check the model
```


```python
single_house = scale.transform(single_house.values.reshape(-1, 19)) #transform to fit into model
```


```python
print("Error:", abs(data.iloc[0]['price']-model.predict(single_house)[0][0]))
```

    Error: 58779.65625


_We can try to reduce error on the current rendered model by removing outliers and by increasing the epochs. We'll try  with removing outliers._

### Re-training and evaluating the model


```python
non_top_1_percent.drop('id',axis=1,inplace=True)
non_top_1_percent['date'] = pd.to_datetime(non_top_1_percent['date'])
non_top_1_percent['year'] = non_top_1_percent['date'].apply(lambda x : x.year)
non_top_1_percent['month'] = non_top_1_percent['date'].apply(lambda x : x.month)
non_top_1_percent.drop('date',axis=1,inplace=True)
non_top_1_percent.drop('zipcode',axis=1,inplace=True)
```


```python
#train test split
X = non_top_1_percent.drop('price', axis=1).values
y = non_top_1_percent['price'].values
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size = 0.3,
                                                random_state=101)
Xtrain = scale.fit_transform(Xtrain)
Xtest = scale.transform(Xtest)
```


```python
model.fit(x=Xtrain,y=ytrain,
          validation_data=(Xtest,ytest),
          batch_size = 128, epochs=400)
```

    Epoch 1/400
    117/117 [==============================] - 0s 2ms/step - loss: 25183025152.0000 - val_loss: 23184056320.0000
    Epoch 2/400
    117/117 [==============================] - 0s 2ms/step - loss: 21834297344.0000 - val_loss: 22692438016.0000
    Epoch 3/400
    117/117 [==============================] - 0s 2ms/step - loss: 21505099776.0000 - val_loss: 22398056448.0000
    Epoch 4/400
    117/117 [==============================] - 0s 2ms/step - loss: 21330776064.0000 - val_loss: 22338795520.0000
    Epoch 5/400
    117/117 [==============================] - 0s 2ms/step - loss: 21221902336.0000 - val_loss: 22094151680.0000
    .
    .
    .
    Epoch 400/400
    117/117 [==============================] - 0s 2ms/step - loss: 13105951744.0000 - val_loss: 13929935872.0000




```python
losses = pd.DataFrame(model.history.history)
losses.plot()
```

<img src="{{ site.url }}{{ site.baseurl }}/images/House Predictions/plot4_keras.png">


_The new model shows some spikes in the val_loss which means its overfitting at a few places. We'll ignore this for now._


```python
predictions = model.predict(Xtest)
```


```python
print("Mean_absolute_error:", mean_absolute_error(ytest,predictions))
print("Mean_squared_error:",  mean_squared_error(ytest,predictions))
print("Root_Mean_Squared_error:", np.sqrt(mean_squared_error(ytest,predictions)))
data['price'].describe()['mean']
```

    Mean_absolute_error: 79021.48011009353
    Mean_squared_error: 13929933365.425364
    Root_Mean_Squared_error: 118025.1387011486





    540296.5735055795




```python
explained_variance_score(ytest,predictions)
```




    0.8321614513919094




```python
plt.figure(figsize=(12,6))
plt.scatter(ytest,predictions)
# Perfect predictions
plt.plot(ytest,ytest,'r')
```

<img src="{{ site.url }}{{ site.baseurl }}/images/House Predictions/scatter7_keras.png">



### Result: We were able to create a model and predict house prices post feature engineering and were also able to improve the error rate from 19% to 14.6%.  
#### Our model is not great at predicting the outliers but can moslty predict values in the most common range.  
#### The ability to explain variance has also increased from 79% to 83%.
