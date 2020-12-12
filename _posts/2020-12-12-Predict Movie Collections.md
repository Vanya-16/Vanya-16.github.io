---
title: "Predict Movie Collections"
date: 2020-12-12
tags: [data science, regression, machine learning algorithms, lasso, ridge, linear regression]
excerpt: "Data Science, machine learning algorithms, lasso, Ridge, Linear Regression"
mathjax: "true"
---

### Objective: Given a dataset of movies, train a model to predict the collection of the movies once released. Also, we would compare Linear, Ridge and Lasso Regressions to determine which one is best suited here.
Data used in the below analysis: [link](https://github.com/Vanya-16/DataSets/blob/master/Movie_collection_test.csv).

```python
#importing libraries required
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
#importing the dataset
movies_data = pd.read_csv('Data Files/Movie_collection_test.csv')
```


```python
#Quick look at the data
movies_data.head(5)
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
      <th>Collection</th>
      <th>Marketin_expense</th>
      <th>Production_expense</th>
      <th>Multiplex_coverage</th>
      <th>Budget</th>
      <th>Movie_length</th>
      <th>Lead_ Actor_Rating</th>
      <th>Lead_Actress_rating</th>
      <th>Director_rating</th>
      <th>Producer_rating</th>
      <th>Critic_rating</th>
      <th>Trailer_views</th>
      <th>Time_taken</th>
      <th>Twitter_hastags</th>
      <th>Genre</th>
      <th>Avg_age_actors</th>
      <th>MPAA_film_rating</th>
      <th>Num_multiplex</th>
      <th>3D_available</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11200</td>
      <td>520.9220</td>
      <td>91.2</td>
      <td>0.307</td>
      <td>33257.785</td>
      <td>173.5</td>
      <td>9.135</td>
      <td>9.31</td>
      <td>9.040</td>
      <td>9.335</td>
      <td>7.96</td>
      <td>308973</td>
      <td>184.24</td>
      <td>220.896</td>
      <td>Drama</td>
      <td>30</td>
      <td>PG</td>
      <td>618</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14400</td>
      <td>304.7240</td>
      <td>91.2</td>
      <td>0.307</td>
      <td>35235.365</td>
      <td>173.5</td>
      <td>9.120</td>
      <td>9.33</td>
      <td>9.095</td>
      <td>9.305</td>
      <td>7.96</td>
      <td>374897</td>
      <td>146.88</td>
      <td>201.152</td>
      <td>Comedy</td>
      <td>50</td>
      <td>PG</td>
      <td>703</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>2</th>
      <td>24200</td>
      <td>211.9142</td>
      <td>91.2</td>
      <td>0.307</td>
      <td>35574.220</td>
      <td>173.5</td>
      <td>9.170</td>
      <td>9.32</td>
      <td>9.115</td>
      <td>9.120</td>
      <td>7.96</td>
      <td>359036</td>
      <td>108.84</td>
      <td>281.936</td>
      <td>Thriller</td>
      <td>42</td>
      <td>PG</td>
      <td>689</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16600</td>
      <td>516.0340</td>
      <td>91.2</td>
      <td>0.307</td>
      <td>29713.695</td>
      <td>169.5</td>
      <td>9.125</td>
      <td>9.31</td>
      <td>9.060</td>
      <td>9.100</td>
      <td>6.96</td>
      <td>384237</td>
      <td>NaN</td>
      <td>301.328</td>
      <td>Thriller</td>
      <td>40</td>
      <td>PG</td>
      <td>677</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17000</td>
      <td>850.5840</td>
      <td>91.2</td>
      <td>0.307</td>
      <td>30724.705</td>
      <td>158.9</td>
      <td>9.050</td>
      <td>9.22</td>
      <td>9.185</td>
      <td>9.330</td>
      <td>7.96</td>
      <td>312011</td>
      <td>169.40</td>
      <td>221.360</td>
      <td>Comedy</td>
      <td>56</td>
      <td>PG</td>
      <td>615</td>
      <td>NO</td>
    </tr>
  </tbody>
</table>
</div>




```python
movies_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 506 entries, 0 to 505
    Data columns (total 19 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   Collection           506 non-null    int64  
     1   Marketin_expense     506 non-null    float64
     2   Production_expense   506 non-null    float64
     3   Multiplex_coverage   506 non-null    float64
     4   Budget               506 non-null    float64
     5   Movie_length         506 non-null    float64
     6   Lead_ Actor_Rating   506 non-null    float64
     7   Lead_Actress_rating  506 non-null    float64
     8   Director_rating      506 non-null    float64
     9   Producer_rating      506 non-null    float64
     10  Critic_rating        506 non-null    float64
     11  Trailer_views        506 non-null    int64  
     12  Time_taken           494 non-null    float64
     13  Twitter_hastags      506 non-null    float64
     14  Genre                506 non-null    object
     15  Avg_age_actors       506 non-null    int64  
     16  MPAA_film_rating     506 non-null    object
     17  Num_multiplex        506 non-null    int64  
     18  3D_available         506 non-null    object
    dtypes: float64(12), int64(4), object(3)
    memory usage: 75.2+ KB


_There are missing values in the Time_Taken field._


```python
movies_data.describe()
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
      <th>Collection</th>
      <th>Marketin_expense</th>
      <th>Production_expense</th>
      <th>Multiplex_coverage</th>
      <th>Budget</th>
      <th>Movie_length</th>
      <th>Lead_ Actor_Rating</th>
      <th>Lead_Actress_rating</th>
      <th>Director_rating</th>
      <th>Producer_rating</th>
      <th>Critic_rating</th>
      <th>Trailer_views</th>
      <th>Time_taken</th>
      <th>Twitter_hastags</th>
      <th>Avg_age_actors</th>
      <th>Num_multiplex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>494.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>45057.707510</td>
      <td>92.270471</td>
      <td>77.273557</td>
      <td>0.445305</td>
      <td>34911.144022</td>
      <td>142.074901</td>
      <td>8.014002</td>
      <td>8.185613</td>
      <td>8.019664</td>
      <td>8.190514</td>
      <td>7.810870</td>
      <td>449860.715415</td>
      <td>157.391498</td>
      <td>260.832095</td>
      <td>39.181818</td>
      <td>545.043478</td>
    </tr>
    <tr>
      <th>std</th>
      <td>18364.351764</td>
      <td>172.030902</td>
      <td>13.720706</td>
      <td>0.115878</td>
      <td>3903.038232</td>
      <td>28.148861</td>
      <td>1.054266</td>
      <td>1.054290</td>
      <td>1.059899</td>
      <td>1.049601</td>
      <td>0.659699</td>
      <td>68917.763145</td>
      <td>31.295161</td>
      <td>104.779133</td>
      <td>12.513697</td>
      <td>106.332889</td>
    </tr>
    <tr>
      <th>min</th>
      <td>10000.000000</td>
      <td>20.126400</td>
      <td>55.920000</td>
      <td>0.129000</td>
      <td>19781.355000</td>
      <td>76.400000</td>
      <td>3.840000</td>
      <td>4.035000</td>
      <td>3.840000</td>
      <td>4.030000</td>
      <td>6.600000</td>
      <td>212912.000000</td>
      <td>0.000000</td>
      <td>201.152000</td>
      <td>3.000000</td>
      <td>333.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>34050.000000</td>
      <td>21.640900</td>
      <td>65.380000</td>
      <td>0.376000</td>
      <td>32693.952500</td>
      <td>118.525000</td>
      <td>7.316250</td>
      <td>7.503750</td>
      <td>7.296250</td>
      <td>7.507500</td>
      <td>7.200000</td>
      <td>409128.000000</td>
      <td>132.300000</td>
      <td>223.796000</td>
      <td>28.000000</td>
      <td>465.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>42400.000000</td>
      <td>25.130200</td>
      <td>74.380000</td>
      <td>0.462000</td>
      <td>34488.217500</td>
      <td>151.000000</td>
      <td>8.307500</td>
      <td>8.495000</td>
      <td>8.312500</td>
      <td>8.465000</td>
      <td>7.960000</td>
      <td>462460.000000</td>
      <td>160.000000</td>
      <td>254.400000</td>
      <td>39.000000</td>
      <td>535.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>50000.000000</td>
      <td>93.541650</td>
      <td>91.200000</td>
      <td>0.551000</td>
      <td>36793.542500</td>
      <td>167.575000</td>
      <td>8.865000</td>
      <td>9.030000</td>
      <td>8.883750</td>
      <td>9.030000</td>
      <td>8.260000</td>
      <td>500247.500000</td>
      <td>181.890000</td>
      <td>283.416000</td>
      <td>50.000000</td>
      <td>614.750000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>100000.000000</td>
      <td>1799.524000</td>
      <td>110.480000</td>
      <td>0.615000</td>
      <td>48772.900000</td>
      <td>173.500000</td>
      <td>9.435000</td>
      <td>9.540000</td>
      <td>9.425000</td>
      <td>9.635000</td>
      <td>9.400000</td>
      <td>567784.000000</td>
      <td>217.520000</td>
      <td>2022.400000</td>
      <td>60.000000</td>
      <td>868.000000</td>
    </tr>
  </tbody>
</table>
</div>



_Marketin_Experience and Bugdet need another look as their mean, median and max values are expanding over a huge range.  _
### EDA!


```python
sns.jointplot(x='Marketin_expense',y='Collection',data=movies_data)
```

<img src="{{ site.url }}{{ site.baseurl }}/images/Predict Movie Collection/Linear Regression and others_8_1.png">



_Outliers are present which need to be treated as a part of pre-processing._


```python

sns.jointplot(x='Budget',y='Collection',data=movies_data)
```



<img src="{{ site.url }}{{ site.baseurl }}/images/Predict Movie Collection/Linear Regression and others_10_1.png">


_Budget seems to be fine as we'll use it as is.  
Let's check our categorical data._


```python
sns.countplot(x='Genre',data=movies_data)
```



<img src="{{ site.url }}{{ site.baseurl }}/images/Predict Movie Collection/Linear Regression and others_12_1.png">



```python
sns.countplot(x='3D_available',data=movies_data)
```

<img src="{{ site.url }}{{ site.baseurl }}/images/Predict Movie Collection/Linear Regression and others_13_1.png">



```python
sns.countplot(x='MPAA_film_rating',data=movies_data)
```


<img src="{{ site.url }}{{ site.baseurl }}/images/Predict Movie Collection/Linear Regression and others_14_1.png">


_Our categorical seems fine to use except MPAA_film_rating. As it has only one value it won't affect our model in any way. We can drop it._


```python
movies_data.drop('MPAA_film_rating',axis=1, inplace=True)
```

### Treating the outliers  
_We would use capping to treat the higher values in Marketin_Expense._


```python
#checking the min and max value
movies_data['Marketin_expense'].min()
```




    20.1264




```python
movies_data['Marketin_expense'].max()
```




    1799.524




```python
#Capping the numbers above 1.5 times the 99 percentile
ul = np.percentile(movies_data['Marketin_expense'],[99])[0]
movies_data[movies_data['Marketin_expense'] > 1.5*ul]
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
      <th>Collection</th>
      <th>Marketin_expense</th>
      <th>Production_expense</th>
      <th>Multiplex_coverage</th>
      <th>Budget</th>
      <th>Movie_length</th>
      <th>Lead_ Actor_Rating</th>
      <th>Lead_Actress_rating</th>
      <th>Director_rating</th>
      <th>Producer_rating</th>
      <th>Critic_rating</th>
      <th>Trailer_views</th>
      <th>Time_taken</th>
      <th>Twitter_hastags</th>
      <th>Genre</th>
      <th>Avg_age_actors</th>
      <th>MPAA_film_rating</th>
      <th>Num_multiplex</th>
      <th>3D_available</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>10000</td>
      <td>1378.416</td>
      <td>91.2</td>
      <td>0.307</td>
      <td>31569.065</td>
      <td>173.5</td>
      <td>9.235</td>
      <td>9.405</td>
      <td>9.280</td>
      <td>9.23</td>
      <td>6.96</td>
      <td>342621</td>
      <td>146.00</td>
      <td>280.800</td>
      <td>Thriller</td>
      <td>38</td>
      <td>PG</td>
      <td>654</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>18</th>
      <td>17600</td>
      <td>1490.682</td>
      <td>91.2</td>
      <td>0.321</td>
      <td>33091.135</td>
      <td>173.5</td>
      <td>9.020</td>
      <td>9.155</td>
      <td>9.075</td>
      <td>9.15</td>
      <td>7.96</td>
      <td>383325</td>
      <td>169.52</td>
      <td>241.408</td>
      <td>Comedy</td>
      <td>52</td>
      <td>PG</td>
      <td>680</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>486</th>
      <td>20800</td>
      <td>1799.524</td>
      <td>91.2</td>
      <td>0.329</td>
      <td>38707.240</td>
      <td>165.4</td>
      <td>9.170</td>
      <td>9.430</td>
      <td>9.155</td>
      <td>9.41</td>
      <td>6.96</td>
      <td>417588</td>
      <td>188.16</td>
      <td>281.664</td>
      <td>Comedy</td>
      <td>21</td>
      <td>PG</td>
      <td>666</td>
      <td>YES</td>
    </tr>
  </tbody>
</table>
</div>




```python
movies_data.Marketin_expense[movies_data['Marketin_expense'] > 1.5*ul] = 1.5*ul
movies_data[movies_data['Marketin_expense'] > ul]
```

    <ipython-input-23-2002ed2d8b50>:1: SettingWithCopyWarning:
    A value is trying to be set on a copy of a slice from a DataFrame

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      movies_data.Marketin_expense[movies_data['Marketin_expense'] > 1.5*ul] = 1.5*ul





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
      <th>Collection</th>
      <th>Marketin_expense</th>
      <th>Production_expense</th>
      <th>Multiplex_coverage</th>
      <th>Budget</th>
      <th>Movie_length</th>
      <th>Lead_ Actor_Rating</th>
      <th>Lead_Actress_rating</th>
      <th>Director_rating</th>
      <th>Producer_rating</th>
      <th>Critic_rating</th>
      <th>Trailer_views</th>
      <th>Time_taken</th>
      <th>Twitter_hastags</th>
      <th>Genre</th>
      <th>Avg_age_actors</th>
      <th>MPAA_film_rating</th>
      <th>Num_multiplex</th>
      <th>3D_available</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>17000</td>
      <td>850.5840</td>
      <td>91.2</td>
      <td>0.307</td>
      <td>30724.705</td>
      <td>158.9</td>
      <td>9.050</td>
      <td>9.220</td>
      <td>9.185</td>
      <td>9.330</td>
      <td>7.96</td>
      <td>312011</td>
      <td>169.40</td>
      <td>221.360</td>
      <td>Comedy</td>
      <td>56</td>
      <td>PG</td>
      <td>615</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>5</th>
      <td>10000</td>
      <td>1271.1099</td>
      <td>91.2</td>
      <td>0.307</td>
      <td>31569.065</td>
      <td>173.5</td>
      <td>9.235</td>
      <td>9.405</td>
      <td>9.280</td>
      <td>9.230</td>
      <td>6.96</td>
      <td>342621</td>
      <td>146.00</td>
      <td>280.800</td>
      <td>Thriller</td>
      <td>38</td>
      <td>PG</td>
      <td>654</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>10</th>
      <td>30000</td>
      <td>1042.7160</td>
      <td>91.2</td>
      <td>0.403</td>
      <td>31980.135</td>
      <td>173.5</td>
      <td>9.155</td>
      <td>9.340</td>
      <td>9.210</td>
      <td>9.470</td>
      <td>6.96</td>
      <td>474055</td>
      <td>192.00</td>
      <td>222.400</td>
      <td>Thriller</td>
      <td>52</td>
      <td>PG</td>
      <td>617</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14000</td>
      <td>934.9220</td>
      <td>91.2</td>
      <td>0.307</td>
      <td>25103.045</td>
      <td>173.5</td>
      <td>9.130</td>
      <td>9.250</td>
      <td>9.050</td>
      <td>9.255</td>
      <td>7.96</td>
      <td>212912</td>
      <td>120.80</td>
      <td>241.120</td>
      <td>Thriller</td>
      <td>40</td>
      <td>PG</td>
      <td>693</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>18</th>
      <td>17600</td>
      <td>1271.1099</td>
      <td>91.2</td>
      <td>0.321</td>
      <td>33091.135</td>
      <td>173.5</td>
      <td>9.020</td>
      <td>9.155</td>
      <td>9.075</td>
      <td>9.150</td>
      <td>7.96</td>
      <td>383325</td>
      <td>169.52</td>
      <td>241.408</td>
      <td>Comedy</td>
      <td>52</td>
      <td>PG</td>
      <td>680</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>486</th>
      <td>20800</td>
      <td>1271.1099</td>
      <td>91.2</td>
      <td>0.329</td>
      <td>38707.240</td>
      <td>165.4</td>
      <td>9.170</td>
      <td>9.430</td>
      <td>9.155</td>
      <td>9.410</td>
      <td>6.96</td>
      <td>417588</td>
      <td>188.16</td>
      <td>281.664</td>
      <td>Comedy</td>
      <td>21</td>
      <td>PG</td>
      <td>666</td>
      <td>YES</td>
    </tr>
  </tbody>
</table>
</div>



### Treating the missing data


```python
movies_data.Time_taken.mean()
```




    157.39149797570857




```python
#missing data
movies_data[movies_data['Time_taken'].isnull()]
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
      <th>Collection</th>
      <th>Marketin_expense</th>
      <th>Production_expense</th>
      <th>Multiplex_coverage</th>
      <th>Budget</th>
      <th>Movie_length</th>
      <th>Lead_ Actor_Rating</th>
      <th>Lead_Actress_rating</th>
      <th>Director_rating</th>
      <th>Producer_rating</th>
      <th>Critic_rating</th>
      <th>Trailer_views</th>
      <th>Time_taken</th>
      <th>Twitter_hastags</th>
      <th>Genre</th>
      <th>Avg_age_actors</th>
      <th>MPAA_film_rating</th>
      <th>Num_multiplex</th>
      <th>3D_available</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>16600</td>
      <td>516.0340</td>
      <td>91.20</td>
      <td>0.307</td>
      <td>29713.695</td>
      <td>169.5</td>
      <td>9.125</td>
      <td>9.310</td>
      <td>9.060</td>
      <td>9.100</td>
      <td>6.96</td>
      <td>384237</td>
      <td>NaN</td>
      <td>301.328</td>
      <td>Thriller</td>
      <td>40</td>
      <td>PG</td>
      <td>677</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>16</th>
      <td>15000</td>
      <td>236.6840</td>
      <td>91.20</td>
      <td>0.321</td>
      <td>37674.010</td>
      <td>164.3</td>
      <td>9.050</td>
      <td>9.230</td>
      <td>8.980</td>
      <td>9.100</td>
      <td>7.96</td>
      <td>335532</td>
      <td>NaN</td>
      <td>201.200</td>
      <td>Thriller</td>
      <td>35</td>
      <td>PG</td>
      <td>647</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>40</th>
      <td>21000</td>
      <td>461.0220</td>
      <td>91.20</td>
      <td>0.260</td>
      <td>32318.990</td>
      <td>165.9</td>
      <td>8.985</td>
      <td>9.170</td>
      <td>9.020</td>
      <td>9.095</td>
      <td>7.96</td>
      <td>360183</td>
      <td>NaN</td>
      <td>241.680</td>
      <td>Comedy</td>
      <td>38</td>
      <td>PG</td>
      <td>753</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>96</th>
      <td>39400</td>
      <td>25.7920</td>
      <td>74.38</td>
      <td>0.415</td>
      <td>29941.450</td>
      <td>146.4</td>
      <td>8.570</td>
      <td>8.695</td>
      <td>8.510</td>
      <td>8.630</td>
      <td>7.16</td>
      <td>380129</td>
      <td>NaN</td>
      <td>243.152</td>
      <td>Thriller</td>
      <td>44</td>
      <td>PG</td>
      <td>611</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>126</th>
      <td>27200</td>
      <td>45.0358</td>
      <td>71.28</td>
      <td>0.462</td>
      <td>30941.350</td>
      <td>171.6</td>
      <td>8.035</td>
      <td>8.205</td>
      <td>7.955</td>
      <td>8.210</td>
      <td>7.80</td>
      <td>371051</td>
      <td>NaN</td>
      <td>302.176</td>
      <td>Action</td>
      <td>44</td>
      <td>PG</td>
      <td>484</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>164</th>
      <td>46600</td>
      <td>23.0890</td>
      <td>65.26</td>
      <td>0.547</td>
      <td>34135.475</td>
      <td>102.7</td>
      <td>6.010</td>
      <td>6.115</td>
      <td>5.965</td>
      <td>6.280</td>
      <td>7.06</td>
      <td>480067</td>
      <td>NaN</td>
      <td>283.728</td>
      <td>Comedy</td>
      <td>22</td>
      <td>PG</td>
      <td>438</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>166</th>
      <td>37400</td>
      <td>22.9864</td>
      <td>65.26</td>
      <td>0.547</td>
      <td>31891.255</td>
      <td>139.7</td>
      <td>6.335</td>
      <td>6.420</td>
      <td>6.235</td>
      <td>6.560</td>
      <td>7.06</td>
      <td>465689</td>
      <td>NaN</td>
      <td>222.992</td>
      <td>Thriller</td>
      <td>30</td>
      <td>PG</td>
      <td>439</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>210</th>
      <td>40200</td>
      <td>22.7920</td>
      <td>72.12</td>
      <td>0.480</td>
      <td>34257.685</td>
      <td>163.5</td>
      <td>8.685</td>
      <td>8.875</td>
      <td>8.660</td>
      <td>8.935</td>
      <td>6.82</td>
      <td>432081</td>
      <td>NaN</td>
      <td>203.216</td>
      <td>Comedy</td>
      <td>20</td>
      <td>PG</td>
      <td>458</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>211</th>
      <td>39000</td>
      <td>22.6524</td>
      <td>72.12</td>
      <td>0.480</td>
      <td>32502.305</td>
      <td>170.2</td>
      <td>8.905</td>
      <td>9.025</td>
      <td>8.935</td>
      <td>8.925</td>
      <td>6.82</td>
      <td>430817</td>
      <td>NaN</td>
      <td>263.120</td>
      <td>Comedy</td>
      <td>57</td>
      <td>PG</td>
      <td>515</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>321</th>
      <td>50000</td>
      <td>23.9604</td>
      <td>76.18</td>
      <td>0.511</td>
      <td>34341.010</td>
      <td>115.9</td>
      <td>7.925</td>
      <td>8.095</td>
      <td>8.020</td>
      <td>8.065</td>
      <td>7.28</td>
      <td>456943</td>
      <td>NaN</td>
      <td>244.000</td>
      <td>Drama</td>
      <td>30</td>
      <td>PG</td>
      <td>480</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>366</th>
      <td>67600</td>
      <td>30.8022</td>
      <td>62.94</td>
      <td>0.353</td>
      <td>40012.665</td>
      <td>155.3</td>
      <td>8.940</td>
      <td>9.025</td>
      <td>8.815</td>
      <td>8.995</td>
      <td>9.40</td>
      <td>483080</td>
      <td>NaN</td>
      <td>225.408</td>
      <td>Drama</td>
      <td>21</td>
      <td>PG</td>
      <td>681</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>465</th>
      <td>45200</td>
      <td>105.2262</td>
      <td>91.20</td>
      <td>0.230</td>
      <td>33952.160</td>
      <td>154.8</td>
      <td>8.610</td>
      <td>8.810</td>
      <td>8.720</td>
      <td>8.845</td>
      <td>6.96</td>
      <td>437945</td>
      <td>NaN</td>
      <td>283.616</td>
      <td>Drama</td>
      <td>26</td>
      <td>PG</td>
      <td>743</td>
      <td>NO</td>
    </tr>
  </tbody>
</table>
</div>




```python
#updating 12 missing values with the mean value
movies_data.Time_taken = movies_data.Time_taken.fillna(movies_data.Time_taken.mean())
```


```python
movies_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 506 entries, 0 to 505
    Data columns (total 19 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   Collection           506 non-null    int64  
     1   Marketin_expense     506 non-null    float64
     2   Production_expense   506 non-null    float64
     3   Multiplex_coverage   506 non-null    float64
     4   Budget               506 non-null    float64
     5   Movie_length         506 non-null    float64
     6   Lead_ Actor_Rating   506 non-null    float64
     7   Lead_Actress_rating  506 non-null    float64
     8   Director_rating      506 non-null    float64
     9   Producer_rating      506 non-null    float64
     10  Critic_rating        506 non-null    float64
     11  Trailer_views        506 non-null    int64  
     12  Time_taken           506 non-null    float64
     13  Twitter_hastags      506 non-null    float64
     14  Genre                506 non-null    object
     15  Avg_age_actors       506 non-null    int64  
     16  MPAA_film_rating     506 non-null    object
     17  Num_multiplex        506 non-null    int64  
     18  3D_available         506 non-null    object
    dtypes: float64(12), int64(4), object(3)
    memory usage: 75.2+ KB


### Variable Transformation  
_This is not a mandatory step but we do it in hopes to get better result!_


```python
sns.pairplot(data=movies_data)
```

<img src="{{ site.url }}{{ site.baseurl }}/images/Predict Movie Collection/Linear Regression and others_28_1.png">


_Marketin_expense and Trailer_views seem to have a non-linear relationship with Collection, let's explore them further!_


```python
sns.jointplot(x='Marketin_expense', y='Collection', data=movies_data)
```


<img src="{{ site.url }}{{ site.baseurl }}/images/Predict Movie Collection/Linear Regression and others_30_1.png">

_It seems like a log relationship but its not very strong so we would ignore it._


```python
sns.jointplot(x='Trailer_views', y='Collection', data=movies_data)
```


<img src="{{ site.url }}{{ site.baseurl }}/images/Predict Movie Collection/Linear Regression and others_32_1.png">


_This is an exp relationship, let's convert it into a linear one for the ease of our model._


```python
movies_data.Trailer_views = np.exp(movies_data.Trailer_views/100000)
sns.jointplot(x='Trailer_views', y='Collection', data=movies_data)
```


<img src="{{ site.url }}{{ site.baseurl }}/images/Predict Movie Collection/Linear Regression and others_34_1.png">

_Now we have a more linear relationship between Trailer_views and collection._

### Converting Categorical Data into dummy variables


```python
feat = ['Genre','3D_available']
movies_data = pd.get_dummies(data=movies_data,columns=feat,drop_first=True)
```


```python
movies_data.head(5)
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
      <th>Collection</th>
      <th>Marketin_expense</th>
      <th>Production_expense</th>
      <th>Multiplex_coverage</th>
      <th>Budget</th>
      <th>Movie_length</th>
      <th>Lead_ Actor_Rating</th>
      <th>Lead_Actress_rating</th>
      <th>Director_rating</th>
      <th>Producer_rating</th>
      <th>Critic_rating</th>
      <th>Trailer_views</th>
      <th>Time_taken</th>
      <th>Twitter_hastags</th>
      <th>Avg_age_actors</th>
      <th>Num_multiplex</th>
      <th>Genre_Comedy</th>
      <th>Genre_Drama</th>
      <th>Genre_Thriller</th>
      <th>3D_available_YES</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11200</td>
      <td>520.9220</td>
      <td>91.2</td>
      <td>0.307</td>
      <td>33257.785</td>
      <td>173.5</td>
      <td>9.135</td>
      <td>9.31</td>
      <td>9.040</td>
      <td>9.335</td>
      <td>7.96</td>
      <td>21.971145</td>
      <td>184.240000</td>
      <td>220.896</td>
      <td>30</td>
      <td>618</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14400</td>
      <td>304.7240</td>
      <td>91.2</td>
      <td>0.307</td>
      <td>35235.365</td>
      <td>173.5</td>
      <td>9.120</td>
      <td>9.33</td>
      <td>9.095</td>
      <td>9.305</td>
      <td>7.96</td>
      <td>42.477308</td>
      <td>146.880000</td>
      <td>201.152</td>
      <td>50</td>
      <td>703</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>24200</td>
      <td>211.9142</td>
      <td>91.2</td>
      <td>0.307</td>
      <td>35574.220</td>
      <td>173.5</td>
      <td>9.170</td>
      <td>9.32</td>
      <td>9.115</td>
      <td>9.120</td>
      <td>7.96</td>
      <td>36.247123</td>
      <td>108.840000</td>
      <td>281.936</td>
      <td>42</td>
      <td>689</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16600</td>
      <td>516.0340</td>
      <td>91.2</td>
      <td>0.307</td>
      <td>29713.695</td>
      <td>169.5</td>
      <td>9.125</td>
      <td>9.31</td>
      <td>9.060</td>
      <td>9.100</td>
      <td>6.96</td>
      <td>46.635871</td>
      <td>157.391498</td>
      <td>301.328</td>
      <td>40</td>
      <td>677</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17000</td>
      <td>850.5840</td>
      <td>91.2</td>
      <td>0.307</td>
      <td>30724.705</td>
      <td>158.9</td>
      <td>9.050</td>
      <td>9.22</td>
      <td>9.185</td>
      <td>9.330</td>
      <td>7.96</td>
      <td>22.648871</td>
      <td>169.400000</td>
      <td>221.360</td>
      <td>56</td>
      <td>615</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
movies_data.corr()
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
      <th>Collection</th>
      <th>Marketin_expense</th>
      <th>Production_expense</th>
      <th>Multiplex_coverage</th>
      <th>Budget</th>
      <th>Movie_length</th>
      <th>Lead_ Actor_Rating</th>
      <th>Lead_Actress_rating</th>
      <th>Director_rating</th>
      <th>Producer_rating</th>
      <th>Critic_rating</th>
      <th>Trailer_views</th>
      <th>Time_taken</th>
      <th>Twitter_hastags</th>
      <th>Avg_age_actors</th>
      <th>Num_multiplex</th>
      <th>Genre_Comedy</th>
      <th>Genre_Drama</th>
      <th>Genre_Thriller</th>
      <th>3D_available_YES</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Collection</th>
      <td>1.000000</td>
      <td>-0.409048</td>
      <td>-0.484754</td>
      <td>0.429300</td>
      <td>0.696304</td>
      <td>-0.377999</td>
      <td>-0.251355</td>
      <td>-0.249459</td>
      <td>-0.246650</td>
      <td>-0.248200</td>
      <td>0.341288</td>
      <td>0.765323</td>
      <td>0.110005</td>
      <td>0.023122</td>
      <td>-0.047426</td>
      <td>-0.391729</td>
      <td>-0.077478</td>
      <td>0.036233</td>
      <td>0.071751</td>
      <td>0.182867</td>
    </tr>
    <tr>
      <th>Marketin_expense</th>
      <td>-0.409048</td>
      <td>1.000000</td>
      <td>0.432125</td>
      <td>-0.447478</td>
      <td>-0.242900</td>
      <td>0.374271</td>
      <td>0.402649</td>
      <td>0.401933</td>
      <td>0.402682</td>
      <td>0.398642</td>
      <td>-0.191898</td>
      <td>-0.395998</td>
      <td>0.020817</td>
      <td>0.013665</td>
      <td>0.071444</td>
      <td>0.405228</td>
      <td>0.059571</td>
      <td>-0.013189</td>
      <td>-0.035181</td>
      <td>-0.098717</td>
    </tr>
    <tr>
      <th>Production_expense</th>
      <td>-0.484754</td>
      <td>0.432125</td>
      <td>1.000000</td>
      <td>-0.763651</td>
      <td>-0.391676</td>
      <td>0.644779</td>
      <td>0.706481</td>
      <td>0.707956</td>
      <td>0.707566</td>
      <td>0.705819</td>
      <td>-0.251565</td>
      <td>-0.589393</td>
      <td>0.015773</td>
      <td>-0.000839</td>
      <td>0.055810</td>
      <td>0.707559</td>
      <td>0.086958</td>
      <td>-0.026590</td>
      <td>-0.098976</td>
      <td>-0.115401</td>
    </tr>
    <tr>
      <th>Multiplex_coverage</th>
      <td>0.429300</td>
      <td>-0.447478</td>
      <td>-0.763651</td>
      <td>1.000000</td>
      <td>0.302188</td>
      <td>-0.731470</td>
      <td>-0.768589</td>
      <td>-0.769724</td>
      <td>-0.769157</td>
      <td>-0.764873</td>
      <td>0.145555</td>
      <td>0.565641</td>
      <td>0.035515</td>
      <td>0.004882</td>
      <td>-0.092104</td>
      <td>-0.915495</td>
      <td>-0.068554</td>
      <td>0.046393</td>
      <td>0.037772</td>
      <td>0.073903</td>
    </tr>
    <tr>
      <th>Budget</th>
      <td>0.696304</td>
      <td>-0.242900</td>
      <td>-0.391676</td>
      <td>0.302188</td>
      <td>1.000000</td>
      <td>-0.240265</td>
      <td>-0.208464</td>
      <td>-0.203981</td>
      <td>-0.201907</td>
      <td>-0.205397</td>
      <td>0.232361</td>
      <td>0.621862</td>
      <td>0.040439</td>
      <td>0.030674</td>
      <td>-0.064694</td>
      <td>-0.282796</td>
      <td>-0.052579</td>
      <td>-0.004195</td>
      <td>0.046251</td>
      <td>0.163774</td>
    </tr>
    <tr>
      <th>Movie_length</th>
      <td>-0.377999</td>
      <td>0.374271</td>
      <td>0.644779</td>
      <td>-0.731470</td>
      <td>-0.240265</td>
      <td>1.000000</td>
      <td>0.746904</td>
      <td>0.746493</td>
      <td>0.747021</td>
      <td>0.746707</td>
      <td>-0.217830</td>
      <td>-0.597070</td>
      <td>-0.019820</td>
      <td>0.009380</td>
      <td>0.075198</td>
      <td>0.673896</td>
      <td>0.092693</td>
      <td>0.003452</td>
      <td>-0.088609</td>
      <td>0.005101</td>
    </tr>
    <tr>
      <th>Lead_ Actor_Rating</th>
      <td>-0.251355</td>
      <td>0.402649</td>
      <td>0.706481</td>
      <td>-0.768589</td>
      <td>-0.208464</td>
      <td>0.746904</td>
      <td>1.000000</td>
      <td>0.997905</td>
      <td>0.997735</td>
      <td>0.994073</td>
      <td>-0.169978</td>
      <td>-0.472630</td>
      <td>0.038050</td>
      <td>0.014463</td>
      <td>0.036794</td>
      <td>0.706331</td>
      <td>0.044592</td>
      <td>-0.035171</td>
      <td>-0.030763</td>
      <td>-0.025208</td>
    </tr>
    <tr>
      <th>Lead_Actress_rating</th>
      <td>-0.249459</td>
      <td>0.401933</td>
      <td>0.707956</td>
      <td>-0.769724</td>
      <td>-0.203981</td>
      <td>0.746493</td>
      <td>0.997905</td>
      <td>1.000000</td>
      <td>0.998097</td>
      <td>0.994003</td>
      <td>-0.165992</td>
      <td>-0.471097</td>
      <td>0.037975</td>
      <td>0.010239</td>
      <td>0.038005</td>
      <td>0.708257</td>
      <td>0.046974</td>
      <td>-0.038965</td>
      <td>-0.030566</td>
      <td>-0.020056</td>
    </tr>
    <tr>
      <th>Director_rating</th>
      <td>-0.246650</td>
      <td>0.402682</td>
      <td>0.707566</td>
      <td>-0.769157</td>
      <td>-0.201907</td>
      <td>0.747021</td>
      <td>0.997735</td>
      <td>0.998097</td>
      <td>1.000000</td>
      <td>0.994126</td>
      <td>-0.166638</td>
      <td>-0.468861</td>
      <td>0.035881</td>
      <td>0.010077</td>
      <td>0.041470</td>
      <td>0.709364</td>
      <td>0.046268</td>
      <td>-0.033510</td>
      <td>-0.033634</td>
      <td>-0.020195</td>
    </tr>
    <tr>
      <th>Producer_rating</th>
      <td>-0.248200</td>
      <td>0.398642</td>
      <td>0.705819</td>
      <td>-0.764873</td>
      <td>-0.205397</td>
      <td>0.746707</td>
      <td>0.994073</td>
      <td>0.994003</td>
      <td>0.994126</td>
      <td>1.000000</td>
      <td>-0.167003</td>
      <td>-0.471498</td>
      <td>0.028695</td>
      <td>0.005850</td>
      <td>0.032542</td>
      <td>0.703518</td>
      <td>0.051274</td>
      <td>-0.031696</td>
      <td>-0.033829</td>
      <td>-0.020022</td>
    </tr>
    <tr>
      <th>Critic_rating</th>
      <td>0.341288</td>
      <td>-0.191898</td>
      <td>-0.251565</td>
      <td>0.145555</td>
      <td>0.232361</td>
      <td>-0.217830</td>
      <td>-0.169978</td>
      <td>-0.165992</td>
      <td>-0.166638</td>
      <td>-0.167003</td>
      <td>1.000000</td>
      <td>0.273364</td>
      <td>-0.014762</td>
      <td>-0.023655</td>
      <td>-0.049797</td>
      <td>-0.128769</td>
      <td>-0.015253</td>
      <td>0.057177</td>
      <td>-0.037129</td>
      <td>0.039235</td>
    </tr>
    <tr>
      <th>Trailer_views</th>
      <td>0.765323</td>
      <td>-0.395998</td>
      <td>-0.589393</td>
      <td>0.565641</td>
      <td>0.621862</td>
      <td>-0.597070</td>
      <td>-0.472630</td>
      <td>-0.471097</td>
      <td>-0.468861</td>
      <td>-0.471498</td>
      <td>0.273364</td>
      <td>1.000000</td>
      <td>0.076065</td>
      <td>0.025024</td>
      <td>-0.039545</td>
      <td>-0.532687</td>
      <td>-0.109355</td>
      <td>0.010627</td>
      <td>0.117332</td>
      <td>0.093246</td>
    </tr>
    <tr>
      <th>Time_taken</th>
      <td>0.110005</td>
      <td>0.020817</td>
      <td>0.015773</td>
      <td>0.035515</td>
      <td>0.040439</td>
      <td>-0.019820</td>
      <td>0.038050</td>
      <td>0.037975</td>
      <td>0.035881</td>
      <td>0.028695</td>
      <td>-0.014762</td>
      <td>0.076065</td>
      <td>1.000000</td>
      <td>-0.006382</td>
      <td>0.072049</td>
      <td>-0.056704</td>
      <td>0.012908</td>
      <td>0.049285</td>
      <td>-0.098138</td>
      <td>-0.024431</td>
    </tr>
    <tr>
      <th>Twitter_hastags</th>
      <td>0.023122</td>
      <td>0.013665</td>
      <td>-0.000839</td>
      <td>0.004882</td>
      <td>0.030674</td>
      <td>0.009380</td>
      <td>0.014463</td>
      <td>0.010239</td>
      <td>0.010077</td>
      <td>0.005850</td>
      <td>-0.023655</td>
      <td>0.025024</td>
      <td>-0.006382</td>
      <td>1.000000</td>
      <td>-0.004840</td>
      <td>0.006255</td>
      <td>0.034407</td>
      <td>0.036442</td>
      <td>-0.058431</td>
      <td>-0.066012</td>
    </tr>
    <tr>
      <th>Avg_age_actors</th>
      <td>-0.047426</td>
      <td>0.071444</td>
      <td>0.055810</td>
      <td>-0.092104</td>
      <td>-0.064694</td>
      <td>0.075198</td>
      <td>0.036794</td>
      <td>0.038005</td>
      <td>0.041470</td>
      <td>0.032542</td>
      <td>-0.049797</td>
      <td>-0.039545</td>
      <td>0.072049</td>
      <td>-0.004840</td>
      <td>1.000000</td>
      <td>0.078811</td>
      <td>-0.030584</td>
      <td>-0.015918</td>
      <td>-0.036611</td>
      <td>-0.013581</td>
    </tr>
    <tr>
      <th>Num_multiplex</th>
      <td>-0.391729</td>
      <td>0.405228</td>
      <td>0.707559</td>
      <td>-0.915495</td>
      <td>-0.282796</td>
      <td>0.673896</td>
      <td>0.706331</td>
      <td>0.708257</td>
      <td>0.709364</td>
      <td>0.703518</td>
      <td>-0.128769</td>
      <td>-0.532687</td>
      <td>-0.056704</td>
      <td>0.006255</td>
      <td>0.078811</td>
      <td>1.000000</td>
      <td>0.070720</td>
      <td>-0.035126</td>
      <td>-0.048863</td>
      <td>-0.052262</td>
    </tr>
    <tr>
      <th>Genre_Comedy</th>
      <td>-0.077478</td>
      <td>0.059571</td>
      <td>0.086958</td>
      <td>-0.068554</td>
      <td>-0.052579</td>
      <td>0.092693</td>
      <td>0.044592</td>
      <td>0.046974</td>
      <td>0.046268</td>
      <td>0.051274</td>
      <td>-0.015253</td>
      <td>-0.109355</td>
      <td>0.012908</td>
      <td>0.034407</td>
      <td>-0.030584</td>
      <td>0.070720</td>
      <td>1.000000</td>
      <td>-0.323621</td>
      <td>-0.500192</td>
      <td>0.004617</td>
    </tr>
    <tr>
      <th>Genre_Drama</th>
      <td>0.036233</td>
      <td>-0.013189</td>
      <td>-0.026590</td>
      <td>0.046393</td>
      <td>-0.004195</td>
      <td>0.003452</td>
      <td>-0.035171</td>
      <td>-0.038965</td>
      <td>-0.033510</td>
      <td>-0.031696</td>
      <td>0.057177</td>
      <td>0.010627</td>
      <td>0.049285</td>
      <td>0.036442</td>
      <td>-0.015918</td>
      <td>-0.035126</td>
      <td>-0.323621</td>
      <td>1.000000</td>
      <td>-0.366563</td>
      <td>0.035491</td>
    </tr>
    <tr>
      <th>Genre_Thriller</th>
      <td>0.071751</td>
      <td>-0.035181</td>
      <td>-0.098976</td>
      <td>0.037772</td>
      <td>0.046251</td>
      <td>-0.088609</td>
      <td>-0.030763</td>
      <td>-0.030566</td>
      <td>-0.033634</td>
      <td>-0.033829</td>
      <td>-0.037129</td>
      <td>0.117332</td>
      <td>-0.098138</td>
      <td>-0.058431</td>
      <td>-0.036611</td>
      <td>-0.048863</td>
      <td>-0.500192</td>
      <td>-0.366563</td>
      <td>1.000000</td>
      <td>0.017341</td>
    </tr>
    <tr>
      <th>3D_available_YES</th>
      <td>0.182867</td>
      <td>-0.098717</td>
      <td>-0.115401</td>
      <td>0.073903</td>
      <td>0.163774</td>
      <td>0.005101</td>
      <td>-0.025208</td>
      <td>-0.020056</td>
      <td>-0.020195</td>
      <td>-0.020022</td>
      <td>0.039235</td>
      <td>0.093246</td>
      <td>-0.024431</td>
      <td>-0.066012</td>
      <td>-0.013581</td>
      <td>-0.052262</td>
      <td>0.004617</td>
      <td>0.035491</td>
      <td>0.017341</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(18,10))
sns.heatmap(movies_data.corr(),annot=True, cmap = 'viridis')
```
<img src="{{ site.url }}{{ site.baseurl }}/images/Predict Movie Collection/Linear Regression and others_40_1.png">


_Following seem to be highly correlated with each other indicating they are not truly independent variables:_  
 - _Num_multiplex and Multiplex coverage_  
 - _Lead_Actress_Rating and Lead_Actor_Rating_  
 - _Director_Rating and Lead_Actor_Rating_  
 - _Producer_Rating and Lead_Actor_Rating_  

_We need to remove one from each pair to avoid the issue of multi-collinearity._


```python
movies_data.corr()['Collection']
```




    Collection             1.000000
    Marketin_expense      -0.409048
    Production_expense    -0.484754
    Multiplex_coverage     0.429300
    Budget                 0.696304
    Movie_length          -0.377999
    Lead_ Actor_Rating    -0.251355
    Lead_Actress_rating   -0.249459
    Director_rating       -0.246650
    Producer_rating       -0.248200
    Critic_rating          0.341288
    Trailer_views          0.765323
    Time_taken             0.110005
    Twitter_hastags        0.023122
    Avg_age_actors        -0.047426
    Num_multiplex         -0.391729
    Genre_Comedy          -0.077478
    Genre_Drama            0.036233
    Genre_Thriller         0.071751
    3D_available_YES       0.182867
    Name: Collection, dtype: float64




```python
del movies_data['Num_multiplex']
del movies_data['Lead_Actress_rating']
del movies_data['Director_rating']
del movies_data['Producer_rating']
```

### Train Test Split


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(movies_data.drop('Collection',axis=1), movies_data['Collection'],
                                                    test_size=0.3, random_state=101)
```

### Train model - Linear regression, Ridge Regression and Lasso Regression


```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
```


```python
lm_linear = LinearRegression()
lm_linear.fit(X_train,y_train)
```




    LinearRegression()




```python
pd.DataFrame(lm_linear.coef_,movies_data.columns.drop('Collection'),columns=['Coefficients'])
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
      <th>Coefficients</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Marketin_expense</th>
      <td>-17.660152</td>
    </tr>
    <tr>
      <th>Production_expense</th>
      <td>-38.573837</td>
    </tr>
    <tr>
      <th>Multiplex_coverage</th>
      <td>21164.321821</td>
    </tr>
    <tr>
      <th>Budget</th>
      <td>1.592359</td>
    </tr>
    <tr>
      <th>Movie_length</th>
      <td>7.722789</td>
    </tr>
    <tr>
      <th>Lead_ Actor_Rating</th>
      <td>4346.690855</td>
    </tr>
    <tr>
      <th>Critic_rating</th>
      <td>3122.433825</td>
    </tr>
    <tr>
      <th>Trailer_views</th>
      <td>151.296125</td>
    </tr>
    <tr>
      <th>Time_taken</th>
      <td>42.416747</td>
    </tr>
    <tr>
      <th>Twitter_hastags</th>
      <td>0.832483</td>
    </tr>
    <tr>
      <th>Avg_age_actors</th>
      <td>28.492446</td>
    </tr>
    <tr>
      <th>Genre_Comedy</th>
      <td>3320.151878</td>
    </tr>
    <tr>
      <th>Genre_Drama</th>
      <td>3573.406719</td>
    </tr>
    <tr>
      <th>Genre_Thriller</th>
      <td>3245.247598</td>
    </tr>
    <tr>
      <th>3D_available_YES</th>
      <td>2526.395364</td>
    </tr>
  </tbody>
</table>
</div>




```python
predict_linear = lm_linear.predict(X_test)
```


```python
sns.jointplot(x=y_test,y=predict_linear)
```


<img src="{{ site.url }}{{ site.baseurl }}/images/Predict Movie Collection/Linear Regression and others_52_1.png">


_We need to standardize data to be used with Ridge and Lasso regression. Also, we need to find an optimum value for the tuning parameter._


```python
from sklearn.model_selection import validation_curve
```


```python
from sklearn.preprocessing import StandardScaler
```


```python
#scaling and transforming data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
```


```python
#Performing cross validation to get best alpha value
param_alpha = np.logspace(-2,8,100)
train_scores, test_scores = validation_curve(Ridge(),X_train_scaled,y_train,"alpha",param_alpha,scoring='r2')
```

    /Users/vanya/opt/anaconda3/lib/python3.8/site-packages/sklearn/utils/validation.py:68: FutureWarning: Pass param_name=alpha, param_range=[1.00000000e-02 1.26185688e-02 1.59228279e-02 2.00923300e-02
     2.53536449e-02 3.19926714e-02 4.03701726e-02 5.09413801e-02
     6.42807312e-02 8.11130831e-02 1.02353102e-01 1.29154967e-01
     1.62975083e-01 2.05651231e-01 2.59502421e-01 3.27454916e-01
     4.13201240e-01 5.21400829e-01 6.57933225e-01 8.30217568e-01
     1.04761575e+00 1.32194115e+00 1.66810054e+00 2.10490414e+00
     2.65608778e+00 3.35160265e+00 4.22924287e+00 5.33669923e+00
     6.73415066e+00 8.49753436e+00 1.07226722e+01 1.35304777e+01
     1.70735265e+01 2.15443469e+01 2.71858824e+01 3.43046929e+01
     4.32876128e+01 5.46227722e+01 6.89261210e+01 8.69749003e+01
     1.09749877e+02 1.38488637e+02 1.74752840e+02 2.20513074e+02
     2.78255940e+02 3.51119173e+02 4.43062146e+02 5.59081018e+02
     7.05480231e+02 8.90215085e+02 1.12332403e+03 1.41747416e+03
     1.78864953e+03 2.25701972e+03 2.84803587e+03 3.59381366e+03
     4.53487851e+03 5.72236766e+03 7.22080902e+03 9.11162756e+03
     1.14975700e+04 1.45082878e+04 1.83073828e+04 2.31012970e+04
     2.91505306e+04 3.67837977e+04 4.64158883e+04 5.85702082e+04
     7.39072203e+04 9.32603347e+04 1.17681195e+05 1.48496826e+05
     1.87381742e+05 2.36448941e+05 2.98364724e+05 3.76493581e+05
     4.75081016e+05 5.99484250e+05 7.56463328e+05 9.54548457e+05
     1.20450354e+06 1.51991108e+06 1.91791026e+06 2.42012826e+06
     3.05385551e+06 3.85352859e+06 4.86260158e+06 6.13590727e+06
     7.74263683e+06 9.77009957e+06 1.23284674e+07 1.55567614e+07
     1.96304065e+07 2.47707636e+07 3.12571585e+07 3.94420606e+07
     4.97702356e+07 6.28029144e+07 7.92482898e+07 1.00000000e+08] as keyword args. From version 0.25 passing these as positional arguments will result in an error
      warnings.warn("Pass {} as keyword args. From version 0.25 "



```python
test_mean = test_scores.mean(axis=1)
```


```python
test_mean
```




    array([ 0.66295654,  0.66295759,  0.66295891,  0.66296058,  0.66296268,
            0.66296533,  0.66296868,  0.66297289,  0.66297821,  0.66298491,
            0.66299335,  0.66300397,  0.66301734,  0.66303415,  0.66305528,
            0.66308179,  0.66311501,  0.66315656,  0.66320842,  0.66327294,
            0.66335292,  0.66345159,  0.66357257,  0.66371973,  0.66389691,
            0.66410742,  0.6643531 ,  0.66463302,  0.66494138,  0.66526465,
            0.6655776 ,  0.66583833,  0.66598227,  0.66591539,  0.6655071 ,
            0.66458335,  0.66292045,  0.66024053,  0.65620961,  0.65044075,
            0.64250523,  0.63195527,  0.61835943,  0.60134782,  0.5806585 ,
            0.55617433,  0.52794199,  0.49617368,  0.46123997,  0.42366408,
            0.38411958,  0.34342169,  0.30249646,  0.26231859,  0.22382464,
            0.18782195,  0.15491768,  0.1254839 ,  0.09966124,  0.07739249,
            0.05847218,  0.04259934,  0.02942417,  0.0185845 ,  0.009731  ,
            0.0025426 , -0.00326576, -0.00794076, -0.01169177, -0.01469385,
           -0.01709169, -0.01900384, -0.02052671, -0.02173833, -0.02270153,
           -0.02346674, -0.02407435, -0.02455663, -0.02493929, -0.02524285,
           -0.0254836 , -0.02567451, -0.02582587, -0.02594587, -0.026041  ,
           -0.02611641, -0.02617618, -0.02622355, -0.0262611 , -0.02629086,
           -0.02631444, -0.02633313, -0.02634795, -0.02635969, -0.02636899,
           -0.02637636, -0.02638221, -0.02638684, -0.02639051, -0.02639342])



_Best value for alpha from our range will be with max R2 value._


```python
np.where(test_mean == test_mean.max())[0][0]
```




    32




```python
lm_ridge = Ridge(alpha=param_alpha[32])
```


```python
lm_ridge.fit(X_train_scaled,y_train)
predict_ridge = lm_ridge.predict(X_test_scaled)
```


```python
sns.jointplot(x=y_test,y=predict_ridge)
```



<img src="{{ site.url }}{{ site.baseurl }}/images/Predict Movie Collection/Linear Regression and others_64_1.png">



```python
train_scores, test_scores = validation_curve(Lasso(),X_train_scaled,y_train,"alpha",param_alpha,scoring='r2')
```

    /Users/vanya/opt/anaconda3/lib/python3.8/site-packages/sklearn/utils/validation.py:68: FutureWarning: Pass param_name=alpha, param_range=[1.00000000e-02 1.26185688e-02 1.59228279e-02 2.00923300e-02
     2.53536449e-02 3.19926714e-02 4.03701726e-02 5.09413801e-02
     6.42807312e-02 8.11130831e-02 1.02353102e-01 1.29154967e-01
     1.62975083e-01 2.05651231e-01 2.59502421e-01 3.27454916e-01
     4.13201240e-01 5.21400829e-01 6.57933225e-01 8.30217568e-01
     1.04761575e+00 1.32194115e+00 1.66810054e+00 2.10490414e+00
     2.65608778e+00 3.35160265e+00 4.22924287e+00 5.33669923e+00
     6.73415066e+00 8.49753436e+00 1.07226722e+01 1.35304777e+01
     1.70735265e+01 2.15443469e+01 2.71858824e+01 3.43046929e+01
     4.32876128e+01 5.46227722e+01 6.89261210e+01 8.69749003e+01
     1.09749877e+02 1.38488637e+02 1.74752840e+02 2.20513074e+02
     2.78255940e+02 3.51119173e+02 4.43062146e+02 5.59081018e+02
     7.05480231e+02 8.90215085e+02 1.12332403e+03 1.41747416e+03
     1.78864953e+03 2.25701972e+03 2.84803587e+03 3.59381366e+03
     4.53487851e+03 5.72236766e+03 7.22080902e+03 9.11162756e+03
     1.14975700e+04 1.45082878e+04 1.83073828e+04 2.31012970e+04
     2.91505306e+04 3.67837977e+04 4.64158883e+04 5.85702082e+04
     7.39072203e+04 9.32603347e+04 1.17681195e+05 1.48496826e+05
     1.87381742e+05 2.36448941e+05 2.98364724e+05 3.76493581e+05
     4.75081016e+05 5.99484250e+05 7.56463328e+05 9.54548457e+05
     1.20450354e+06 1.51991108e+06 1.91791026e+06 2.42012826e+06
     3.05385551e+06 3.85352859e+06 4.86260158e+06 6.13590727e+06
     7.74263683e+06 9.77009957e+06 1.23284674e+07 1.55567614e+07
     1.96304065e+07 2.47707636e+07 3.12571585e+07 3.94420606e+07
     4.97702356e+07 6.28029144e+07 7.92482898e+07 1.00000000e+08] as keyword args. From version 0.25 passing these as positional arguments will result in an error
      warnings.warn("Pass {} as keyword args. From version 0.25 "



```python
test_mean = test_scores.mean(axis=1)
```


```python
lm_lasso = Lasso(alpha=param_alpha[np.where(test_mean==test_mean.max())[0][0]])
lm_lasso.fit(X_train_scaled,y_train)
predict_lasso = lm_lasso.predict(X_test_scaled)
```


```python
sns.jointplot(x=y_test,y=predict_lasso)
```

<img src="{{ site.url }}{{ site.baseurl }}/images/Predict Movie Collection/Linear Regression and others_68_1.png">


### Evaluating the model


```python
from sklearn.metrics import r2_score, mean_squared_error
```


```python
print("R2 Score --> higher is better")
print("Linear:", r2_score(y_test,predict_linear))
print("Ridge:", r2_score(y_test,predict_ridge))
print("Lasso:", r2_score(y_test,predict_lasso))
```

    R2 Score --> higher is better
    Linear: 0.7468007748323722
    Ridge: 0.7569419920394107
    Lasso: 0.7571808357758425



```python
print("Root mean square error --> lower is better")
print("Linear:", np.sqrt(mean_squared_error(y_test,predict_linear)))
print("Ridge:", np.sqrt(mean_squared_error(y_test,predict_ridge)))
print("Lasso:", np.sqrt(mean_squared_error(y_test,predict_lasso)))
```

    Root mean square error --> lower is better
    Linear: 9194.62359790369
    Ridge: 9008.608966959373
    Lasso: 9004.181672649082


### Result: We were able to train our model using the data avaliable to determine movie collection using Linear, Ridge and Lasso regression techniques. As per the result, all three are quite close in R2 score and RMSE with Lasso being the best. Owing to the small size of the dataset, the results are very close.
