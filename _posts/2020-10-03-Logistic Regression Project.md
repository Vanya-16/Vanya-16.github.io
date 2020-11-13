---
title: "Basics : Logistic Regression"
date: 2020-10-03
tags: [data science, logistic regression, machine learning algorithms]
header:
  #image: "/images/perceptron/percept.jpg"
excerpt: "Data Science, logistic regression, matplotlib, seaborn"
mathjax: "true"
---


### Objective: To figure out if a user clicked on an advertisement and predict whether or not they will click on an ad based off the features of that user.
Source: Python for Data Science and Machine Learning Bootcamp on Udemy  
Data used in the below analysis: [link](https://github.com/Vanya-16/DataSets/blob/master/advertising.csv)

```python
#importing libraries we'll need
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#used to view plots within jupyter notebook
%matplotlib inline
sns.set_style('whitegrid')
```


```python
#read data from csv file
ad_data = pd.read_csv('advertising.csv')
```


```python
ad_data.head(2)#view data
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
      <th>Daily Time Spent on Site</th>
      <th>Age</th>
      <th>Area Income</th>
      <th>Daily Internet Usage</th>
      <th>Ad Topic Line</th>
      <th>City</th>
      <th>Male</th>
      <th>Country</th>
      <th>Timestamp</th>
      <th>Clicked on Ad</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>68.95</td>
      <td>35</td>
      <td>61833.90</td>
      <td>256.09</td>
      <td>Cloned 5thgeneration orchestration</td>
      <td>Wrightburgh</td>
      <td>0</td>
      <td>Tunisia</td>
      <td>2016-03-27 00:53:11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>80.23</td>
      <td>31</td>
      <td>68441.85</td>
      <td>193.77</td>
      <td>Monitored national standardization</td>
      <td>West Jodi</td>
      <td>1</td>
      <td>Nauru</td>
      <td>2016-04-04 01:39:02</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#view some info on the dataset
ad_data.describe()
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
      <th>Daily Time Spent on Site</th>
      <th>Age</th>
      <th>Area Income</th>
      <th>Daily Internet Usage</th>
      <th>Male</th>
      <th>Clicked on Ad</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>65.000200</td>
      <td>36.009000</td>
      <td>55000.000080</td>
      <td>180.000100</td>
      <td>0.481000</td>
      <td>0.50000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>15.853615</td>
      <td>8.785562</td>
      <td>13414.634022</td>
      <td>43.902339</td>
      <td>0.499889</td>
      <td>0.50025</td>
    </tr>
    <tr>
      <th>min</th>
      <td>32.600000</td>
      <td>19.000000</td>
      <td>13996.500000</td>
      <td>104.780000</td>
      <td>0.000000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>51.360000</td>
      <td>29.000000</td>
      <td>47031.802500</td>
      <td>138.830000</td>
      <td>0.000000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>68.215000</td>
      <td>35.000000</td>
      <td>57012.300000</td>
      <td>183.130000</td>
      <td>0.000000</td>
      <td>0.50000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>78.547500</td>
      <td>42.000000</td>
      <td>65470.635000</td>
      <td>218.792500</td>
      <td>1.000000</td>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>91.430000</td>
      <td>61.000000</td>
      <td>79484.800000</td>
      <td>269.960000</td>
      <td>1.000000</td>
      <td>1.00000</td>
    </tr>
  </tbody>
</table>
</div>




```python
ad_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 10 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   Daily Time Spent on Site  1000 non-null   float64
     1   Age                       1000 non-null   int64  
     2   Area Income               1000 non-null   float64
     3   Daily Internet Usage      1000 non-null   float64
     4   Ad Topic Line             1000 non-null   object
     5   City                      1000 non-null   object
     6   Male                      1000 non-null   int64  
     7   Country                   1000 non-null   object
     8   Timestamp                 1000 non-null   object
     9   Clicked on Ad             1000 non-null   int64  
    dtypes: float64(3), int64(3), object(4)
    memory usage: 78.2+ KB



```python
#Plots to explore the dataset
sns.distplot(ad_data['Age'],bins=30,kde=False)
```


<img src="{{ site.url }}{{ site.baseurl }}/images/Logistic_regression/distplot_LogReg.png" alt="Histogram">



```python
sns.jointplot(x='Age',y='Area Income',data=ad_data)
```


<img src="{{ site.url }}{{ site.baseurl }}/images/Logistic_regression/Jointplot1_LogReg.png" alt="Age vs Area Income plot">





```python
sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,kind='kde')
```



<img src="{{ site.url }}{{ site.baseurl }}/images/Logistic_regression/jointplot2_LogReg.png" alt="Age vs Daily time spent on Site">


```python
sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data)
```



<img src="{{ site.url }}{{ site.baseurl }}/images/Logistic_regression/jointplot3_LogReg.png" alt="Daily Time spent on Site vs Daily Internet Usage">



```python
sns.pairplot(data=ad_data,hue='Clicked on Ad',palette='bwr')
```



<img src="{{ site.url }}{{ site.baseurl }}/images/Logistic_regression/pairplot_LogReg.png" alt="Pairplot">


### Now we start training the model to predict whether the user clicked on Ad


```python
#import required libraries and split the data into train and test
from sklearn.model_selection import train_test_split
```


```python
ad_data.columns
```




    Index(['Daily Time Spent on Site', 'Age', 'Area Income',
           'Daily Internet Usage', 'Ad Topic Line', 'City', 'Male', 'Country',
           'Timestamp', 'Clicked on Ad'],
          dtype='object')



*We can get rid of 'Ad Topic Line', 'City','Country','Timestamp' as these are not numeric column and we are not dealing with non-numeric data to train models yet*


```python
X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income',
       'Daily Internet Usage','Male',]]
y = ad_data['Clicked on Ad']
```


```python
from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test =train_test_split(X, y, test_size=0.3, random_state=101)
```


```python
#import Logistic Regression model and fit data
from sklearn.linear_model import LogisticRegression
```


```python
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
```




    LogisticRegression()




```python
predictions = logmodel.predict(X_test)
```


```python
#Get important metrics to evaluate model
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
```

                  precision    recall  f1-score   support

               0       0.91      0.95      0.93       157
               1       0.94      0.90      0.92       143

        accuracy                           0.93       300
       macro avg       0.93      0.93      0.93       300
    weighted avg       0.93      0.93      0.93       300



### Result: We were able to train a model with an accuracy and recall of 93% indicating its a well fitted model!
