---
title: "Basics : Linear Regression"
date: 2020-10-01
tags: [data science, linear regression, machine learning algorithms]
header:
  #image: "/images/perceptron/percept.jpg"
excerpt: "Data Science, linear regression, matplotlib, seaborn"
mathjax: "true"
---


### Objective: Given the customer data of a company, whether it should focus on their website or mobile app experience to increase sales.
Source: [Udemy](https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/)
#### Python for Data Science and Machine Learning Bootcamp  
Data used in the below analysis: [link](https://github.com/Vanya-16/DataSets/blob/master/Ecommerce%20Customers)



```python
#Import all the libraries for data analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#used to view plots within jupyter notebook
%matplotlib inline
sns.set_style("whitegrid") #setting view for plots, optional
```


```python
customers = pd.read_csv('Ecommerce Customers') #import dataset
```


```python
customers.head(2) #view dataset
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
      <th>Email</th>
      <th>Address</th>
      <th>Avatar</th>
      <th>Avg. Session Length</th>
      <th>Time on App</th>
      <th>Time on Website</th>
      <th>Length of Membership</th>
      <th>Yearly Amount Spent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>mstephenson@fernandez.com</td>
      <td>835 Frank Tunnel\nWrightmouth, MI 82180-9605</td>
      <td>Violet</td>
      <td>34.497268</td>
      <td>12.655651</td>
      <td>39.577668</td>
      <td>4.082621</td>
      <td>587.951054</td>
    </tr>
    <tr>
      <th>1</th>
      <td>hduke@hotmail.com</td>
      <td>4547 Archer Common\nDiazchester, CA 06566-8576</td>
      <td>DarkGreen</td>
      <td>31.926272</td>
      <td>11.109461</td>
      <td>37.268959</td>
      <td>2.664034</td>
      <td>392.204933</td>
    </tr>
  </tbody>
</table>
</div>




```python
#view some informormation of the dataset
customers.describe()
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
      <th>Avg. Session Length</th>
      <th>Time on App</th>
      <th>Time on Website</th>
      <th>Length of Membership</th>
      <th>Yearly Amount Spent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>500.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>33.053194</td>
      <td>12.052488</td>
      <td>37.060445</td>
      <td>3.533462</td>
      <td>499.314038</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.992563</td>
      <td>0.994216</td>
      <td>1.010489</td>
      <td>0.999278</td>
      <td>79.314782</td>
    </tr>
    <tr>
      <th>min</th>
      <td>29.532429</td>
      <td>8.508152</td>
      <td>33.913847</td>
      <td>0.269901</td>
      <td>256.670582</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>32.341822</td>
      <td>11.388153</td>
      <td>36.349257</td>
      <td>2.930450</td>
      <td>445.038277</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>33.082008</td>
      <td>11.983231</td>
      <td>37.069367</td>
      <td>3.533975</td>
      <td>498.887875</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>33.711985</td>
      <td>12.753850</td>
      <td>37.716432</td>
      <td>4.126502</td>
      <td>549.313828</td>
    </tr>
    <tr>
      <th>max</th>
      <td>36.139662</td>
      <td>15.126994</td>
      <td>40.005182</td>
      <td>6.922689</td>
      <td>765.518462</td>
    </tr>
  </tbody>
</table>
</div>




```python
customers.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 500 entries, 0 to 499
    Data columns (total 8 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   Email                 500 non-null    object
     1   Address               500 non-null    object
     2   Avatar                500 non-null    object
     3   Avg. Session Length   500 non-null    float64
     4   Time on App           500 non-null    float64
     5   Time on Website       500 non-null    float64
     6   Length of Membership  500 non-null    float64
     7   Yearly Amount Spent   500 non-null    float64
    dtypes: float64(5), object(3)
    memory usage: 31.4+ KB



```python
#View numerics data in the dataset and the relationship with other columns
sns.pairplot(data=customers)
```


<img src="{{ site.url }}{{ site.baseurl }}/images/Linear_regression/Pairplot-LinearReg.png" alt="Pair Plot">



*It seems Lenth of Membership is most correlated with Yearly Amount Spent*


```python
#Playing aorund with the data to check for more correlations
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)
```



<img src="{{ site.url }}{{ site.baseurl }}/images/Linear_regression/Jointgrid_LinearReg.png" alt="Pair Plot">




```python
sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)
```
<img src="{{ site.url }}{{ site.baseurl }}/images/Linear_regression/Jointgrid2_LinearReg.png" alt="Pair Plot">





```python
sns.jointplot(x='Time on App',y='Length of Membership',data=customers,kind='hex')
```


<img src="{{ site.url }}{{ site.baseurl }}/images/Linear_regression/Jointgrid3_LinearReg.png" alt="Pair Plot">



```python
#Can see the correlation clearly here
sns.lmplot(x='Length of Membership',y='Yearly Amount Spent', data = customers)
```


<img src="{{ site.url }}{{ site.baseurl }}/images/Linear_regression/Facetgrid_LinearReg.png" alt="Pair Plot">




### Now we start training the model to predict Yearly Amount spent based on given information


```python
#view column names in the dataset
customers.columns
```




    Index(['Email', 'Address', 'Avatar', 'Avg. Session Length', 'Time on App',
           'Time on Website', 'Length of Membership', 'Yearly Amount Spent'],
          dtype='object')




```python
#divide data into X and y DataFrames
X = customers[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']
#Split the X and y into Training and Test
from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test =train_test_split(X, y, test_size=0.3, random_state=101)
```


```python
#import Linear Regression model from Sci-kit Learn
from sklearn.linear_model import LinearRegression
```


```python
lm = LinearRegression()
lm.fit(X_train,y_train)
```




    LinearRegression()




```python
#view coefficients, or amount of correlation among Yearly Amount spent and other data provided
lm.coef_
```




    array([25.98154972, 38.59015875,  0.19040528, 61.27909654])




```python
#predict the values on the test data
predictions = lm.predict(X_test)
```


```python
#plot the result alongwith the actual data
plt.scatter(y_test,predictions)
```




<img src="{{ site.url }}{{ site.baseurl }}/images/Linear_regression/Scatter_LinearReg.png" alt="Pair Plot">



*Model rendered seems like a good fit*


```python
#import metrics to evaluate model performance
from sklearn import metrics
```


```python
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
```

    MAE: 7.228148653430839
    MSE: 79.81305165097469
    RMSE: 8.933815066978648



```python
#plot the residual amount to check model performance
sns.distplot(y_test-predictions,bins=30)
```


<img src="{{ site.url }}{{ site.baseurl }}/images/Linear_regression/Distplot_LinearReg.png" alt="Pair Plot">



*Check the correlation with coefficients seen before*


```python
Coeff = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficients'])
Coeff
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
      <th>Avg. Session Length</th>
      <td>25.981550</td>
    </tr>
    <tr>
      <th>Time on App</th>
      <td>38.590159</td>
    </tr>
    <tr>
      <th>Time on Website</th>
      <td>0.190405</td>
    </tr>
    <tr>
      <th>Length of Membership</th>
      <td>61.279097</td>
    </tr>
  </tbody>
</table>
</div>



*The above data signify that 1 change in Avg. Session Length will read to 25.98 change in Yearly Amount Spent.*  
*Similarly, for Time on App, Time on Website and Length of Membership*  
*As predicted before, the most impact if from Length of Membership*

### Result: The most correlated field is Length of Membership.  
### However, we need to determine the focus of the company on ehancing the Website / App for users. This could be done depending on time and resources and the approach the company would like to take. Here are three options:  
#### 1. Enhance Website as it doesn't help in the sales right now.  
#### 2. Enhance App for even better performance.  
#### 3. Look for other metrics as the most sales are coming in due to length of membership and neither from Website nor App.
