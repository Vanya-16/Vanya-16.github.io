---
title: "Basics : Linear Regression"
date: 2020-05-28
tags: [data science, linear regression]
header:
  #image: "/images/perceptron/percept.jpg"
excerpt: "Data Science, linear regression, matplotlib, seaborn"
mathjax: "true"
---

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
customers = pd.read_csv('Ecommerce Customers')
```


```python
customers.head()
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
    <tr>
      <th>2</th>
      <td>pallen@yahoo.com</td>
      <td>24645 Valerie Unions Suite 582\nCobbborough, D...</td>
      <td>Bisque</td>
      <td>33.000915</td>
      <td>11.330278</td>
      <td>37.110597</td>
      <td>4.104543</td>
      <td>487.547505</td>
    </tr>
    <tr>
      <th>3</th>
      <td>riverarebecca@gmail.com</td>
      <td>1414 David Throughway\nPort Jason, OH 22070-1220</td>
      <td>SaddleBrown</td>
      <td>34.305557</td>
      <td>13.717514</td>
      <td>36.721283</td>
      <td>3.120179</td>
      <td>581.852344</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mstephens@davidson-herman.com</td>
      <td>14023 Rodriguez Passage\nPort Jacobville, PR 3...</td>
      <td>MediumAquaMarine</td>
      <td>33.330673</td>
      <td>12.795189</td>
      <td>37.536653</td>
      <td>4.446308</td>
      <td>599.406092</td>
    </tr>
  </tbody>
</table>
</div>




```python
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
sns.set_style("whitegrid")
```


```python
sns.pairplot(data=customers)
```




    <seaborn.axisgrid.PairGrid at 0x7fee434466a0>




![png](Linear%20Regression%20Project_files/Linear%20Regression%20Project_6_1.png)



```python
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)
```




    <seaborn.axisgrid.JointGrid at 0x7fee4003f850>




![png](Linear%20Regression%20Project_files/Linear%20Regression%20Project_7_1.png)



```python
sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)
```




    <seaborn.axisgrid.JointGrid at 0x7fee44913760>




![png](Linear%20Regression%20Project_files/Linear%20Regression%20Project_8_1.png)



```python
sns.jointplot(x='Time on App',y='Length of Membership',data=customers,kind='hex')
```




    <seaborn.axisgrid.JointGrid at 0x7fee44afd9d0>




![png](Linear%20Regression%20Project_files/Linear%20Regression%20Project_9_1.png)



```python
sns.lmplot(x='Length of Membership',y='Yearly Amount Spent', data = customers)
```




    <seaborn.axisgrid.FacetGrid at 0x7fee437f7e20>




![png](Linear%20Regression%20Project_files/Linear%20Regression%20Project_10_1.png)



```python
customers.columns
```




    Index(['Email', 'Address', 'Avatar', 'Avg. Session Length', 'Time on App',
           'Time on Website', 'Length of Membership', 'Yearly Amount Spent'],
          dtype='object')




```python
X = customers[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']

from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test =train_test_split(X, y, test_size=0.3, random_state=101)
```


```python
from sklearn.linear_model import LinearRegression
```


```python
lm = LinearRegression()
```


```python
lm.fit(X_train,y_train)
```




    LinearRegression()




```python
lm.coef_
```




    array([25.98154972, 38.59015875,  0.19040528, 61.27909654])




```python
lm.predict(X_test)
```




    array([456.44186104, 402.72005312, 409.2531539 , 591.4310343 ,
           590.01437275, 548.82396607, 577.59737969, 715.44428115,
           473.7893446 , 545.9211364 , 337.8580314 , 500.38506697,
           552.93478041, 409.6038964 , 765.52590754, 545.83973731,
           693.25969124, 507.32416226, 573.10533175, 573.2076631 ,
           397.44989709, 555.0985107 , 458.19868141, 482.66899911,
           559.2655959 , 413.00946082, 532.25727408, 377.65464817,
           535.0209653 , 447.80070905, 595.54339577, 667.14347072,
           511.96042791, 573.30433971, 505.02260887, 565.30254655,
           460.38785393, 449.74727868, 422.87193429, 456.55615271,
           598.10493696, 449.64517443, 615.34948995, 511.88078685,
           504.37568058, 515.95249276, 568.64597718, 551.61444684,
           356.5552241 , 464.9759817 , 481.66007708, 534.2220025 ,
           256.28674001, 505.30810714, 520.01844434, 315.0298707 ,
           501.98080155, 387.03842642, 472.97419543, 432.8704675 ,
           539.79082198, 590.03070739, 752.86997652, 558.27858232,
           523.71988382, 431.77690078, 425.38411902, 518.75571466,
           641.9667215 , 481.84855126, 549.69830187, 380.93738919,
           555.18178277, 403.43054276, 472.52458887, 501.82927633,
           473.5561656 , 456.76720365, 554.74980563, 702.96835044,
           534.68884588, 619.18843136, 500.11974127, 559.43899225,
           574.8730604 , 505.09183544, 529.9537559 , 479.20749452,
           424.78407899, 452.20986599, 525.74178343, 556.60674724,
           425.7142882 , 588.8473985 , 490.77053065, 562.56866231,
           495.75782933, 445.17937217, 456.64011682, 537.98437395,
           367.06451757, 421.12767301, 551.59651363, 528.26019754,
           493.47639211, 495.28105313, 519.81827269, 461.15666582,
           528.8711677 , 442.89818166, 543.20201646, 350.07871481,
           401.49148567, 606.87291134, 577.04816561, 524.50431281,
           554.11225704, 507.93347015, 505.35674292, 371.65146821,
           342.37232987, 634.43998975, 523.46931378, 532.7831345 ,
           574.59948331, 435.57455636, 599.92586678, 487.24017405,
           457.66383406, 425.25959495, 331.81731213, 443.70458331,
           563.47279005, 466.14764208, 463.51837671, 381.29445432,
           411.88795623, 473.48087683, 573.31745784, 417.55430913,
           543.50149858, 547.81091537, 547.62977348, 450.99057409,
           561.50896321, 478.30076589, 484.41029555, 457.59099941,
           411.52657592, 375.47900638])




```python
predictions = lm.predict(X_test)
```


```python
plt.scatter(y_test,predictions)
```




    <matplotlib.collections.PathCollection at 0x7fcceb522ee0>




![png](Linear%20Regression%20Project_files/Linear%20Regression%20Project_19_1.png)



```python
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
sns.distplot(y_test-predictions,bins=30)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fcce6e62640>




![png](Linear%20Regression%20Project_files/Linear%20Regression%20Project_22_1.png)



```python
Coeff = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficients'])
```


```python
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




```python

```
