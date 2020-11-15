---
title: "Basics : K-Nearest Neighbors"
date: 2020-10-04
tags: [data science, KNN, K nearest neighbors, machine learning algorithms]
excerpt: "Data Science, KNN, matplotlib, seaborn, StandardScaler, K nearest neighbors"
mathjax: "true"
---

### Objective: Given some "Classified Data", train the model to categorize data points.
Source: [Udemy Course:](https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/)Python for Data Science and Machine Learning Bootcamp  
Data used in the below analysis: [link](https://github.com/Vanya-16/DataSets/blob/master/KNN_Project_Data)

```python
#import necessary libraries for analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#set this to view plots in jupyter notebook
%matplotlib inline
sns.set_style('whitegrid')
```


```python
knn_data = pd.read_csv('KNN_Project_Data') #read data into dataframe
```


```python
knn_data.head(3) #view some data entries
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
      <th>XVPM</th>
      <th>GWYH</th>
      <th>TRAT</th>
      <th>TLLZ</th>
      <th>IGGA</th>
      <th>HYKR</th>
      <th>EDFS</th>
      <th>GUUB</th>
      <th>MGJM</th>
      <th>JHZC</th>
      <th>TARGET CLASS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1636.670614</td>
      <td>817.988525</td>
      <td>2565.995189</td>
      <td>358.347163</td>
      <td>550.417491</td>
      <td>1618.870897</td>
      <td>2147.641254</td>
      <td>330.727893</td>
      <td>1494.878631</td>
      <td>845.136088</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1013.402760</td>
      <td>577.587332</td>
      <td>2644.141273</td>
      <td>280.428203</td>
      <td>1161.873391</td>
      <td>2084.107872</td>
      <td>853.404981</td>
      <td>447.157619</td>
      <td>1193.032521</td>
      <td>861.081809</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1300.035501</td>
      <td>820.518697</td>
      <td>2025.854469</td>
      <td>525.562292</td>
      <td>922.206261</td>
      <td>2552.355407</td>
      <td>818.676686</td>
      <td>845.491492</td>
      <td>1968.367513</td>
      <td>1647.186291</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



*Since this data is artificial, we would view all the features for better understanding*


```python
#pair plot of the entire data
sns.pairplot(data=knn_data,hue='TARGET CLASS')
```

<img src="{{ site.url }}{{ site.baseurl }}/images/KNN/pairplot_knn.png" alt="Pair plot of all features">


### Standardize the variables


```python
from sklearn.preprocessing import StandardScaler
```


```python
scaler = StandardScaler()
```


```python
scaler.fit(knn_data.drop('TARGET CLASS',axis=1))
```




    StandardScaler()




```python
scaled_version = scaler.transform(knn_data.drop('TARGET CLASS',axis=1)) #scaled the features
```


```python
#convert the scaled array into a DataFrame
knn_data_scaled = pd.DataFrame(scaled_version,columns=knn_data.columns[:-1])
```


```python
knn_data_scaled.head() #check the head to see if scaling worked
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
      <th>XVPM</th>
      <th>GWYH</th>
      <th>TRAT</th>
      <th>TLLZ</th>
      <th>IGGA</th>
      <th>HYKR</th>
      <th>EDFS</th>
      <th>GUUB</th>
      <th>MGJM</th>
      <th>JHZC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.568522</td>
      <td>-0.443435</td>
      <td>1.619808</td>
      <td>-0.958255</td>
      <td>-1.128481</td>
      <td>0.138336</td>
      <td>0.980493</td>
      <td>-0.932794</td>
      <td>1.008313</td>
      <td>-1.069627</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.112376</td>
      <td>-1.056574</td>
      <td>1.741918</td>
      <td>-1.504220</td>
      <td>0.640009</td>
      <td>1.081552</td>
      <td>-1.182663</td>
      <td>-0.461864</td>
      <td>0.258321</td>
      <td>-1.041546</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.660647</td>
      <td>-0.436981</td>
      <td>0.775793</td>
      <td>0.213394</td>
      <td>-0.053171</td>
      <td>2.030872</td>
      <td>-1.240707</td>
      <td>1.149298</td>
      <td>2.184784</td>
      <td>0.342811</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.011533</td>
      <td>0.191324</td>
      <td>-1.433473</td>
      <td>-0.100053</td>
      <td>-1.507223</td>
      <td>-1.753632</td>
      <td>-1.183561</td>
      <td>-0.888557</td>
      <td>0.162310</td>
      <td>-0.002793</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.099059</td>
      <td>0.820815</td>
      <td>-0.904346</td>
      <td>1.609015</td>
      <td>-0.282065</td>
      <td>-0.365099</td>
      <td>-1.095644</td>
      <td>0.391419</td>
      <td>-1.365603</td>
      <td>0.787762</td>
    </tr>
  </tbody>
</table>
</div>



_We can see that the data is now in a standard scale and has closer values as compared to huge values in the initial dataset_

### Split the data into train and test


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(knn_data_scaled,knn_data['TARGET CLASS'],test_size=0.3,random_state=101)
```

### Apply KNN on the data


```python
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=1)
```


```python
KNN.fit(X_train,y_train)
```




    KNeighborsClassifier(n_neighbors=1)



### Predictions and Evaluations


```python
predictions = KNN.predict(X_test)
```


```python
from sklearn.metrics import classification_report, confusion_matrix
```


```python
print(confusion_matrix(y_test,predictions))
```

    [[109  43]
     [ 41 107]]



```python
print(classification_report(y_test,predictions))
```

                  precision    recall  f1-score   support

               0       0.73      0.72      0.72       152
               1       0.71      0.72      0.72       148

        accuracy                           0.72       300
       macro avg       0.72      0.72      0.72       300
    weighted avg       0.72      0.72      0.72       300



### Choosing a K value for better accuracy

_We will use the elbow method to pick a better K value_


```python
error = []
for i in range (1,40):
    KNN = KNeighborsClassifier(n_neighbors=i)
    KNN.fit(X_train,y_train)
    pred_i = KNN.predict(X_test)
    error.append(np.mean(y_test != pred_i))
```


```python
#view the error list
print(error)
```

    [0.28, 0.29, 0.21666666666666667, 0.22, 0.20666666666666667, 0.21, 0.18333333333333332, 0.19, 0.19, 0.17666666666666667, 0.18333333333333332, 0.18333333333333332, 0.18333333333333332, 0.18, 0.18, 0.18, 0.17, 0.17333333333333334, 0.17666666666666667, 0.18333333333333332, 0.17666666666666667, 0.18333333333333332, 0.16666666666666666, 0.18, 0.16666666666666666, 0.17, 0.16666666666666666, 0.17333333333333334, 0.16666666666666666, 0.17333333333333334, 0.16, 0.16666666666666666, 0.17333333333333334, 0.17333333333333334, 0.17, 0.16666666666666666, 0.16, 0.16333333333333333, 0.16]



```python
#plot the error vs K values to better view the result
plt.figure(figsize=(16,7))
plt.plot(range(1,40),error,linestyle='--',color='blue',marker='o',markerfacecolor='red',markersize=10)
plt.xlabel('Values of K')
plt.ylabel('Mean Error')
plt.title('Error rate vs K Value')
```




<img src="{{ site.url }}{{ site.baseurl }}/images/KNN/errorvsK_Knn.png" alt="Error vs Mean error">


### Retraining the data with the better K Value as per the plot above

_Choosing K value as 31_


```python
KNN = KNeighborsClassifier(n_neighbors=31)
KNN.fit(X_train,y_train)
predictions = KNN.predict(X_test)
```


```python
print(confusion_matrix(y_test,predictions))
```

    [[123  29]
     [ 19 129]]



```python
print(classification_report(y_test,predictions))
```

                  precision    recall  f1-score   support

               0       0.87      0.81      0.84       152
               1       0.82      0.87      0.84       148

        accuracy                           0.84       300
       macro avg       0.84      0.84      0.84       300
    weighted avg       0.84      0.84      0.84       300



### Result: We were able to train our model to classify the data into respective TARGET CLASS with an accuracy of 84% (K = 31)
