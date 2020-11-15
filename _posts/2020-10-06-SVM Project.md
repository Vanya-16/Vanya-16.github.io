---
title: "Basics : Support Vector Machines"
date: 2020-10-06
tags: [data science, svm, iris dataset, grid search, machine learning algorithms]
excerpt: "Data Science, Support Vector Machines, Matplotlib, Seaborn"
mathjax: "true"
---


### Objective: Apply SVM (Support Vector Machines) to the popular Iris dataset and classify the flowers on the basis of their features.
[Source:](https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/) Udemy | Python for Data Science and Machine Learning Bootcamp    


```python
#importing libraries to be used
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#view plots in jupyter notebook
%matplotlib inline
sns.set_style('whitegrid') #setting style for plots, optional
```


```python
#importing data from the seaborn datasets
iris = sns.load_dataset('iris')
```


```python
#check data info
iris
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
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>145</th>
      <td>6.7</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.3</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>146</th>
      <td>6.3</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>1.9</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>147</th>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.0</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>148</th>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>149</th>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
      <td>virginica</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 5 columns</p>
</div>




```python
#visualize/explore dataset
sns.pairplot(data=iris,hue='species')
```

<img src="{{ site.url }}{{ site.baseurl }}/images/SVM/pairplot_svm.png" alt="pairplot of iris data">


_The species 'Setosa' seems to be the most separable_


```python
sns.jointplot(x='sepal_width',y='sepal_length',data=iris[iris['species']=='setosa'],kind='kde',cmap='coolwarm_r')
```


<img src="{{ site.url }}{{ site.baseurl }}/images/SVM/kde_svm.png" alt="KDE plot of width and length of Setosa">

### Split data into Train and test datasets and train model


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.drop('species',axis=1),iris['species'],test_size=0.3, random_state=101)
```


```python
from sklearn.svm import SVC
```


```python
model_svc = SVC()
```


```python
model_svc.fit(X_train,y_train)
```




    SVC()




```python
predictions = model_svc.predict(X_test)
```


```python
from sklearn.metrics import classification_report, confusion_matrix
```


```python
print(confusion_matrix(y_test,predictions))
```

    [[13  0  0]
     [ 0 19  1]
     [ 0  0 12]]



```python
print(classification_report(y_test,predictions))
```

                  precision    recall  f1-score   support

          setosa       1.00      1.00      1.00        13
      versicolor       1.00      0.95      0.97        20
       virginica       0.92      1.00      0.96        12

        accuracy                           0.98        45
       macro avg       0.97      0.98      0.98        45
    weighted avg       0.98      0.98      0.98        45



_The model is already a good fit with an accuracy of 98%. But usually, we might not get such results with default parameters in SVM and we need to do cross validations to get the best parameters to run our model. We can do this with GridSearch._

### Gridsearch Practise


```python
from sklearn.model_selection import GridSearchCV
#Creating a dictionary to sepcify values to do cross validation on
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001]}
```


```python
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(X_train,y_train)
```

    Fitting 5 folds for each of 25 candidates, totalling 125 fits
    [CV] C=0.1, gamma=1 ..................................................
    [CV] ...................... C=0.1, gamma=1, score=0.905, total=   0.0s
    [CV] C=0.1, gamma=1 ..................................................
    [CV] ...................... C=0.1, gamma=1, score=1.000, total=   0.0s
    [CV] C=0.1, gamma=1 ..................................................
    [CV] ...................... C=0.1, gamma=1, score=0.905, total=   0.0s
    [CV] C=0.1, gamma=1 ..................................................
    [CV] ...................... C=0.1, gamma=1, score=0.905, total=   0.0s
    [CV] C=0.1, gamma=1 ..................................................
    .
    .
    .
    [CV] C=1, gamma=0.001 ................................................


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   2 out of   2 | elapsed:    0.0s remaining:    0.0s


    [CV] .................... C=1, gamma=0.001, score=0.714, total=   0.0s
    [CV] C=1, gamma=0.001 ................................................
    [CV] .................... C=1, gamma=0.001, score=0.714, total=   0.0s
    [CV] C=1, gamma=0.001 ................................................
    [CV] .................... C=1, gamma=0.001, score=0.714, total=   0.0s
    [CV] C=1, gamma=0.0001 ...............................................
    [CV] ................... C=1, gamma=0.0001, score=0.333, total=   0.0s
    .
    .
    .
    [CV] ................ C=1000, gamma=0.0001, score=1.000, total=   0.0s


    [Parallel(n_jobs=1)]: Done 125 out of 125 | elapsed:    0.6s finished


    GridSearchCV(estimator=SVC(),
                 param_grid={'C': [0.1, 1, 10, 100, 1000],
                             'gamma': [1, 0.1, 0.01, 0.001, 0.0001]},
                 verbose=3)


```python
n_predictions = grid.predict(X_test)
```


```python
print(confusion_matrix(y_test,n_predictions))
```

    [[13  0  0]
     [ 0 19  1]
     [ 0  0 12]]



```python
print(classification_report(y_test,n_predictions))
```

                  precision    recall  f1-score   support

          setosa       1.00      1.00      1.00        13
      versicolor       1.00      0.95      0.97        20
       virginica       0.92      1.00      0.96        12

        accuracy                           0.98        45
       macro avg       0.97      0.98      0.98        45
    weighted avg       0.98      0.98      0.98        45



### Result: We were able to successfully train our model and predict iris flower species.  
### The accurancy didn't improve much after grid search as it was already great to begin with and data set was quite small.
