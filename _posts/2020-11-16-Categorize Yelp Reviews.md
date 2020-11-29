---
title: "Categorize Yelp Reviews"
date: 2020-11-16
tags: [data science, machine learning algorithms, naive bayes, svm, nlp, classification]
excerpt: "Data Science, Natural Language Processing, machine learning algorithms"
mathjax: "true"
---

### Objective: Given the Yelp reviews dataset, classify the reviews into 1 star or 5 star categories based off the text content in the reviews.
[Source:](https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/) Udemy | Python for Data Science and Machine Learning Bootcamp  
Data used in the below analysis: [Yelp Review Data Set from Kaggle](https://www.kaggle.com/c/yelp-recsys-2013).


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
yelp = pd.read_csv('yelp.csv')
```


```python
#viewing the dataset
yelp.head(3)
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
      <th>business_id</th>
      <th>date</th>
      <th>review_id</th>
      <th>stars</th>
      <th>text</th>
      <th>type</th>
      <th>user_id</th>
      <th>cool</th>
      <th>useful</th>
      <th>funny</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9yKzy9PApeiPPOUJEtnvkg</td>
      <td>2011-01-26</td>
      <td>fWKvX83p0-ka4JS3dc6E5A</td>
      <td>5</td>
      <td>My wife took me here on my birthday for breakf...</td>
      <td>review</td>
      <td>rLtl8ZkDX5vH5nAx9C3q5Q</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ZRJwVLyzEJq1VAihDhYiow</td>
      <td>2011-07-27</td>
      <td>IjZ33sJrzXqU-0X6U8NwyA</td>
      <td>5</td>
      <td>I have no idea why some people give bad review...</td>
      <td>review</td>
      <td>0a2KyEL0d3Yb1V6aivbIuQ</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6oRAC4uyJCsJl1X0WZpVSA</td>
      <td>2012-06-14</td>
      <td>IESLBzqUCLdSzSqm0eCSxQ</td>
      <td>4</td>
      <td>love the gyro plate. Rice is so good and I als...</td>
      <td>review</td>
      <td>0hT2KtfLiobPvh6cDC8JQg</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
yelp.describe()
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
      <th>stars</th>
      <th>cool</th>
      <th>useful</th>
      <th>funny</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.777500</td>
      <td>0.876800</td>
      <td>1.409300</td>
      <td>0.701300</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.214636</td>
      <td>2.067861</td>
      <td>2.336647</td>
      <td>1.907942</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.000000</td>
      <td>77.000000</td>
      <td>76.000000</td>
      <td>57.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
yelp.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10000 entries, 0 to 9999
    Data columns (total 10 columns):
     #   Column       Non-Null Count  Dtype
    ---  ------       --------------  -----
     0   business_id  10000 non-null  object
     1   date         10000 non-null  object
     2   review_id    10000 non-null  object
     3   stars        10000 non-null  int64
     4   text         10000 non-null  object
     5   type         10000 non-null  object
     6   user_id      10000 non-null  object
     7   cool         10000 non-null  int64
     8   useful       10000 non-null  int64
     9   funny        10000 non-null  int64
    dtypes: int64(4), object(6)
    memory usage: 781.4+ KB



```python
yelp['text length'] = yelp['text'].apply(len) #getting the length of the reviews
```


```python
yelp.head(3)
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
      <th>business_id</th>
      <th>date</th>
      <th>review_id</th>
      <th>stars</th>
      <th>text</th>
      <th>type</th>
      <th>user_id</th>
      <th>cool</th>
      <th>useful</th>
      <th>funny</th>
      <th>text length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9yKzy9PApeiPPOUJEtnvkg</td>
      <td>2011-01-26</td>
      <td>fWKvX83p0-ka4JS3dc6E5A</td>
      <td>5</td>
      <td>My wife took me here on my birthday for breakf...</td>
      <td>review</td>
      <td>rLtl8ZkDX5vH5nAx9C3q5Q</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>889</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ZRJwVLyzEJq1VAihDhYiow</td>
      <td>2011-07-27</td>
      <td>IjZ33sJrzXqU-0X6U8NwyA</td>
      <td>5</td>
      <td>I have no idea why some people give bad review...</td>
      <td>review</td>
      <td>0a2KyEL0d3Yb1V6aivbIuQ</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1345</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6oRAC4uyJCsJl1X0WZpVSA</td>
      <td>2012-06-14</td>
      <td>IESLBzqUCLdSzSqm0eCSxQ</td>
      <td>4</td>
      <td>love the gyro plate. Rice is so good and I als...</td>
      <td>review</td>
      <td>0hT2KtfLiobPvh6cDC8JQg</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>76</td>
    </tr>
  </tbody>
</table>
</div>



### Some EDA (Explolatory data analysis)


```python
g = sns.FacetGrid(data=yelp,col='stars')
g.map(plt.hist,'text length')
```


<img src="{{ site.url }}{{ site.baseurl }}/images/NLP-Yelp_reviews/facet_nlp.png" alt="ratings vs review length">



```python
plt.figure(figsize=(12,5))
sns.boxplot(x='stars',y='text length',data=yelp)
```


<img src="{{ site.url }}{{ site.baseurl }}/images/NLP-Yelp_reviews/box_nlp.png" alt="ratings vs review length">




```python
plt.figure(figsize=(10,5))
sns.countplot(yelp['stars'])
```




<img src="{{ site.url }}{{ site.baseurl }}/images/NLP-Yelp_reviews/count_nlp.png" alt="stars count">




```python
#Group by and take average
yelp_grp = yelp.groupby(by='stars').mean()
yelp_grp
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
      <th>cool</th>
      <th>useful</th>
      <th>funny</th>
      <th>text length</th>
    </tr>
    <tr>
      <th>stars</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.576769</td>
      <td>1.604806</td>
      <td>1.056075</td>
      <td>826.515354</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.719525</td>
      <td>1.563107</td>
      <td>0.875944</td>
      <td>842.256742</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.788501</td>
      <td>1.306639</td>
      <td>0.694730</td>
      <td>758.498289</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.954623</td>
      <td>1.395916</td>
      <td>0.670448</td>
      <td>712.923142</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.944261</td>
      <td>1.381780</td>
      <td>0.608631</td>
      <td>624.999101</td>
    </tr>
  </tbody>
</table>
</div>




```python
yelp_grp.corr() #check correlation
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
      <th>cool</th>
      <th>useful</th>
      <th>funny</th>
      <th>text length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>cool</th>
      <td>1.000000</td>
      <td>-0.743329</td>
      <td>-0.944939</td>
      <td>-0.857664</td>
    </tr>
    <tr>
      <th>useful</th>
      <td>-0.743329</td>
      <td>1.000000</td>
      <td>0.894506</td>
      <td>0.699881</td>
    </tr>
    <tr>
      <th>funny</th>
      <td>-0.944939</td>
      <td>0.894506</td>
      <td>1.000000</td>
      <td>0.843461</td>
    </tr>
    <tr>
      <th>text length</th>
      <td>-0.857664</td>
      <td>0.699881</td>
      <td>0.843461</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.heatmap(yelp_grp.corr(),cmap='coolwarm',annot=True)
```



<img src="{{ site.url }}{{ site.baseurl }}/images/NLP-Yelp_reviews/heat_nlp.png" alt="correlation between numeric features">



### NLP Classification


```python
#working with only 1 or 5 star reviews
yelp_class = yelp[(yelp.stars == 1) | (yelp.stars == 5)]
```


```python
X = yelp_class['text']
y = yelp_class['stars']
```


```python
#import countvector to convert text into vectors
from sklearn.feature_extraction.text import CountVectorizer
CV = CountVectorizer()
```


```python
X = CV.fit_transform(X)
```


```python
#splitting data into train test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)
```


```python
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
```


```python
nb.fit(X_train,y_train)
```




    MultinomialNB()




```python
predict = nb.predict(X_test)
```


```python
from sklearn.metrics import classification_report, confusion_matrix
```


```python
print(confusion_matrix(y_test,predict))
```

    [[159  69]
     [ 22 976]]



```python
print(classification_report(y_test,predict))
```

                  precision    recall  f1-score   support

               1       0.88      0.70      0.78       228
               5       0.93      0.98      0.96       998

        accuracy                           0.93      1226
       macro avg       0.91      0.84      0.87      1226
    weighted avg       0.92      0.93      0.92      1226



### Applying some text pre-processing


```python
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
```


```python
pipeline = Pipeline([('bow', CountVectorizer()),
                    ('Tf-IDF weights', TfidfTransformer()),('Naive Bayes classifier', MultinomialNB())])
```

_Since all the pre-processing is now incorporated within the pipeline, we need to re-create the train test split_


```python
X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)
```


```python
pipeline.fit(X_train,y_train)
```




    Pipeline(steps=[('bow', CountVectorizer()),
                    ('Tf-IDF weights', TfidfTransformer()),
                    ('Naive Bayes classifier', MultinomialNB())])




```python
n_predict = pipeline.predict(X_test)
```


```python
print(confusion_matrix(y_test,n_predict))
```

    [[  0 228]
     [  0 998]]



```python
print(classification_report(y_test,n_predict))
```

                  precision    recall  f1-score   support

               1       0.00      0.00      0.00       228
               5       0.81      1.00      0.90       998

        accuracy                           0.81      1226
       macro avg       0.41      0.50      0.45      1226
    weighted avg       0.66      0.81      0.73      1226



    /Users/vanya/opt/anaconda3/lib/python3.8/site-packages/sklearn/metrics/_classification.py:1221: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))


_It seems like incorporating TF-IDF in our dataset made things pretty bad. We can try to use a different classifier with TF-IDF or not use TF-IDF at all like before!_
#### Replacing the Naive Bayes with SVM


```python
from sklearn.svm import SVC
```


```python
pipeline = Pipeline([('bow', CountVectorizer()),
                    ('Tf-IDF weights', TfidfTransformer()),('SVM classifier', SVC())])
```


```python
pipeline.fit(X_train,y_train)
n_predict = pipeline.predict(X_test)
```


```python
print(confusion_matrix(y_test,n_predict))
```

    [[134  94]
     [  6 992]]



```python
#Comparing SVM and TF-IDF results with Naive Bayes without TF-IDF
print("SVM and TF-IDF")
print(classification_report(y_test,n_predict))
print('Naive Bayes \n')
print(classification_report(y_test,predict))
```

    SVM and TF-IDF
                  precision    recall  f1-score   support

               1       0.96      0.59      0.73       228
               5       0.91      0.99      0.95       998

        accuracy                           0.92      1226
       macro avg       0.94      0.79      0.84      1226
    weighted avg       0.92      0.92      0.91      1226

    Naive Bayes

                  precision    recall  f1-score   support

               1       0.88      0.70      0.78       228
               5       0.93      0.98      0.96       998

        accuracy                           0.93      1226
       macro avg       0.91      0.84      0.87      1226
    weighted avg       0.92      0.93      0.92      1226



### Result: We were able to train and categorize reviews to 1 or 5 star ratings.
#### Our dataset didn't perform well with Naive Bayes and TF-IDF processing, though applying SVM with TF-IDF gave us better precision. Only using the Naive Bayes classifier with text pre-processing also gave us great results, with a better recall rate than the former.  
#### It depends which metric needs to be improved upon and accordingly we can choose the model to apply!
