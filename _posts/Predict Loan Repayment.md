---
title: "Predict Loan Repayment"
date: 2020-11-29
tags: [data science, deep learning, classification, neural network]
excerpt: "Data Science, Deep Learning, Neural Network"
mathjax: "true"
---


### Objective: Given historical data on loans given out with information on whether or not the borrower defaulted (charge-off), determine whether or not a borrower will pay back their loan.
[Source:](https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/) Udemy | Python for Data Science and Machine Learning Bootcamp  
Data used in the below analysis: [Subset of LendingClub DataSet from Kaggle](https://www.kaggle.com/wordsforthewise/lending-club).  
Actual files used: [Info](https://github.com/Vanya-16/DataSets/blob/master/lending_club_info.csv) | [Data](https://github.com/Vanya-16/DataSets/blob/master/lending_club_loan_two.csv)


```python
#importing libraries to be used
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
data_info = pd.read_csv('../DATA/lending_club_info.csv', index_col='LoanStatNew') #reading data desc
```


```python
#created a function to get the desc of any column in our dataset
def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])

```


```python
sns.set_style('whitegrid')
```


```python
#reading data in DataFrame
df = pd.read_csv('../DATA/lending_club_loan_two.csv')
```


```python
#Getting details of the data
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 396030 entries, 0 to 396029
    Data columns (total 27 columns):
     #   Column                Non-Null Count   Dtype  
    ---  ------                --------------   -----  
     0   loan_amnt             396030 non-null  float64
     1   term                  396030 non-null  object
     2   int_rate              396030 non-null  float64
     3   installment           396030 non-null  float64
     4   grade                 396030 non-null  object
     5   sub_grade             396030 non-null  object
     6   emp_title             373103 non-null  object
     7   emp_length            377729 non-null  object
     8   home_ownership        396030 non-null  object
     9   annual_inc            396030 non-null  float64
     10  verification_status   396030 non-null  object
     11  issue_d               396030 non-null  object
     12  loan_status           396030 non-null  object
     13  purpose               396030 non-null  object
     14  title                 394275 non-null  object
     15  dti                   396030 non-null  float64
     16  earliest_cr_line      396030 non-null  object
     17  open_acc              396030 non-null  float64
     18  pub_rec               396030 non-null  float64
     19  revol_bal             396030 non-null  float64
     20  revol_util            395754 non-null  float64
     21  total_acc             396030 non-null  float64
     22  initial_list_status   396030 non-null  object
     23  application_type      396030 non-null  object
     24  mort_acc              358235 non-null  float64
     25  pub_rec_bankruptcies  395495 non-null  float64
     26  address               396030 non-null  object
    dtypes: float64(12), object(15)
    memory usage: 81.6+ MB



```python
df.describe().transpose()
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
      <th>loan_amnt</th>
      <td>396030.0</td>
      <td>14113.888089</td>
      <td>8357.441341</td>
      <td>500.00</td>
      <td>8000.00</td>
      <td>12000.00</td>
      <td>20000.00</td>
      <td>40000.00</td>
    </tr>
    <tr>
      <th>int_rate</th>
      <td>396030.0</td>
      <td>13.639400</td>
      <td>4.472157</td>
      <td>5.32</td>
      <td>10.49</td>
      <td>13.33</td>
      <td>16.49</td>
      <td>30.99</td>
    </tr>
    <tr>
      <th>installment</th>
      <td>396030.0</td>
      <td>431.849698</td>
      <td>250.727790</td>
      <td>16.08</td>
      <td>250.33</td>
      <td>375.43</td>
      <td>567.30</td>
      <td>1533.81</td>
    </tr>
    <tr>
      <th>annual_inc</th>
      <td>396030.0</td>
      <td>74203.175798</td>
      <td>61637.621158</td>
      <td>0.00</td>
      <td>45000.00</td>
      <td>64000.00</td>
      <td>90000.00</td>
      <td>8706582.00</td>
    </tr>
    <tr>
      <th>dti</th>
      <td>396030.0</td>
      <td>17.379514</td>
      <td>18.019092</td>
      <td>0.00</td>
      <td>11.28</td>
      <td>16.91</td>
      <td>22.98</td>
      <td>9999.00</td>
    </tr>
    <tr>
      <th>open_acc</th>
      <td>396030.0</td>
      <td>11.311153</td>
      <td>5.137649</td>
      <td>0.00</td>
      <td>8.00</td>
      <td>10.00</td>
      <td>14.00</td>
      <td>90.00</td>
    </tr>
    <tr>
      <th>pub_rec</th>
      <td>396030.0</td>
      <td>0.178191</td>
      <td>0.530671</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>86.00</td>
    </tr>
    <tr>
      <th>revol_bal</th>
      <td>396030.0</td>
      <td>15844.539853</td>
      <td>20591.836109</td>
      <td>0.00</td>
      <td>6025.00</td>
      <td>11181.00</td>
      <td>19620.00</td>
      <td>1743266.00</td>
    </tr>
    <tr>
      <th>revol_util</th>
      <td>395754.0</td>
      <td>53.791749</td>
      <td>24.452193</td>
      <td>0.00</td>
      <td>35.80</td>
      <td>54.80</td>
      <td>72.90</td>
      <td>892.30</td>
    </tr>
    <tr>
      <th>total_acc</th>
      <td>396030.0</td>
      <td>25.414744</td>
      <td>11.886991</td>
      <td>2.00</td>
      <td>17.00</td>
      <td>24.00</td>
      <td>32.00</td>
      <td>151.00</td>
    </tr>
    <tr>
      <th>mort_acc</th>
      <td>358235.0</td>
      <td>1.813991</td>
      <td>2.147930</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>3.00</td>
      <td>34.00</td>
    </tr>
    <tr>
      <th>pub_rec_bankruptcies</th>
      <td>395495.0</td>
      <td>0.121648</td>
      <td>0.356174</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>8.00</td>
    </tr>
  </tbody>
</table>
</div>



### Starting with Exploratory Data Analysis!


```python
sns.countplot(x='loan_status',data=df)
```


<img src="{{ site.url }}{{ site.baseurl }}/images/Predict Loan Repayment/Keras project_9_1.png">


_Since we are trying to predict loan_status, the above plot makes it clear that our historic data is not well balanced. This means our model will not have a great precision / recall as data is unbalanced._


```python
plt.figure(figsize=(12,5))
sns.distplot(df['loan_amnt'], bins=50, kde=False)
```

<img src="{{ site.url }}{{ site.baseurl }}/images/Predict Loan Repayment/Keras project_11_1.png">



_Most of the loan amounts is concentrated around 5000-20000 with very few values at 40000_


```python
#checking correlations among continuous features
df.corr()
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
      <th>loan_amnt</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>annual_inc</th>
      <th>dti</th>
      <th>open_acc</th>
      <th>pub_rec</th>
      <th>revol_bal</th>
      <th>revol_util</th>
      <th>total_acc</th>
      <th>mort_acc</th>
      <th>pub_rec_bankruptcies</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>loan_amnt</th>
      <td>1.000000</td>
      <td>0.168921</td>
      <td>0.953929</td>
      <td>0.336887</td>
      <td>0.016636</td>
      <td>0.198556</td>
      <td>-0.077779</td>
      <td>0.328320</td>
      <td>0.099911</td>
      <td>0.223886</td>
      <td>0.222315</td>
      <td>-0.106539</td>
    </tr>
    <tr>
      <th>int_rate</th>
      <td>0.168921</td>
      <td>1.000000</td>
      <td>0.162758</td>
      <td>-0.056771</td>
      <td>0.079038</td>
      <td>0.011649</td>
      <td>0.060986</td>
      <td>-0.011280</td>
      <td>0.293659</td>
      <td>-0.036404</td>
      <td>-0.082583</td>
      <td>0.057450</td>
    </tr>
    <tr>
      <th>installment</th>
      <td>0.953929</td>
      <td>0.162758</td>
      <td>1.000000</td>
      <td>0.330381</td>
      <td>0.015786</td>
      <td>0.188973</td>
      <td>-0.067892</td>
      <td>0.316455</td>
      <td>0.123915</td>
      <td>0.202430</td>
      <td>0.193694</td>
      <td>-0.098628</td>
    </tr>
    <tr>
      <th>annual_inc</th>
      <td>0.336887</td>
      <td>-0.056771</td>
      <td>0.330381</td>
      <td>1.000000</td>
      <td>-0.081685</td>
      <td>0.136150</td>
      <td>-0.013720</td>
      <td>0.299773</td>
      <td>0.027871</td>
      <td>0.193023</td>
      <td>0.236320</td>
      <td>-0.050162</td>
    </tr>
    <tr>
      <th>dti</th>
      <td>0.016636</td>
      <td>0.079038</td>
      <td>0.015786</td>
      <td>-0.081685</td>
      <td>1.000000</td>
      <td>0.136181</td>
      <td>-0.017639</td>
      <td>0.063571</td>
      <td>0.088375</td>
      <td>0.102128</td>
      <td>-0.025439</td>
      <td>-0.014558</td>
    </tr>
    <tr>
      <th>open_acc</th>
      <td>0.198556</td>
      <td>0.011649</td>
      <td>0.188973</td>
      <td>0.136150</td>
      <td>0.136181</td>
      <td>1.000000</td>
      <td>-0.018392</td>
      <td>0.221192</td>
      <td>-0.131420</td>
      <td>0.680728</td>
      <td>0.109205</td>
      <td>-0.027732</td>
    </tr>
    <tr>
      <th>pub_rec</th>
      <td>-0.077779</td>
      <td>0.060986</td>
      <td>-0.067892</td>
      <td>-0.013720</td>
      <td>-0.017639</td>
      <td>-0.018392</td>
      <td>1.000000</td>
      <td>-0.101664</td>
      <td>-0.075910</td>
      <td>0.019723</td>
      <td>0.011552</td>
      <td>0.699408</td>
    </tr>
    <tr>
      <th>revol_bal</th>
      <td>0.328320</td>
      <td>-0.011280</td>
      <td>0.316455</td>
      <td>0.299773</td>
      <td>0.063571</td>
      <td>0.221192</td>
      <td>-0.101664</td>
      <td>1.000000</td>
      <td>0.226346</td>
      <td>0.191616</td>
      <td>0.194925</td>
      <td>-0.124532</td>
    </tr>
    <tr>
      <th>revol_util</th>
      <td>0.099911</td>
      <td>0.293659</td>
      <td>0.123915</td>
      <td>0.027871</td>
      <td>0.088375</td>
      <td>-0.131420</td>
      <td>-0.075910</td>
      <td>0.226346</td>
      <td>1.000000</td>
      <td>-0.104273</td>
      <td>0.007514</td>
      <td>-0.086751</td>
    </tr>
    <tr>
      <th>total_acc</th>
      <td>0.223886</td>
      <td>-0.036404</td>
      <td>0.202430</td>
      <td>0.193023</td>
      <td>0.102128</td>
      <td>0.680728</td>
      <td>0.019723</td>
      <td>0.191616</td>
      <td>-0.104273</td>
      <td>1.000000</td>
      <td>0.381072</td>
      <td>0.042035</td>
    </tr>
    <tr>
      <th>mort_acc</th>
      <td>0.222315</td>
      <td>-0.082583</td>
      <td>0.193694</td>
      <td>0.236320</td>
      <td>-0.025439</td>
      <td>0.109205</td>
      <td>0.011552</td>
      <td>0.194925</td>
      <td>0.007514</td>
      <td>0.381072</td>
      <td>1.000000</td>
      <td>0.027239</td>
    </tr>
    <tr>
      <th>pub_rec_bankruptcies</th>
      <td>-0.106539</td>
      <td>0.057450</td>
      <td>-0.098628</td>
      <td>-0.050162</td>
      <td>-0.014558</td>
      <td>-0.027732</td>
      <td>0.699408</td>
      <td>-0.124532</td>
      <td>-0.086751</td>
      <td>0.042035</td>
      <td>0.027239</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



_We can better visualize this with a heat map!_


```python
plt.figure(figsize=(13,8))
sns.heatmap(df.corr(), annot=True, cmap='viridis')
```


<img src="{{ site.url }}{{ site.baseurl }}/images/Predict Loan Repayment/Keras project_15_1.png">


_We can explore the high correlation of installment with loan amount. This is as per expectations as we know the amount of installment must depend on the loan taken._


```python
feat_info('installment')
```

    The monthly payment owed by the borrower if the loan originates.



```python
feat_info('loan_amnt')
```

    The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.



```python
plt.figure(figsize=(12,7))
sns.scatterplot(x='installment',y='loan_amnt',data=df)
```



<img src="{{ site.url }}{{ site.baseurl }}/images/Predict Loan Repayment/Keras project_19_1.png">


```python
plt.figure(figsize=(10,5))
sns.boxplot(x='loan_status',y='loan_amnt',data=df)
```


<img src="{{ site.url }}{{ site.baseurl }}/images/Predict Loan Repayment/Keras project_20_1.png">



```python
df.groupby(by='loan_status')['loan_amnt'].describe()
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
    <tr>
      <th>loan_status</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Charged Off</th>
      <td>77673.0</td>
      <td>15126.300967</td>
      <td>8505.090557</td>
      <td>1000.0</td>
      <td>8525.0</td>
      <td>14000.0</td>
      <td>20000.0</td>
      <td>40000.0</td>
    </tr>
    <tr>
      <th>Fully Paid</th>
      <td>318357.0</td>
      <td>13866.878771</td>
      <td>8302.319699</td>
      <td>500.0</td>
      <td>7500.0</td>
      <td>12000.0</td>
      <td>19225.0</td>
      <td>40000.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns
```




    Index(['loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade',
           'emp_title', 'emp_length', 'home_ownership', 'annual_inc',
           'verification_status', 'issue_d', 'loan_status', 'purpose', 'title',
           'dti', 'earliest_cr_line', 'open_acc', 'pub_rec', 'revol_bal',
           'revol_util', 'total_acc', 'initial_list_status', 'application_type',
           'mort_acc', 'pub_rec_bankruptcies', 'address'],
          dtype='object')




```python
df['grade'].unique()
```




    array(['B', 'A', 'C', 'E', 'D', 'F', 'G'], dtype=object)




```python
df['sub_grade'].unique()
```




    array(['B4', 'B5', 'B3', 'A2', 'C5', 'C3', 'A1', 'B2', 'C1', 'A5', 'E4',
           'A4', 'A3', 'D1', 'C2', 'B1', 'D3', 'D5', 'D2', 'E1', 'E2', 'E5',
           'F4', 'E3', 'D4', 'G1', 'F5', 'G2', 'C4', 'F1', 'F3', 'G5', 'G4',
           'F2', 'G3'], dtype=object)




```python
plt.figure(figsize=(10,5))
grade_sorted = sorted(df['grade'].unique())
sns.countplot(x='grade',data=df,hue='loan_status', order=grade_sorted)
```



<img src="{{ site.url }}{{ site.baseurl }}/images/Predict Loan Repayment/Keras project_25_1.png">



```python
plt.figure(figsize=(13,6))
sub_grade_sorted = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data=df, order=sub_grade_sorted, palette = 'coolwarm')
```


<img src="{{ site.url }}{{ site.baseurl }}/images/Predict Loan Repayment/Keras project_26_1.png">


```python
plt.figure(figsize=(15,6))
sub_grade_sorted = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data=df, order=sub_grade_sorted, hue='loan_status')
```

<img src="{{ site.url }}{{ site.baseurl }}/images/Predict Loan Repayment/Keras project_27_1.png">


_It seems like F and G grade loans don't get paid back often_


```python
plt.figure(figsize=(10,6))
df_grade_FG = df[(df['grade']=='F') | (df['grade'] == 'G')]
sub_grade_sorted = sorted(df_grade_FG['sub_grade'].unique())
sns.countplot(x='sub_grade', data = df_grade_FG, order = sub_grade_sorted, hue='loan_status')
```



<img src="{{ site.url }}{{ site.baseurl }}/images/Predict Loan Repayment/Keras project_29_1.png">



```python
def loan_status(string):
    if string == 'Fully Paid':
        return 1
    else:
        return 0
```


```python
df['loan_repaid'] = df['loan_status'].apply(loan_status)
```


```python
df.head()
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
      <th>loan_amnt</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>emp_title</th>
      <th>emp_length</th>
      <th>home_ownership</th>
      <th>annual_inc</th>
      <th>...</th>
      <th>pub_rec</th>
      <th>revol_bal</th>
      <th>revol_util</th>
      <th>total_acc</th>
      <th>initial_list_status</th>
      <th>application_type</th>
      <th>mort_acc</th>
      <th>pub_rec_bankruptcies</th>
      <th>address</th>
      <th>loan_repaid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10000.0</td>
      <td>36 months</td>
      <td>11.44</td>
      <td>329.48</td>
      <td>B</td>
      <td>B4</td>
      <td>Marketing</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>117000.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>36369.0</td>
      <td>41.8</td>
      <td>25.0</td>
      <td>w</td>
      <td>INDIVIDUAL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0174 Michelle Gateway\nMendozaberg, OK 22690</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8000.0</td>
      <td>36 months</td>
      <td>11.99</td>
      <td>265.68</td>
      <td>B</td>
      <td>B5</td>
      <td>Credit analyst</td>
      <td>4 years</td>
      <td>MORTGAGE</td>
      <td>65000.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>20131.0</td>
      <td>53.3</td>
      <td>27.0</td>
      <td>f</td>
      <td>INDIVIDUAL</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>1076 Carney Fort Apt. 347\nLoganmouth, SD 05113</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15600.0</td>
      <td>36 months</td>
      <td>10.49</td>
      <td>506.97</td>
      <td>B</td>
      <td>B3</td>
      <td>Statistician</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>43057.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>11987.0</td>
      <td>92.2</td>
      <td>26.0</td>
      <td>f</td>
      <td>INDIVIDUAL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>87025 Mark Dale Apt. 269\nNew Sabrina, WV 05113</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7200.0</td>
      <td>36 months</td>
      <td>6.49</td>
      <td>220.65</td>
      <td>A</td>
      <td>A2</td>
      <td>Client Advocate</td>
      <td>6 years</td>
      <td>RENT</td>
      <td>54000.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>5472.0</td>
      <td>21.5</td>
      <td>13.0</td>
      <td>f</td>
      <td>INDIVIDUAL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>823 Reid Ford\nDelacruzside, MA 00813</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24375.0</td>
      <td>60 months</td>
      <td>17.27</td>
      <td>609.33</td>
      <td>C</td>
      <td>C5</td>
      <td>Destiny Management Inc.</td>
      <td>9 years</td>
      <td>MORTGAGE</td>
      <td>55000.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>24584.0</td>
      <td>69.8</td>
      <td>43.0</td>
      <td>f</td>
      <td>INDIVIDUAL</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>679 Luna Roads\nGreggshire, VA 11650</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>




```python
plt.figure(figsize=(10,6))
df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')
```



<img src="{{ site.url }}{{ site.baseurl }}/images/Predict Loan Repayment/Keras project_33_1.png">


_The loan repayment is most correlated with annual_inc and mort_acc and highly uncorrelated with interest rate._


```python
print("annual_inc:")
feat_info('annual_inc')
print('mort_acc:')
feat_info('mort_acc')
print('int_rate:')
feat_info('int_rate')
```

    annual_inc:
    The self-reported annual income provided by the borrower during registration.
    mort_acc:
    Number of mortgage accounts.
    int_rate:
    Interest Rate on the loan


### Data PreProcessing!
_We need to fill missing data or remove unnecessary features and convert categorical features into numerical ones using dummy variable (one hot encoding)_


```python
df.shape[0] #number of entries in our data
```




    396030




```python
df.isnull().sum()
```




    loan_amnt                   0
    term                        0
    int_rate                    0
    installment                 0
    grade                       0
    sub_grade                   0
    emp_title               22927
    emp_length              18301
    home_ownership              0
    annual_inc                  0
    verification_status         0
    issue_d                     0
    loan_status                 0
    purpose                     0
    title                    1755
    dti                         0
    earliest_cr_line            0
    open_acc                    0
    pub_rec                     0
    revol_bal                   0
    revol_util                276
    total_acc                   0
    initial_list_status         0
    application_type            0
    mort_acc                37795
    pub_rec_bankruptcies      535
    address                     0
    loan_repaid                 0
    dtype: int64




```python
#convert into % of total data frame
df.isnull().sum()/df.shape[0]
```




    loan_amnt               0.000000
    term                    0.000000
    int_rate                0.000000
    installment             0.000000
    grade                   0.000000
    sub_grade               0.000000
    emp_title               0.057892
    emp_length              0.046211
    home_ownership          0.000000
    annual_inc              0.000000
    verification_status     0.000000
    issue_d                 0.000000
    loan_status             0.000000
    purpose                 0.000000
    title                   0.004431
    dti                     0.000000
    earliest_cr_line        0.000000
    open_acc                0.000000
    pub_rec                 0.000000
    revol_bal               0.000000
    revol_util              0.000697
    total_acc               0.000000
    initial_list_status     0.000000
    application_type        0.000000
    mort_acc                0.095435
    pub_rec_bankruptcies    0.001351
    address                 0.000000
    loan_repaid             0.000000
    dtype: float64



_We need to keep/remove features with missing values. Let's determine that one by one._


```python
print("emp_title:")
feat_info('emp_title')
print("emp_length:")
feat_info('emp_length')
```

    emp_title:
    The job title supplied by the Borrower when applying for the loan.*
    emp_length:
    Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.



```python
df['emp_title'].nunique()
```




    173105




```python
df['emp_title'].value_counts()
```




    Teacher                                   4389
    Manager                                   4250
    Registered Nurse                          1856
    RN                                        1846
    Supervisor                                1830
                                              ...
    Color tech                                   1
    New York City Department of Education        1
    Bartender/Food Server                        1
    Commercial Ara Executive                     1
    El Cortez Hotel & Casino                     1
    Name: emp_title, Length: 173105, dtype: int64



_There are way too many unique titles to convert them into dummy variable, we shall drop it_


```python
df.drop('emp_title', axis=1, inplace=True)
```


```python
sorted(df['emp_length'].dropna().unique())
```




    ['1 year',
     '10+ years',
     '2 years',
     '3 years',
     '4 years',
     '5 years',
     '6 years',
     '7 years',
     '8 years',
     '9 years',
     '< 1 year']




```python
emp_length_order = [ '< 1 year',
                      '1 year',
                     '2 years',
                     '3 years',
                     '4 years',
                     '5 years',
                     '6 years',
                     '7 years',
                     '8 years',
                     '9 years',
                     '10+ years']
```


```python
plt.figure(figsize=(13,5))
sns.countplot(x='emp_length',data=df,order=emp_length_order)
```



<img src="{{ site.url }}{{ site.baseurl }}/images/Predict Loan Repayment/Keras project_48_1.png">



```python
plt.figure(figsize=(13,5))
sns.countplot(x='emp_length',data=df,order=emp_length_order, hue='loan_status')
```



<img src="{{ site.url }}{{ site.baseurl }}/images/Predict Loan Repayment/Keras project_49_1.png">

_This doesn't really inform us if there is a strong relationship between employment length and being charged off, we need a percentage of charge offs per category or what percent of people per employment category didn't pay back their loan._


```python
data_for_emp_len = df[df['loan_repaid']==1].groupby(by='emp_length').count()
data_for_emp_len = pd.merge(data_for_emp_len,
                                    pd.DataFrame(df.groupby(by='emp_length').count()
                                                 ['loan_repaid']).reset_index(),on = 'emp_length')
data_for_emp_len['percentage'] = data_for_emp_len['loan_repaid_x'] / data_for_emp_len['loan_repaid_y']
data_for_emp_len.set_index('emp_length',inplace=True)
data_for_emp_len
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
      <th>loan_amnt</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>home_ownership</th>
      <th>annual_inc</th>
      <th>verification_status</th>
      <th>issue_d</th>
      <th>...</th>
      <th>revol_util</th>
      <th>total_acc</th>
      <th>initial_list_status</th>
      <th>application_type</th>
      <th>mort_acc</th>
      <th>pub_rec_bankruptcies</th>
      <th>address</th>
      <th>loan_repaid_x</th>
      <th>loan_repaid_y</th>
      <th>percentage</th>
    </tr>
    <tr>
      <th>emp_length</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1 year</th>
      <td>20728</td>
      <td>20728</td>
      <td>20728</td>
      <td>20728</td>
      <td>20728</td>
      <td>20728</td>
      <td>20728</td>
      <td>20728</td>
      <td>20728</td>
      <td>20728</td>
      <td>...</td>
      <td>20712</td>
      <td>20728</td>
      <td>20728</td>
      <td>20728</td>
      <td>18126</td>
      <td>20666</td>
      <td>20728</td>
      <td>20728</td>
      <td>25882</td>
      <td>0.800865</td>
    </tr>
    <tr>
      <th>10+ years</th>
      <td>102826</td>
      <td>102826</td>
      <td>102826</td>
      <td>102826</td>
      <td>102826</td>
      <td>102826</td>
      <td>102826</td>
      <td>102826</td>
      <td>102826</td>
      <td>102826</td>
      <td>...</td>
      <td>102766</td>
      <td>102826</td>
      <td>102826</td>
      <td>102826</td>
      <td>95511</td>
      <td>102753</td>
      <td>102826</td>
      <td>102826</td>
      <td>126041</td>
      <td>0.815814</td>
    </tr>
    <tr>
      <th>2 years</th>
      <td>28903</td>
      <td>28903</td>
      <td>28903</td>
      <td>28903</td>
      <td>28903</td>
      <td>28903</td>
      <td>28903</td>
      <td>28903</td>
      <td>28903</td>
      <td>28903</td>
      <td>...</td>
      <td>28886</td>
      <td>28903</td>
      <td>28903</td>
      <td>28903</td>
      <td>25355</td>
      <td>28848</td>
      <td>28903</td>
      <td>28903</td>
      <td>35827</td>
      <td>0.806738</td>
    </tr>
    <tr>
      <th>3 years</th>
      <td>25483</td>
      <td>25483</td>
      <td>25483</td>
      <td>25483</td>
      <td>25483</td>
      <td>25483</td>
      <td>25483</td>
      <td>25483</td>
      <td>25483</td>
      <td>25483</td>
      <td>...</td>
      <td>25468</td>
      <td>25483</td>
      <td>25483</td>
      <td>25483</td>
      <td>22220</td>
      <td>25437</td>
      <td>25483</td>
      <td>25483</td>
      <td>31665</td>
      <td>0.804769</td>
    </tr>
    <tr>
      <th>4 years</th>
      <td>19344</td>
      <td>19344</td>
      <td>19344</td>
      <td>19344</td>
      <td>19344</td>
      <td>19344</td>
      <td>19344</td>
      <td>19344</td>
      <td>19344</td>
      <td>19344</td>
      <td>...</td>
      <td>19333</td>
      <td>19344</td>
      <td>19344</td>
      <td>19344</td>
      <td>16526</td>
      <td>19321</td>
      <td>19344</td>
      <td>19344</td>
      <td>23952</td>
      <td>0.807615</td>
    </tr>
    <tr>
      <th>5 years</th>
      <td>21403</td>
      <td>21403</td>
      <td>21403</td>
      <td>21403</td>
      <td>21403</td>
      <td>21403</td>
      <td>21403</td>
      <td>21403</td>
      <td>21403</td>
      <td>21403</td>
      <td>...</td>
      <td>21391</td>
      <td>21403</td>
      <td>21403</td>
      <td>21403</td>
      <td>18691</td>
      <td>21381</td>
      <td>21403</td>
      <td>21403</td>
      <td>26495</td>
      <td>0.807813</td>
    </tr>
    <tr>
      <th>6 years</th>
      <td>16898</td>
      <td>16898</td>
      <td>16898</td>
      <td>16898</td>
      <td>16898</td>
      <td>16898</td>
      <td>16898</td>
      <td>16898</td>
      <td>16898</td>
      <td>16898</td>
      <td>...</td>
      <td>16884</td>
      <td>16898</td>
      <td>16898</td>
      <td>16898</td>
      <td>15002</td>
      <td>16878</td>
      <td>16898</td>
      <td>16898</td>
      <td>20841</td>
      <td>0.810806</td>
    </tr>
    <tr>
      <th>7 years</th>
      <td>16764</td>
      <td>16764</td>
      <td>16764</td>
      <td>16764</td>
      <td>16764</td>
      <td>16764</td>
      <td>16764</td>
      <td>16764</td>
      <td>16764</td>
      <td>16764</td>
      <td>...</td>
      <td>16747</td>
      <td>16764</td>
      <td>16764</td>
      <td>16764</td>
      <td>15284</td>
      <td>16751</td>
      <td>16764</td>
      <td>16764</td>
      <td>20819</td>
      <td>0.805226</td>
    </tr>
    <tr>
      <th>8 years</th>
      <td>15339</td>
      <td>15339</td>
      <td>15339</td>
      <td>15339</td>
      <td>15339</td>
      <td>15339</td>
      <td>15339</td>
      <td>15339</td>
      <td>15339</td>
      <td>15339</td>
      <td>...</td>
      <td>15327</td>
      <td>15339</td>
      <td>15339</td>
      <td>15339</td>
      <td>14142</td>
      <td>15323</td>
      <td>15339</td>
      <td>15339</td>
      <td>19168</td>
      <td>0.800240</td>
    </tr>
    <tr>
      <th>9 years</th>
      <td>12244</td>
      <td>12244</td>
      <td>12244</td>
      <td>12244</td>
      <td>12244</td>
      <td>12244</td>
      <td>12244</td>
      <td>12244</td>
      <td>12244</td>
      <td>12244</td>
      <td>...</td>
      <td>12235</td>
      <td>12244</td>
      <td>12244</td>
      <td>12244</td>
      <td>11192</td>
      <td>12233</td>
      <td>12244</td>
      <td>12244</td>
      <td>15314</td>
      <td>0.799530</td>
    </tr>
    <tr>
      <th>&lt; 1 year</th>
      <td>25162</td>
      <td>25162</td>
      <td>25162</td>
      <td>25162</td>
      <td>25162</td>
      <td>25162</td>
      <td>25162</td>
      <td>25162</td>
      <td>25162</td>
      <td>25162</td>
      <td>...</td>
      <td>25139</td>
      <td>25162</td>
      <td>25162</td>
      <td>25162</td>
      <td>21629</td>
      <td>25055</td>
      <td>25162</td>
      <td>25162</td>
      <td>31725</td>
      <td>0.793128</td>
    </tr>
  </tbody>
</table>
<p>11 rows × 28 columns</p>
</div>




```python
plt.figure(figsize=(13,5))
data_for_emp_len['percentage'].plot(kind='bar')
```



<img src="{{ site.url }}{{ site.baseurl }}/images/Predict Loan Repayment/Keras project_52_1.png">


_The above plot shows % of poeple who paid their loans from the total number of people who took loan grouped at employee length. We can see that this is extremely same across all emp_lengths so we can drop the same._


```python
df.drop('emp_length',axis=1,inplace=True)
```


```python
df.isnull().sum()
```




    loan_amnt                   0
    term                        0
    int_rate                    0
    installment                 0
    grade                       0
    sub_grade                   0
    home_ownership              0
    annual_inc                  0
    verification_status         0
    issue_d                     0
    loan_status                 0
    purpose                     0
    title                    1755
    dti                         0
    earliest_cr_line            0
    open_acc                    0
    pub_rec                     0
    revol_bal                   0
    revol_util                276
    total_acc                   0
    initial_list_status         0
    application_type            0
    mort_acc                37795
    pub_rec_bankruptcies      535
    address                     0
    loan_repaid                 0
    dtype: int64




```python
print("purpose:")
feat_info('purpose')
print("title:")
feat_info('title')
```

    purpose:
    A category provided by the borrower for the loan request.
    title:
    The loan title provided by the borrower



```python
df['purpose'].unique()
```




    array(['vacation', 'debt_consolidation', 'credit_card',
           'home_improvement', 'small_business', 'major_purchase', 'other',
           'medical', 'wedding', 'car', 'moving', 'house', 'educational',
           'renewable_energy'], dtype=object)




```python
df['title'].unique()
```




    array(['Vacation', 'Debt consolidation', 'Credit card refinancing', ...,
           'Credit buster ', 'Loanforpayoff', 'Toxic Debt Payoff'],
          dtype=object)



_The title is just a sub category for purpose, we can drop it!_


```python
df.drop('title', axis=1, inplace=True)
```


```python
df['mort_acc'].value_counts()
```




    0.0     139777
    1.0      60416
    2.0      49948
    3.0      38049
    4.0      27887
    5.0      18194
    6.0      11069
    7.0       6052
    8.0       3121
    9.0       1656
    10.0       865
    11.0       479
    12.0       264
    13.0       146
    14.0       107
    15.0        61
    16.0        37
    17.0        22
    18.0        18
    19.0        15
    20.0        13
    24.0        10
    22.0         7
    21.0         4
    25.0         4
    27.0         3
    23.0         2
    32.0         2
    26.0         2
    31.0         2
    30.0         1
    28.0         1
    34.0         1
    Name: mort_acc, dtype: int64



_There are many ways we could deal with this missing data. Build a simple model to fill it in, fill it in based on the mean of the other columns, or bin the columns into categories and then set NaN as its own category._



```python
df.corr()['mort_acc'].sort_values()
```




    int_rate               -0.082583
    dti                    -0.025439
    revol_util              0.007514
    pub_rec                 0.011552
    pub_rec_bankruptcies    0.027239
    loan_repaid             0.073111
    open_acc                0.109205
    installment             0.193694
    revol_bal               0.194925
    loan_amnt               0.222315
    annual_inc              0.236320
    total_acc               0.381072
    mort_acc                1.000000
    Name: mort_acc, dtype: float64



_The column mort_acc is most correlated with total_acc._


```python
mean_mort_total = df.groupby(by='total_acc').mean()['mort_acc']
mean_mort_total
```




    total_acc
    2.0      0.000000
    3.0      0.052023
    4.0      0.066743
    5.0      0.103289
    6.0      0.151293
               ...   
    124.0    1.000000
    129.0    1.000000
    135.0    3.000000
    150.0    2.000000
    151.0    0.000000
    Name: mort_acc, Length: 118, dtype: float64



_We'll try to fill the missing values using this series_


```python
def fill_mort_acc(total,mort):
    if np.isnan(mort):
        return mean_mort_total[total]
    else:
        return mort
```


```python
df['mort_acc'] = df.apply(lambda x : fill_mort_acc(x['total_acc'],x['mort_acc']), axis=1)
```


```python
df.isnull().sum()/df.shape[0]
```




    loan_amnt               0.000000
    term                    0.000000
    int_rate                0.000000
    installment             0.000000
    grade                   0.000000
    sub_grade               0.000000
    home_ownership          0.000000
    annual_inc              0.000000
    verification_status     0.000000
    issue_d                 0.000000
    loan_status             0.000000
    purpose                 0.000000
    dti                     0.000000
    earliest_cr_line        0.000000
    open_acc                0.000000
    pub_rec                 0.000000
    revol_bal               0.000000
    revol_util              0.000697
    total_acc               0.000000
    initial_list_status     0.000000
    application_type        0.000000
    mort_acc                0.000000
    pub_rec_bankruptcies    0.001351
    address                 0.000000
    loan_repaid             0.000000
    dtype: float64



_As the data covered by revol_util and pub_rec_bankruptcies account for less than 0.5% of the data, we'll drop these._


```python
df.drop('revol_util',axis=1,inplace=True)
df.drop('pub_rec_bankruptcies',axis=1,inplace=True)
```


```python
df.isnull().sum()
```




    loan_amnt              0
    term                   0
    int_rate               0
    installment            0
    grade                  0
    sub_grade              0
    home_ownership         0
    annual_inc             0
    verification_status    0
    issue_d                0
    loan_status            0
    purpose                0
    dti                    0
    earliest_cr_line       0
    open_acc               0
    pub_rec                0
    revol_bal              0
    total_acc              0
    initial_list_status    0
    application_type       0
    mort_acc               0
    address                0
    loan_repaid            0
    dtype: int64



### Convert categorical data into dummy variables


```python
#getting textual columns
df.select_dtypes(include='object').columns
```




    Index(['term', 'grade', 'sub_grade', 'home_ownership', 'verification_status',
           'issue_d', 'loan_status', 'purpose', 'earliest_cr_line',
           'initial_list_status', 'application_type', 'address'],
          dtype='object')




```python
feat_info('term')
```

    The number of payments on the loan. Values are in months and can be either 36 or 60.



```python
df['term'].unique()
```




    array([' 36 months', ' 60 months'], dtype=object)




```python
def term_change(string):
    if string == ' 36 months':
        return 36
    else:
        return 60
```


```python
df['term'] = df['term'].apply(term_change)
```


```python
df['term'].unique()
```




    array([36, 60])



_Since grade is a part of sub_grade, we can drop it!_


```python
df.drop('grade',axis=1,inplace=True)
```


```python
#converting sub grade into dummy variables
dummy_df = pd.get_dummies(df,columns=['sub_grade'],drop_first=True)
```


```python
dummy_df.select_dtypes(include='object').columns
```




    Index(['home_ownership', 'verification_status', 'issue_d', 'loan_status',
           'purpose', 'earliest_cr_line', 'initial_list_status',
           'application_type', 'address'],
          dtype='object')




```python
features = ['verification_status', 'application_type','initial_list_status','purpose']
dummy_df = pd.get_dummies(dummy_df,columns=features,drop_first=True)
```


```python
dummy_df.columns
```




    Index(['loan_amnt', 'term', 'int_rate', 'installment', 'home_ownership',
           'annual_inc', 'issue_d', 'loan_status', 'dti', 'earliest_cr_line',
           'open_acc', 'pub_rec', 'revol_bal', 'total_acc', 'mort_acc', 'address',
           'loan_repaid', 'sub_grade_A2', 'sub_grade_A3', 'sub_grade_A4',
           'sub_grade_A5', 'sub_grade_B1', 'sub_grade_B2', 'sub_grade_B3',
           'sub_grade_B4', 'sub_grade_B5', 'sub_grade_C1', 'sub_grade_C2',
           'sub_grade_C3', 'sub_grade_C4', 'sub_grade_C5', 'sub_grade_D1',
           'sub_grade_D2', 'sub_grade_D3', 'sub_grade_D4', 'sub_grade_D5',
           'sub_grade_E1', 'sub_grade_E2', 'sub_grade_E3', 'sub_grade_E4',
           'sub_grade_E5', 'sub_grade_F1', 'sub_grade_F2', 'sub_grade_F3',
           'sub_grade_F4', 'sub_grade_F5', 'sub_grade_G1', 'sub_grade_G2',
           'sub_grade_G3', 'sub_grade_G4', 'sub_grade_G5',
           'verification_status_Source Verified', 'verification_status_Verified',
           'application_type_INDIVIDUAL', 'application_type_JOINT',
           'initial_list_status_w', 'purpose_credit_card',
           'purpose_debt_consolidation', 'purpose_educational',
           'purpose_home_improvement', 'purpose_house', 'purpose_major_purchase',
           'purpose_medical', 'purpose_moving', 'purpose_other',
           'purpose_renewable_energy', 'purpose_small_business',
           'purpose_vacation', 'purpose_wedding'],
          dtype='object')




```python
dummy_df.select_dtypes(include='object').columns
```




    Index(['home_ownership', 'issue_d', 'loan_status', 'earliest_cr_line',
           'address'],
          dtype='object')




```python
dummy_df['home_ownership'].value_counts()
```




    MORTGAGE    198348
    RENT        159790
    OWN          37746
    OTHER          112
    NONE            31
    ANY              3
    Name: home_ownership, dtype: int64




```python
def home_owner(string):
    if string == 'NONE' or string == 'ANY':
        return 'OTHER'
    else:
        return string
```


```python
dummy_df['home_ownership'] = dummy_df['home_ownership'].apply(home_owner)
dummy_df['home_ownership'].value_counts()
```




    MORTGAGE    198348
    RENT        159790
    OWN          37746
    OTHER          146
    Name: home_ownership, dtype: int64




```python
dummy_df = pd.get_dummies(dummy_df,columns=['home_ownership'],drop_first=True)
```


```python
dummy_df.select_dtypes(include='object').columns
```




    Index(['issue_d', 'loan_status', 'earliest_cr_line', 'address'], dtype='object')




```python
dummy_df['address'] = dummy_df['address'].apply(lambda x : x[-5:])
dummy_df['address']                            
```




    0         22690
    1         05113
    2         05113
    3         00813
    4         11650
              ...  
    396025    30723
    396026    05113
    396027    70466
    396028    29597
    396029    48052
    Name: address, Length: 396030, dtype: object




```python
dummy_df = pd.get_dummies(dummy_df,columns=['address'],drop_first=True)
dummy_df.columns
```




    Index(['loan_amnt', 'term', 'int_rate', 'installment', 'annual_inc', 'issue_d',
           'loan_status', 'dti', 'earliest_cr_line', 'open_acc', 'pub_rec',
           'revol_bal', 'total_acc', 'mort_acc', 'loan_repaid', 'sub_grade_A2',
           'sub_grade_A3', 'sub_grade_A4', 'sub_grade_A5', 'sub_grade_B1',
           'sub_grade_B2', 'sub_grade_B3', 'sub_grade_B4', 'sub_grade_B5',
           'sub_grade_C1', 'sub_grade_C2', 'sub_grade_C3', 'sub_grade_C4',
           'sub_grade_C5', 'sub_grade_D1', 'sub_grade_D2', 'sub_grade_D3',
           'sub_grade_D4', 'sub_grade_D5', 'sub_grade_E1', 'sub_grade_E2',
           'sub_grade_E3', 'sub_grade_E4', 'sub_grade_E5', 'sub_grade_F1',
           'sub_grade_F2', 'sub_grade_F3', 'sub_grade_F4', 'sub_grade_F5',
           'sub_grade_G1', 'sub_grade_G2', 'sub_grade_G3', 'sub_grade_G4',
           'sub_grade_G5', 'verification_status_Source Verified',
           'verification_status_Verified', 'application_type_INDIVIDUAL',
           'application_type_JOINT', 'initial_list_status_w',
           'purpose_credit_card', 'purpose_debt_consolidation',
           'purpose_educational', 'purpose_home_improvement', 'purpose_house',
           'purpose_major_purchase', 'purpose_medical', 'purpose_moving',
           'purpose_other', 'purpose_renewable_energy', 'purpose_small_business',
           'purpose_vacation', 'purpose_wedding', 'home_ownership_OTHER',
           'home_ownership_OWN', 'home_ownership_RENT', 'address_05113',
           'address_11650', 'address_22690', 'address_29597', 'address_30723',
           'address_48052', 'address_70466', 'address_86630', 'address_93700'],
          dtype='object')




```python
#drop issue_d as we shouldn't know beforehand whether loadn would be issued or now
dummy_df.drop('issue_d',axis=1,inplace=True)
```


```python
feat_info('earliest_cr_line')
```

    The month the borrower's earliest reported credit line was opened



```python
dummy_df['earliest_cr_line'] = dummy_df['earliest_cr_line'].apply(lambda x : int(x[-4:]))
dummy_df['earliest_cr_line']
```




    0         1990
    1         2004
    2         2007
    3         2006
    4         1999
              ...
    396025    2004
    396026    2006
    396027    1997
    396028    1990
    396029    1998
    Name: earliest_cr_line, Length: 396030, dtype: int64




```python
dummy_df.select_dtypes(include='object').columns
```




    Index(['loan_status'], dtype='object')



### We can now start with Model building!


```python
from sklearn.model_selection import train_test_split
dummy_df.drop('loan_status',axis=1,inplace=True) #we alreday have loan_repaid in 0 and 1
```


```python
X = dummy_df.drop('loan_repaid',axis=1).values
y = dummy_df['loan_repaid'].values
```


```python
from sklearn.preprocessing import MinMaxScaler
X_train,X_test,y_train, y_test = train_test_split(X,y,random_state=101,test_size=0.2)
```


```python
scale = MinMaxScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)
```


```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.optimizers import Adam
```


```python
X_train.shape
```




    (316824, 76)


_We are creating a Sequential Model, with activation function as Rectified Linear Unit, with a Dropout layer of 20% neurones switching off, sigmoid function as activation for output. Since its a binary classification problem, we are using Binary Cross-Entropy as loss function and Adam as optimizer._

```python
model = Sequential()

model.add(Dense(units=76, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=38,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=19,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
```


```python
model.fit(X_train,y_train,batch_size=256,epochs=50,validation_data=(X_test,y_test),
          verbose=1)
```

    Epoch 1/50
    1238/1238 [==============================] - 2s 2ms/step - loss: 0.3007 - val_loss: 0.2633
    Epoch 2/50
    1238/1238 [==============================] - 2s 2ms/step - loss: 0.2664 - val_loss: 0.2604
    Epoch 3/50
    1238/1238 [==============================] - 2s 2ms/step - loss: 0.2637 - val_loss: 0.2597
    Epoch 4/50
    1238/1238 [==============================] - 2s 2ms/step - loss: 0.2626 - val_loss: 0.2592
    Epoch 5/50
    1238/1238 [==============================] - 2s 2ms/step - loss: 0.2618 - val_loss: 0.2591
    Epoch 6/50
    1238/1238 [==============================] - 2s 2ms/step - loss: 0.2612 - val_loss: 0.2593
    .
    .
    .
    Epoch 50/50
    1238/1238 [==============================] - 3s 2ms/step - loss: 0.2549 - val_loss: 0.2575





    <tensorflow.python.keras.callbacks.History at 0x7ff31fbf8130>




```python
loss = pd.DataFrame(model.history.history)
loss.plot()
```


<img src="{{ site.url }}{{ site.baseurl }}/images/Predict Loan Repayment/Keras project_107_1.png">


_The model is overfitting. We can try to fix that using Early stopping. We can even play around with dropout layers._


```python
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
```


```python
n_model = Sequential()

n_model.add(Dense(units=76, activation='relu'))
n_model.add(Dropout(0.2))

n_model.add(Dense(units=38,activation='relu'))
n_model.add(Dropout(0.2))

n_model.add(Dense(units=19,activation='relu'))
n_model.add(Dropout(0.2))

n_model.add(Dense(units=1,activation='sigmoid'))
n_model.compile(loss='binary_crossentropy', optimizer='adam')
```


```python
n_model.fit(X_train,y_train,batch_size=256,epochs=50,validation_data=(X_test,y_test),
          verbose=1, callbacks = [early_stop])
```

    Epoch 1/50
    1238/1238 [==============================] - 3s 2ms/step - loss: 0.3013 - val_loss: 0.2625
    Epoch 2/50
    1238/1238 [==============================] - 3s 2ms/step - loss: 0.2666 - val_loss: 0.2607
    Epoch 3/50
    1238/1238 [==============================] - 4s 3ms/step - loss: 0.2639 - val_loss: 0.2593
    Epoch 4/50
    1238/1238 [==============================] - 3s 3ms/step - loss: 0.2630 - val_loss: 0.2593
    Epoch 5/50
    1238/1238 [==============================] - 3s 2ms/step - loss: 0.2618 - val_loss: 0.2594
    Epoch 6/50
    1238/1238 [==============================] - 3s 2ms/step - loss: 0.2614 - val_loss: 0.2589
    Epoch 7/50
    .
    .
    .
    Epoch 21/50
    1238/1238 [==============================] - 3s 2ms/step - loss: 0.2580 - val_loss: 0.2587
    Epoch 00021: early stopping





    <tensorflow.python.keras.callbacks.History at 0x7ff2fe632940>




```python
loss = pd.DataFrame(n_model.history.history)
loss.plot()
```


<img src="{{ site.url }}{{ site.baseurl }}/images/Predict Loan Repayment/Keras project_112_1.png">


### Evaluate the model


```python
from sklearn.metrics import classification_report, confusion_matrix
n_predictions = n_model.predict_classes(X_test)
```


```python
print(confusion_matrix(y_test,n_predictions))
```

    [[ 6840  8653]
     [  114 63599]]



```python
print(classification_report(y_test,n_predictions))
```

                  precision    recall  f1-score   support

               0       0.98      0.44      0.61     15493
               1       0.88      1.00      0.94     63713

        accuracy                           0.89     79206
       macro avg       0.93      0.72      0.77     79206
    weighted avg       0.90      0.89      0.87     79206



### Predicting a new entry


```python
import random
random.seed(101)
random_ind = random.randint(0,len(df))

new_customer = dummy_df.drop('loan_repaid',axis=1).iloc[random_ind]
new_customer
```




    loan_amnt        24000.00
    term                60.00
    int_rate            13.11
    installment        547.43
    annual_inc       85000.00
                       ...   
    address_30723        0.00
    address_48052        0.00
    address_70466        0.00
    address_86630        0.00
    address_93700        0.00
    Name: 304691, Length: 76, dtype: float64




```python
new_customer = scale.transform(new_customer.values.reshape(1,76))
n_model.predict_classes(new_customer)
```




    array([[1]], dtype=int32)



_We've predicted that we would give this person the loan. Let's check if they returned it._


```python
dummy_df.iloc[random_ind]['loan_repaid']
```




    1.0



### Result: We were able to pre-process our data, do some feature engineering and create a model that predicts whether a loan would get repaid or not with an accuracy of 89%.  
#### A mentioned before, due to unbalanced data, out true metric evaluation is ideally given by the F1 score of parametric with less values, i.e. 0. Here, it is 61%.  
#### Whether it is bad or not depends on the metric and the actual business requirement of the model we created. Given the skewed data, this is still pretty good.
