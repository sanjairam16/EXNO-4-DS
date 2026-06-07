# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
FEATURE SCALING
~~~
   import pandas as pd
   from scipy import stats
   import numpy as np
   df=pd.read_csv("bmi.csv")
   df.head()
~~~

  <img width="567" height="271" alt="e4-1" src="https://github.com/user-attachments/assets/516257a0-4c91-4322-bb9e-7ccd771b3eb6" />

~~~

   df.dropna()
~~~


<img width="520" height="490" alt="e4-2" src="https://github.com/user-attachments/assets/1cd7ca1f-58bf-4973-b086-74e3d6470692" />

~~~
   max_vals=np.max(np.abs(df[['Height','Weight']]))
   max_vals
~~~

<img width="657" height="261" alt="e4-3" src="https://github.com/user-attachments/assets/758b2bf6-c805-4550-a0fb-a9d90e8b5f05" />

```
   from sklearn.preprocessing import StandardScaler
   sc=StandardScaler()
   df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
   df.head(10)
```
<img width="817" height="478" alt="e4-4" src="https://github.com/user-attachments/assets/8b4916b9-776d-4f51-9060-6fd3fec1afa1" />

```
   from sklearn.preprocessing import MinMaxScaler
   scaler=MinMaxScaler()
   df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
   df.head(10)

```
<img width="820" height="465" alt="e4-5" src="https://github.com/user-attachments/assets/3015fc29-3682-4270-a74b-2f7d54dfc121" />

~~~
   from sklearn.preprocessing import Normalizer
   scaler=Normalizer()
   df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
   df
~~~

<img width="823" height="508" alt="e4-6" src="https://github.com/user-attachments/assets/7e5199d5-0df5-4d9e-914a-534aca78cdcd" />

~~~
   df1=pd.read_csv("/content/bmi.csv")
   from sklearn.preprocessing import MaxAbsScaler
   scaler=MaxAbsScaler()
   df1[['Height','Weight']]=scaler.fit_transform(df1[['Height','Weight']])
   df1
~~~
<img width="828" height="495" alt="e4-7" src="https://github.com/user-attachments/assets/b3a0e8c4-e15a-412f-82b5-e6b176c249fe" />
~~~
   df2=pd.read_csv("/content/bmi.csv")
   from sklearn.preprocessing import RobustScaler
   scaler=RobustScaler()
   df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
   df2.head()
~~~
<img width="826" height="268" alt="e4-8" src="https://github.com/user-attachments/assets/c06f3858-fc0c-41c9-bd16-7fd0dcaca00f" />

FEATURE SELECTION
~~~
   import pandas as pd
   import numpy as np
   from scipy.stats import chi2_contingency
   import seaborn as sns
   tips=sns.load_dataset('tips')
   tips.head()
~~~

<img width="660" height="285" alt="e4-9" src="https://github.com/user-attachments/assets/242f261c-143e-48de-8303-af3563423cfb" />

~~~
   contingency_table=pd.crosstab(tips['sex'],tips['time'])
   print(contingency_table)
~~~

<img width="706" height="179" alt="e4-10" src="https://github.com/user-attachments/assets/6d712335-ed08-419b-ac2d-4dc9bbf32222" />

~~~
   chi2, p, _, _ = chi2_contingency(contingency_table)
   print(f"Chi-Square Statistic: {chi2}")
   print(f"P-value: {p}")
~~~

<img width="656" height="170" alt="e4-11" src="https://github.com/user-attachments/assets/1aa4e077-94cb-49d7-88d6-5472bacd7fb8" />

~~~
   import pandas as pd
   from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
   data={
       'Feature1':[1,2,3,4,5],
       'Feature2': ['A','B','C','A','B'],
       'Feature3':[0,1,1,0,1],
       'Target' :[0,1,1,0,1]
   }
   df=pd.DataFrame(data)
   X=df[['Feature1','Feature3']]
   y=df['Target']
   selector=SelectKBest(score_func=mutual_info_classif, k=1)
   X_new = selector.fit_transform (X,y)
   selected_feature_indices = selector.get_support(indices=True)
   selected_features = X.columns[selected_feature_indices]
   print("Selected Features:")
   print(selected_features)
~~~


<img width="825" height="260" alt="e4-12" src="https://github.com/user-attachments/assets/8a49fd9f-88ee-43dc-898d-1b9d2ea1b948" />


# RESULT:
Hence , the Feature Scaling and Feature Selection process of the dataset has been carried out successfully.
