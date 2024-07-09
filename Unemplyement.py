import pandas as pd
import numpy as np


# -------------------------------------------------------------------- Step-1 : DATA MANIPULATION .  



# importing the dataset.
data = pd.read_csv('rate_in_india.csv')
df = pd.DataFrame(data)  

# checking if there is any null values
null_vals = df.isnull().sum()
# print(null_vals)

# As there are 28 rows which are totally null for each column , so we will drop these values .
df.dropna(inplace=True)


# checking if there is any duplicated value.
duplicates = df.duplicated().sum()
# print(duplicates)

# printing all column names so that we can find useless columns.
# print(df.columns)

# frequency has only one data name monthly so it will be not used in visualisation or any analiyzation .
df.drop([' Frequency']  , axis=1, inplace=True)

# this is for usability the column .
df['employed_number'] = df[' Estimated Employed']
df.drop(' Estimated Employed',axis=1, inplace=True)

df['Unemployed_rate'] = df[' Estimated Unemployment Rate (%)']
df.drop(' Estimated Unemployment Rate (%)',axis=1, inplace=True)
 

# Dates are not helping in visualisation in default formate , so change it in year formate. 
for i in df[' Date']:
    if '2019' in i:
       df[' Date'].replace(to_replace=i , value='2019', inplace=True)
    else:
        df[' Date'].replace(to_replace=i , value='2020', inplace=True)





# PRINTING FINAL DATASET.
# print(df)




# -------------------------------------------------------------------STEP-2 : DATA VISUALIZATION.



import seaborn as sns
import matplotlib.pyplot as plt



# ploting a graph all vs all.
# sns.pairplot( data=df  )  
# plt.show()


# ploting the barplot between unemployement rate and region , they will be saperated by Date . 

# sns.barplot(x =df['Unemployed_rate'], y = df['Region']  , data=df , errwidth=0 , hue=df[' Date'] , orient='h'  )  
# plt.title('Rate of Unemployement during covid behalf of year ' ,fontsize = 20)
# plt.xlabel('Unemployed_rate' , fontsize=18)
# plt.ylabel('Region' , fontsize=18)
# plt.show()

# ploting the barplot between unemployement rate and region , they will be saperated by Area . 

# sns.barplot(x =df['Unemployed_rate'], y = df['Region']  , data=df , errwidth=0 , hue=df['Area']  ) 
# plt.title('Rate of Unemployement during covid behalf of Area ' ,fontsize = 20) 
# plt.xlabel('Unemployed_rate' , fontsize=18)
# plt.ylabel('Region' , fontsize=18)
# plt.show()




# --------------------------------------------------------- step - 3 ). CREATING AND TRAINING THE MODEL.



# we will create the KNN Model as we want to predict the area is urban or rural.


# DATA PREPROCESSING.

# Data scaling in Unemployed data. as the data is much scattered.

from sklearn.preprocessing import StandardScaler  , MinMaxScaler
ss = StandardScaler()
df['Unemployed_rate'] = ss.fit_transform(df[['Unemployed_rate']]) 
df['employed_number'] = ss.fit_transform(df[['employed_number']]) 
df[' Estimated Labour Participation Rate (%)'] = ss.fit_transform(df[[' Estimated Labour Participation Rate (%)']]) 

from sklearn.preprocessing import OneHotEncoder  , OrdinalEncoder
df['Area'] = OneHotEncoder().fit_transform(df[['Area']]).toarray()[:,1]   
df['Region'] = OrdinalEncoder().fit_transform(df[['Region']])


# print(df)
# feature data is in x.
x = df[[' Estimated Labour Participation Rate (%)' ,'employed_number' , 'Region' ,' Date' ,'Unemployed_rate'  ]] 
print(x)
# target data is in y.
y = df['Area']  

# spliting the data for training and testing. 
from sklearn.model_selection import train_test_split
x_train  , x_test  , y_train ,y_test = train_test_split(x , y , test_size=0.2 , random_state=0)
  


# MAKING OUR KNN  MODEL.
from sklearn.neighbors import KNeighborsClassifier 
import math
k=math.sqrt(len(y_test)) 
kn = KNeighborsClassifier(n_neighbors=9,  metric='euclidean')
kn.fit(x_train , y_train)
pred = kn.predict(x_test)


 # PRINTING THE ACCURACY OF OUR MODEL.
from sklearn.metrics import accuracy_score
sc = accuracy_score(y_test , pred)
print('Accuracy : ' , sc*100 ,'%')


# PRINTING THE CONFUSION MATRIX .
from sklearn.metrics import confusion_matrix
a = confusion_matrix(y_test , pred)
print()
print('Confusion metrics : ')
print(a)
print()
 