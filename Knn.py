import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import pandas as pd
import os
from sklearn.model_selection import train_test_split

data = pd.read_csv('DATASET PATH')
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

#delete irrevelant columns/features
data=data.drop(['col1'],axis=1)
data=data.drop(['col2'],axis=1)
data=data.drop(['coln'],axis=1)

#Assign dependent&independent variables/sets
y=data.state
x=data.drop(['state'],axis=1)

#prevent missing values
imp = SimpleImputer(missing_values=-99999, strategy="mean")
x.fillna(1, inplace=True)
x= imp.fit_transform(x)

#split data to test and train sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

#create model and accuracy score/fit datas 
predict = KNeighborsClassifier(n_neighbors=3, weights='uniform',algorithm='kd_tree',leaf_size=300,p=2,metric='euclidean',metric_params=None,n_jobs=1)
predict.fit(X_train,y_train)
accuracy=predict.score(X_test, y_test)
print("accuracy score : ",accuracy)

#predict class of new data from knn model
print("%",accuracy*100," oranında:" )
print(predict.predict([['NEW_DATA_COLUMN1","NEW_DATA_COLUMN2","NEW_DATA_COLUMN3","NEW_COLUMN_N"]]))
