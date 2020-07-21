#Libraries
import numpy as np
import pandas as pd
import pickle

#Dataset
dataset= pd.read_csv('Social_Network_Ads.csv')
X= dataset.iloc[:,2:4].values
Y= dataset.iloc[:,4].values

#Splitting dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.25, random_state=0)

#Scaling data
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

#Classifier
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train,Y_train)

#Prediction
Y_pred= classifier.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)

#Accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(Y_test,Y_pred)

#Creating pickle file
pickle_out=open('classifier.pkl','wb')
pickle.dump(classifier,pickle_out)
pickle_out.close()
