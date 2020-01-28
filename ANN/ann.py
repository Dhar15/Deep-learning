#AN EXAMPLE OF ARTIFICIAL NEURAL NETWORK

#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3: 13].values   #get all columns except last one
y = dataset.iloc[:, 13].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X1 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])
labelencoder_X2 = LabelEncoder()
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],remainder='passthrough')
X = ct.fit_transform(X)
X = X[:, 1:] #take one less dummy variable

#Splitting data into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#PART 2 - MAKING THE ANN
#Importing Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialising ANN
classifier = Sequential()
#Adding the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
#Adding another hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
#Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#Compiling ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Fitting the ANN to training set
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)

#PART 3 - MAKING PREDICTIONS AND EVALUATING THE MODEL
#Predicting test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
