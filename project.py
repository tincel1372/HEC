# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 17:46:21 2019

@author: Amine
"""

#Importer les librairies

from functools import reduce
import os
import re 
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import neural_network
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Importer le jeu de données


Homicide=pd.read_csv(('database.csv'),sep=',')
display(Homicide.head())

###
# Statistiques descriptives
###

# Vérifier s'il y a des valeurs nulles 

print (Homicide.isnull().sum())
Homicide.isnull().values.any()
Homicide.ismissing().values.any()

##
#Variables catégorielles
##


# variable Agency type 

print(Homicide['Agency Type'].value_counts())
print (Homicide['Agency Type'].isnull())

# variable Crime type 

print(Homicide['Crime Type'].value_counts())
print (Homicide['Crime Type'].isnull())

# variable Crime solved

print(Homicide['Crime Solved'].value_counts())
print (Homicide['Crime Solved'].isnull())

# variable Victim Sex

print(Homicide['Victim Sex'].value_counts())
print (Homicide['Victim Sex'].isnull())

# variable Victim Race

print(Homicide['Victim Race'].value_counts())
print (Homicide['Victim Race'].isnull())

# variable Victim Ethnicity

print(Homicide['Victim Ethnicity'].value_counts())
print (Homicide['Victim Ethnicity'].isnull())


# variable Perpetrator Sex

print(Homicide['Perpetrator Sex'].value_counts())
print (Homicide['Perpetrator Sex'].isnull())


# variable Perpetrator Sex

print(Homicide['Perpetrator Race'].value_counts())
print (Homicide['Perpetrator Race'].isnull())

# variable Perpetrator Ethnicity

print(Homicide['Perpetrator Ethnicity'].value_counts())
print (Homicide['Perpetrator Ethnicity'].isnull())


# variable Relationship

print(Homicide['Relationship'].value_counts())
print (Homicide['Relationship'].isnull())


# variable Record Source

print(Homicide['Record Source'].value_counts())
print (Homicide['Record Source'].isnull())


###
#Modifier les valeurs des variables
###

#Binariser la variable cible "Crime solved"


Y=Homicide.iloc[:,10]

from sklearn.preprocessing import LabelEncoder
labelencoder_Y=LabelEncoder()
Y=labelencoder_Y.fit_transform(Y)

#Catégoriser les variables explicatives

HomicideX=Homicide.drop("Crime Solved", axis=1)
X=HomicideX.iloc[:,]
X=X.drop("Record ID", axis=1)

X["Crime Type"]=labelencoder_Y.fit_transform(X["Crime Type"])
X["Agency Code"]=labelencoder_Y.fit_transform(X["Agency Code"])
X["Agency Name"]=labelencoder_Y.fit_transform(X["Agency Name"])
X["Agency Type"]=labelencoder_Y.fit_transform(X["Agency Type"])
X["City"]=labelencoder_Y.fit_transform(X["City"])
X["State"]=labelencoder_Y.fit_transform(X["State"])
X["Month"]=labelencoder_Y.fit_transform(X["Month"])
X["Victim Sex"]=labelencoder_Y.fit_transform(X["Victim Sex"])
X["Victim Race"]=labelencoder_Y.fit_transform(X["Victim Race"])
X["Victim Ethnicity"]=labelencoder_Y.fit_transform(X["Victim Ethnicity"])
X["Perpetrator Sex"]=labelencoder_Y.fit_transform(X["Perpetrator Sex"])
X["Perpetrator Race"]=labelencoder_Y.fit_transform(X["Perpetrator Race"])
X["Perpetrator Ethnicity"]=labelencoder_Y.fit_transform(X["Perpetrator Ethnicity"])
X["Relationship"]=labelencoder_Y.fit_transform(X["Relationship"])
X["Weapon"]=labelencoder_Y.fit_transform(X["Weapon"])
X["Record Source"]=labelencoder_Y.fit_transform(X["Record Source"])



#Convert to a float
X['Crime Type'] = X['Crime Type'].astype(float)
X["Agency Code"]=X["Agency Code"].astype(float)
X["Agency Name"]=X["Agency Name"].astype(float)
X["Agency Type"]=X["Agency Type"].astype(float)
X["City"]= X["City"].astype(float)
X["State"]=X["State"].astype(float)
X["Month"]=X["Month"].astype(float)
X["Victim Sex"]=X["Victim Sex"].astype(float)
X["Victim Race"]=X["Victim Race"].astype(float)
X["Victim Ethnicity"]=X["Victim Ethnicity"].astype(float)
X["Perpetrator Sex"]=X["Perpetrator Sex"].astype(float)
X["Perpetrator Race"]=X["Perpetrator Race"].astype(float)
X["Perpetrator Ethnicity"]=X["Perpetrator Ethnicity"].astype(float)
X["Relationship"]=X["Relationship"].astype(float)
X["Weapon"]=X["Weapon"].astype(float)
X["Record Source"]=X["Record Source"].astype(float)
X["Year"]=X["Year"].astype(float)
X["Incident"]=X["Incident"].astype(float)
X["Victim Age"]=X["Victim Age"].astype(float)
X["Victim Count"]=X["Victim Count"].astype(float)
X["Perpetrator Count"]=X["Perpetrator Count"].astype(float)
X["Perpetrator Age"]=X["Perpetrator Age"].astype(str)
X["Perpetrator Age"]=X["Perpetrator Age"].replace(" ",str(101))
X["Perpetrator Age"]=X["Perpetrator Age"].astype(float)

print(np.unique(X["Perpetrator Age"]))
print(np.unique(Homicide["Weapon"]))
print(np.unique(Homicide["Victim Age"]))
print(np.unique(X["Victim Age"]))
print(np.unique(Homicide["Year"]))



###
#Séparer le jeu de données en test train 
###


#Train-test-Split
X_train, X_test, y_train, y_test = train_test_split(
    X, Y ,test_size=0.2, random_state=1234, shuffle=True)


#Modèle réseau Neurones

# Avec classifier

from sklearn.neural_network import MLPClassifier

clf_NNC = MLPClassifier(activation='tanh', alpha= 0.1, batch_size='auto',
              beta_1=0.9, beta_2=0.999, early_stopping=False,
              epsilon=1e-08, hidden_layer_sizes=(100, 50),
              learning_rate='constant', learning_rate_init=0.0005,
              max_iter=1000, momentum=0.1, n_iter_no_change=10,
              nesterovs_momentum=True, power_t=0.5, random_state=None,
              shuffle=True, solver='adam', tol=0.0001,
              validation_fraction=0.0005, verbose=False, warm_start=False) 

clf_NNC.fit(X_train, y_train)                         
y_predict_testNNC=clf_NNC.predict(X_test)
y_predict_trainNNC=clf_NNC.predict(X_train)

print("Erreur carrée moyenne de test: %.3f"
      % mean_squared_error(y_test, y_predict_testNNC))

print("Erreur carrée moyenne d'apprentissage: %.3f"
      % mean_squared_error(y_train, y_predict_trainNNC))

from sklearn.metrics import confusion_matrix
cmNNC=confusion_matrix(y_test,y_predict_testNNC)
 

# Regression logistique

#Feature scaling

from sklearn.preprocessing import StandardScaler
Sc_X=StandardScaler()
X_trainlog=Sc_X.fit_transform(X_train)
X_testlog=Sc_X.fit_transform(X_test) 

#Regression

from sklearn.linear_model import LogisticRegression

classifier=LogisticRegression(random_state=None)
classifier.fit(X_trainlog,y_train)

#Predict the test set results

y_pred=classifier.predict(X_testlog)
y_pred_train=classifier.predict(X_trainlog)
print("Erreur carrée moyenne de test: %.5f"
      % mean_squared_error(y_test, y_pred))

print("Erreur carrée moyenne d'apprentissage: %.5f"
      % mean_squared_error(y_train, y_pred_train))

#Matrice de confusion

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)


# Visualisation des résultats

from matplotlib.colors import ListedColormap
X_set, y_set = X_testlog, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()



# Suport Vector Machine

#Fitting SVM  to the training set

from sklearn.svm import SVC

classifiersvm=SVC(kernel='linear',random_state=None)
classifier.fit(X_trainlog,y_train)

y_predsvm=classifier.predict(X_testlog)
cmsvm=confusion_matrix(y_test,y_predsvm)

print("Erreur carrée moyenne de test: %.5f"
      % mean_squared_error(y_test, y_predsvm))


#Voir les valeurs aberrantes

print(Homicide['Perpetrator Age'].value_counts())
print(Homicide['Victim Age'].value_counts())
print(Homicide['Weapon'].value_counts())

#Sommes nous mieux capable de prédire pour un type d'arme donné ?

Knife=Homicide[Homicide.Weapon == "Knife"]
Handgun=Homicide[Homicide.Weapon == "Handgun"]


HomicideX=Homicide.drop("Crime Solved", axis=1)
Xk=HomicideX.iloc[:,]
Xk=Xk.drop("Record ID", axis=1)

Xk["Crime Type"]=labelencoder_Y.fit_transform(X["Crime Type"])
Xk["Agency Code"]=labelencoder_Y.fit_transform(X["Agency Code"])
Xk["Agency Name"]=labelencoder_Y.fit_transform(X["Agency Name"])
Xk["Agency Type"]=labelencoder_Y.fit_transform(X["Agency Type"])
Xk["City"]=labelencoder_Y.fit_transform(X["City"])
Xk["State"]=labelencoder_Y.fit_transform(X["State"])
Xk["Month"]=labelencoder_Y.fit_transform(X["Month"])
Xk["Victim Sex"]=labelencoder_Y.fit_transform(X["Victim Sex"])
Xk["Victim Race"]=labelencoder_Y.fit_transform(X["Victim Race"])
Xk["Victim Ethnicity"]=labelencoder_Y.fit_transform(X["Victim Ethnicity"])
Xk["Perpetrator Sex"]=labelencoder_Y.fit_transform(X["Perpetrator Sex"])
Xk["Perpetrator Race"]=labelencoder_Y.fit_transform(X["Perpetrator Race"])
Xk["Perpetrator Ethnicity"]=labelencoder_Y.fit_transform(X["Perpetrator Ethnicity"])
Xk["Relationship"]=labelencoder_Y.fit_transform(X["Relationship"])
Xk["Weapon"]=labelencoder_Y.fit_transform(X["Weapon"])
Xk["Record Source"]=labelencoder_Y.fit_transform(X["Record Source"])



#Convert to a float
Xk['Crime Type'] = X['Crime Type'].astype(float)
Xk["Agency Code"]=X["Agency Code"].astype(float)
Xk["Agency Name"]=X["Agency Name"].astype(float)
Xk["Agency Type"]=X["Agency Type"].astype(float)
Xk["City"]= X["City"].astype(float)
Xk["State"]=X["State"].astype(float)
Xk["Month"]=X["Month"].astype(float)
Xk["Victim Sex"]=X["Victim Sex"].astype(float)
Xk["Victim Race"]=X["Victim Race"].astype(float)
Xk["Victim Ethnicity"]=X["Victim Ethnicity"].astype(float)
Xk["Perpetrator Sex"]=X["Perpetrator Sex"].astype(float)
Xk["Perpetrator Race"]=X["Perpetrator Race"].astype(float)
Xk["Perpetrator Ethnicity"]=X["Perpetrator Ethnicity"].astype(float)
Xk["Relationship"]=X["Relationship"].astype(float)
Xk["Weapon"]=X["Weapon"].astype(float)
Xk["Record Source"]=X["Record Source"].astype(float)
Xk["Year"]=X["Year"].astype(float)
Xk["Incident"]=X["Incident"].astype(float)
Xk["Victim Age"]=X["Victim Age"].astype(float)
Xk["Victim Count"]=X["Victim Count"].astype(float)
Xk["Perpetrator Count"]=X["Perpetrator Count"].astype(float)
Xk["Perpetrator Age"]=X["Perpetrator Age"].astype(str)
Xk["Perpetrator Age"]=X["Perpetrator Age"].replace(" ",str(101))
Xk["Perpetrator Age"]=X["Perpetrator Age"].astype(float)




