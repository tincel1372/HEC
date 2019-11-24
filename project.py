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

#Regression logistique

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
classifiersvm.fit(X_trainlog,y_train)

y_predsvm=classifiersvm.predict(X_testlog)
cmsvm=confusion_matrix(y_test,y_predsvm)
print("Erreur carrée moyenne de test: %.5f"
      % mean_squared_error(y_test, y_predsvm))

# Kernel SVM

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifierker = SVC(kernel = 'rbf', random_state = None)
classifierker.fit(X_trainlog, y_train)

# Predicting the Test set results
y_predker = classifierker.predict(X_testlog)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cmker = confusion_matrix(y_test, y_predker)

print("Erreur carrée moyenne de test: %.5f"
      % mean_squared_error(y_test, y_predker))


#Voir les valeurs aberrantes

print(Homicide['Perpetrator Age'].value_counts())
print(Homicide['Victim Age'].value_counts())
print(Homicide['Weapon'].value_counts())
X["Perpetrator Age"].quantile([0.25,0.5,0.75])


#Sommes nous mieux capable de prédire pour un type d'arme donné ?

#Création de jeux de données donc le weapon est knife,et dont le weapon est handgun
Knife=Homicide[Homicide.Weapon == "Knife"]
Handgun=Homicide[Homicide.Weapon == "Handgun"]

# Séparatiion du jeu de données knife en X et Y
HomicideK=Knife.drop("Crime Solved", axis=1)
Xk=HomicideK.iloc[:,]
Xk=Xk.drop("Record ID", axis=1)

Yk=Knife.iloc[:,10]

labelencoder_Y=LabelEncoder()
Yk=labelencoder_Y.fit_transform(Yk)

# Conversion des variables textuelles en variables catégorielles

Xk["Crime Type"]=labelencoder_Y.fit_transform(Xk["Crime Type"])
Xk["Agency Code"]=labelencoder_Y.fit_transform(Xk["Agency Code"])
Xk["Agency Name"]=labelencoder_Y.fit_transform(Xk["Agency Name"])
Xk["Agency Type"]=labelencoder_Y.fit_transform(Xk["Agency Type"])
Xk["City"]=labelencoder_Y.fit_transform(Xk["City"])
Xk["State"]=labelencoder_Y.fit_transform(Xk["State"])
Xk["Month"]=labelencoder_Y.fit_transform(Xk["Month"])
Xk["Victim Sex"]=labelencoder_Y.fit_transform(Xk["Victim Sex"])
Xk["Victim Race"]=labelencoder_Y.fit_transform(Xk["Victim Race"])
Xk["Victim Ethnicity"]=labelencoder_Y.fit_transform(Xk["Victim Ethnicity"])
Xk["Perpetrator Sex"]=labelencoder_Y.fit_transform(Xk["Perpetrator Sex"])
Xk["Perpetrator Race"]=labelencoder_Y.fit_transform(Xk["Perpetrator Race"])
Xk["Perpetrator Ethnicity"]=labelencoder_Y.fit_transform(Xk["Perpetrator Ethnicity"])
Xk["Relationship"]=labelencoder_Y.fit_transform(Xk["Relationship"])
Xk["Weapon"]=labelencoder_Y.fit_transform(Xk["Weapon"])
Xk["Record Source"]=labelencoder_Y.fit_transform(Xk["Record Source"])



#Convert to a float
Xk['Crime Type'] = Xk['Crime Type'].astype(float)
Xk["Agency Code"]=Xk["Agency Code"].astype(float)
Xk["Agency Name"]=Xk["Agency Name"].astype(float)
Xk["Agency Type"]=Xk["Agency Type"].astype(float)
Xk["City"]= Xk["City"].astype(float)
Xk["State"]=Xk["State"].astype(float)
Xk["Month"]=Xk["Month"].astype(float)
Xk["Victim Sex"]=Xk["Victim Sex"].astype(float)
Xk["Victim Race"]=Xk["Victim Race"].astype(float)
Xk["Victim Ethnicity"]=Xk["Victim Ethnicity"].astype(float)
Xk["Perpetrator Sex"]=Xk["Perpetrator Sex"].astype(float)
Xk["Perpetrator Race"]=Xk["Perpetrator Race"].astype(float)
Xk["Perpetrator Ethnicity"]=Xk["Perpetrator Ethnicity"].astype(float)
Xk["Relationship"]=Xk["Relationship"].astype(float)
Xk["Weapon"]=Xk["Weapon"].astype(float)
Xk["Record Source"]=Xk["Record Source"].astype(float)
Xk["Year"]=Xk["Year"].astype(float)
Xk["Incident"]=Xk["Incident"].astype(float)
Xk["Victim Age"]=Xk["Victim Age"].astype(float)
Xk["Victim Count"]=Xk["Victim Count"].astype(float)
Xk["Perpetrator Count"]=Xk["Perpetrator Count"].astype(float)
Xk["Perpetrator Age"]=Xk["Perpetrator Age"].astype(str)
Xk["Perpetrator Age"]=Xk["Perpetrator Age"].replace(" ",str(101))
Xk["Perpetrator Age"]=Xk["Perpetrator Age"].astype(float)

from sklearn.preprocessing import StandardScaler
Sc_X=StandardScaler()
Xk=Sc_X.fit_transform(Xk)
Xk=Sc_X.fit_transform(Xk) 



X_trainK, X_testK, y_trainK, y_testK = train_test_split(
    Xk, Yk ,test_size=0.2, random_state=1234, shuffle=True)

# Tester le SVM

classifiersvm=SVC(kernel='linear',random_state=None)
classifiersvm.fit(X_trainK,y_trainK)

y_predK=classifiersvm.predict(X_testK)
cmK=confusion_matrix(y_testK,y_predK)
cmK
print("Erreur carrée moyenne de test: %.5f"
      % mean_squared_error(y_testK, y_predK))

# Testons à présent pour Handgun

HomicideH=Handgun.drop("Crime Solved", axis=1)
Xh=HomicideH.iloc[:,]
Xh=Xh.drop("Record ID", axis=1)

Yh=Handgun.iloc[:,10]

labelencoder_Y=LabelEncoder()
Yh=labelencoder_Y.fit_transform(Yh)

# Conversion des variables textuelles en variables catégorielles

Xh["Crime Type"]=labelencoder_Y.fit_transform(Xh["Crime Type"])
Xh["Agency Code"]=labelencoder_Y.fit_transform(Xh["Agency Code"])
Xh["Agency Name"]=labelencoder_Y.fit_transform(Xh["Agency Name"])
Xh["Agency Type"]=labelencoder_Y.fit_transform(Xh["Agency Type"])
Xh["City"]=labelencoder_Y.fit_transform(Xh["City"])
Xh["State"]=labelencoder_Y.fit_transform(Xh["State"])
Xh["Month"]=labelencoder_Y.fit_transform(Xh["Month"])
Xh["Victim Sex"]=labelencoder_Y.fit_transform(Xh["Victim Sex"])
Xh["Victim Race"]=labelencoder_Y.fit_transform(Xh["Victim Race"])
Xh["Victim Ethnicity"]=labelencoder_Y.fit_transform(Xh["Victim Ethnicity"])
Xh["Perpetrator Sex"]=labelencoder_Y.fit_transform(Xh["Perpetrator Sex"])
Xh["Perpetrator Race"]=labelencoder_Y.fit_transform(Xh["Perpetrator Race"])
Xh["Perpetrator Ethnicity"]=labelencoder_Y.fit_transform(Xh["Perpetrator Ethnicity"])
Xh["Relationship"]=labelencoder_Y.fit_transform(Xh["Relationship"])
Xh["Weapon"]=labelencoder_Y.fit_transform(Xh["Weapon"])
Xh["Record Source"]=labelencoder_Y.fit_transform(Xh["Record Source"])



#Convert to a float
Xh['Crime Type'] = Xh['Crime Type'].astype(float)
Xh["Agency Code"]=Xh["Agency Code"].astype(float)
Xh["Agency Name"]=Xh["Agency Name"].astype(float)
Xh["Agency Type"]=Xh["Agency Type"].astype(float)
Xh["City"]= Xh["City"].astype(float)
Xh["State"]=Xh["State"].astype(float)
Xh["Month"]=Xh["Month"].astype(float)
Xh["Victim Sex"]=Xh["Victim Sex"].astype(float)
Xh["Victim Race"]=Xh["Victim Race"].astype(float)
Xh["Victim Ethnicity"]=Xh["Victim Ethnicity"].astype(float)
Xh["Perpetrator Sex"]=Xh["Perpetrator Sex"].astype(float)
Xh["Perpetrator Race"]=Xh["Perpetrator Race"].astype(float)
Xh["Perpetrator Ethnicity"]=Xh["Perpetrator Ethnicity"].astype(float)
Xh["Relationship"]=Xh["Relationship"].astype(float)
Xh["Weapon"]=Xh["Weapon"].astype(float)
Xh["Record Source"]=Xh["Record Source"].astype(float)
Xh["Year"]=Xh["Year"].astype(float)
Xh["Incident"]=Xh["Incident"].astype(float)
Xh["Victim Age"]=Xh["Victim Age"].astype(float)
Xh["Victim Count"]=Xh["Victim Count"].astype(float)
Xh["Perpetrator Count"]=Xh["Perpetrator Count"].astype(float)
Xh["Perpetrator Age"]=Xh["Perpetrator Age"].astype(str)
Xh["Perpetrator Age"]=Xh["Perpetrator Age"].replace(" ",str(101))
Xh["Perpetrator Age"]=Xh["Perpetrator Age"].astype(float)

from sklearn.preprocessing import StandardScaler
Sc_X=StandardScaler()
Xh=Sc_X.fit_transform(Xh)
Xh=Sc_X.fit_transform(Xh) 

X_trainH, X_testH, y_trainH, y_testH = train_test_split(
    Xh, Yh ,test_size=0.2, random_state=1234, shuffle=True)

# Tester le SVM

classifiersvm=SVC(kernel='linear',random_state=None)
classifiersvm.fit(X_trainH,y_trainH)

y_predH=classifiersvm.predict(X_testH)
cmH=confusion_matrix(y_testH,y_predH)
cmH
print("Erreur carrée moyenne de test: %.5f"
      % mean_squared_error(y_testH, y_predH))


#Sommes nous mieux capable de prédire si la victime est un homme ou une femme  ?

#Création de jeux de données donc le weapon est knife,et dont le weapon est handgun
Homicide= Homicide.rename(columns={"Victim Sex":"Victim_Sex"})
Male=Homicide[Homicide.Victim_Sex == "Male"]
Female=Homicide[Homicide.Victim_Sex == "Female"]

# Séparatiion du jeu de données knife en X et Y pour les hommes
HomicideM=Male.drop("Crime Solved", axis=1)
Xm=HomicideM.iloc[:,]
Xm=Xm.drop("Record ID", axis=1)

Ym=Male.iloc[:,10]

labelencoder_Y=LabelEncoder()
Ym=labelencoder_Y.fit_transform(Ym)

# Conversion des variables textuelles en variables catégorielles

Xm["Crime Type"]=labelencoder_Y.fit_transform(Xm["Crime Type"])
Xm["Agency Code"]=labelencoder_Y.fit_transform(Xm["Agency Code"])
Xm["Agency Name"]=labelencoder_Y.fit_transform(Xm["Agency Name"])
Xm["Agency Type"]=labelencoder_Y.fit_transform(Xm["Agency Type"])
Xm["City"]=labelencoder_Y.fit_transform(Xm["City"])
Xm["State"]=labelencoder_Y.fit_transform(Xm["State"])
Xm["Month"]=labelencoder_Y.fit_transform(Xm["Month"])
Xm["Victim_Sex"]=labelencoder_Y.fit_transform(Xm["Victim_Sex"])
Xm["Victim Race"]=labelencoder_Y.fit_transform(Xm["Victim Race"])
Xm["Victim Ethnicity"]=labelencoder_Y.fit_transform(Xm["Victim Ethnicity"])
Xm["Perpetrator Sex"]=labelencoder_Y.fit_transform(Xm["Perpetrator Sex"])
Xm["Perpetrator Race"]=labelencoder_Y.fit_transform(Xm["Perpetrator Race"])
Xm["Perpetrator Ethnicity"]=labelencoder_Y.fit_transform(Xm["Perpetrator Ethnicity"])
Xm["Relationship"]=labelencoder_Y.fit_transform(Xm["Relationship"])
Xm["Weapon"]=labelencoder_Y.fit_transform(Xm["Weapon"])
Xm["Record Source"]=labelencoder_Y.fit_transform(Xm["Record Source"])



#Convert to a float
Xm['Crime Type'] = Xm['Crime Type'].astype(float)
Xm["Agency Code"]=Xm["Agency Code"].astype(float)
Xm["Agency Name"]=Xm["Agency Name"].astype(float)
Xm["Agency Type"]=Xm["Agency Type"].astype(float)
Xm["City"]= Xm["City"].astype(float)
Xm["State"]=Xm["State"].astype(float)
Xm["Month"]=Xm["Month"].astype(float)
Xm["Victim_Sex"]=Xm["Victim_Sex"].astype(float)
Xm["Victim Race"]=Xm["Victim Race"].astype(float)
Xm["Victim Ethnicity"]=Xm["Victim Ethnicity"].astype(float)
Xm["Perpetrator Sex"]=Xm["Perpetrator Sex"].astype(float)
Xm["Perpetrator Race"]=Xm["Perpetrator Race"].astype(float)
Xm["Perpetrator Ethnicity"]=Xm["Perpetrator Ethnicity"].astype(float)
Xm["Relationship"]=Xm["Relationship"].astype(float)
Xm["Weapon"]=Xm["Weapon"].astype(float)
Xm["Record Source"]=Xm["Record Source"].astype(float)
Xm["Year"]=Xm["Year"].astype(float)
Xm["Incident"]=Xm["Incident"].astype(float)
Xm["Victim Age"]=Xm["Victim Age"].astype(float)
Xm["Victim Count"]=Xm["Victim Count"].astype(float)
Xm["Perpetrator Count"]=Xm["Perpetrator Count"].astype(float)
Xm["Perpetrator Age"]=Xm["Perpetrator Age"].astype(str)
Xm["Perpetrator Age"]=Xm["Perpetrator Age"].replace(" ",str(101))
Xm["Perpetrator Age"]=Xm["Perpetrator Age"].astype(float)

from sklearn.preprocessing import StandardScaler
Sc_X=StandardScaler()
Xm=Sc_X.fit_transform(Xm) 

X_trainM, X_testM, y_trainM, y_testM = train_test_split(
    Xm, Ym ,test_size=0.2, random_state=1234, shuffle=True)

# Tester le SVM

classifiersvm=SVC(kernel='linear',random_state=None)
classifiersvm.fit(X_trainM,y_trainM)

y_predM=classifiersvm.predict(X_testM)
cmM=confusion_matrix(y_testM,y_predM)
cmM
print("Erreur carrée moyenne de test: %.5f"
      % mean_squared_error(y_testM, y_predM))

# A présent, testons les résultats pour les femmes

HomicideF=Female.drop("Crime Solved", axis=1)
Xf=HomicideF.iloc[:,]
Xf=Xf.drop("Record ID", axis=1)

Yf=Female.iloc[:,10]

labelencoder_Y=LabelEncoder()
Yf=labelencoder_Y.fit_transform(Yf)

# Conversion des variables textuelles en variables catégorielles

Xf["Crime Type"]=labelencoder_Y.fit_transform(Xf["Crime Type"])
Xf["Agency Code"]=labelencoder_Y.fit_transform(Xf["Agency Code"])
Xf["Agency Name"]=labelencoder_Y.fit_transform(Xf["Agency Name"])
Xf["Agency Type"]=labelencoder_Y.fit_transform(Xf["Agency Type"])
Xf["City"]=labelencoder_Y.fit_transform(Xf["City"])
Xf["State"]=labelencoder_Y.fit_transform(Xf["State"])
Xf["Month"]=labelencoder_Y.fit_transform(Xf["Month"])
Xf["Victim_Sex"]=labelencoder_Y.fit_transform(Xf["Victim_Sex"])
Xf["Victim Race"]=labelencoder_Y.fit_transform(Xf["Victim Race"])
Xf["Victim Ethnicity"]=labelencoder_Y.fit_transform(Xf["Victim Ethnicity"])
Xf["Perpetrator Sex"]=labelencoder_Y.fit_transform(Xf["Perpetrator Sex"])
Xf["Perpetrator Race"]=labelencoder_Y.fit_transform(Xf["Perpetrator Race"])
Xf["Perpetrator Ethnicity"]=labelencoder_Y.fit_transform(Xf["Perpetrator Ethnicity"])
Xf["Relationship"]=labelencoder_Y.fit_transform(Xf["Relationship"])
Xf["Weapon"]=labelencoder_Y.fit_transform(Xf["Weapon"])
Xf["Record Source"]=labelencoder_Y.fit_transform(Xf["Record Source"])

#Convert to a float
Xf['Crime Type'] = Xf['Crime Type'].astype(float)
Xf["Agency Code"]=Xf["Agency Code"].astype(float)
Xf["Agency Name"]=Xf["Agency Name"].astype(float)
Xf["Agency Type"]=Xf["Agency Type"].astype(float)
Xf["City"]= Xf["City"].astype(float)
Xf["State"]=Xf["State"].astype(float)
Xf["Month"]=Xf["Month"].astype(float)
Xf["Victim_Sex"]=Xf["Victim_Sex"].astype(float)
Xf["Victim Race"]=Xf["Victim Race"].astype(float)
Xf["Victim Ethnicity"]=Xf["Victim Ethnicity"].astype(float)
Xf["Perpetrator Sex"]=Xf["Perpetrator Sex"].astype(float)
Xf["Perpetrator Race"]=Xf["Perpetrator Race"].astype(float)
Xf["Perpetrator Ethnicity"]=Xf["Perpetrator Ethnicity"].astype(float)
Xf["Relationship"]=Xf["Relationship"].astype(float)
Xf["Weapon"]=Xf["Weapon"].astype(float)
Xf["Record Source"]=Xf["Record Source"].astype(float)
Xf["Year"]=Xf["Year"].astype(float)
Xf["Incident"]=Xf["Incident"].astype(float)
Xf["Victim Age"]=Xf["Victim Age"].astype(float)
Xf["Victim Count"]=Xf["Victim Count"].astype(float)
Xf["Perpetrator Count"]=Xf["Perpetrator Count"].astype(float)
Xf["Perpetrator Age"]=Xf["Perpetrator Age"].astype(str)
Xf["Perpetrator Age"]=Xf["Perpetrator Age"].replace(" ",str(101))
Xf["Perpetrator Age"]=Xf["Perpetrator Age"].astype(float)


from sklearn.preprocessing import StandardScaler
Sc_X=StandardScaler()
Xf=Sc_X.fit_transform(Xf)


X_trainF, X_testF, y_trainF, y_testF = train_test_split(
    Xf, Yf ,test_size=0.2, random_state=1234, shuffle=True)

# Tester le SVM

classifiersvmF=SVC(kernel='linear',random_state=None)
classifiersvmF.fit(X_trainF,y_trainF)

y_predF=classifiersvmF.predict(X_testF)
cmF=confusion_matrix(y_testF,y_predF)
cmF
print("Erreur carrée moyenne de test: %.5f"
      % mean_squared_error(y_testF, y_predF))



# Sommes nous mieux capable de prédire si un crime sera résolu en  fonction de la race de la victime  ?

#Création de jeux de données donc le weapon est knife,et dont le weapon est handgun
Homicide= Homicide.rename(columns={"Victim Race":"Victim_Race"})
White=Homicide[Homicide.Victim_Race == "White"]
Black=Homicide[Homicide.Victim_Race == "Black"]

# Séparatiion du jeu de données knife en X et Y
HomicideW=White.drop("Crime Solved", axis=1)
Xw=HomicideW.iloc[:,]
Xw=Xw.drop("Record ID", axis=1)

Yw=White.iloc[:,10]

labelencoder_Y=LabelEncoder()
Yw=labelencoder_Y.fit_transform(Yw)

# Conversion des variables textuelles en variables catégorielles

Xw["Crime Type"]=labelencoder_Y.fit_transform(Xw["Crime Type"])
Xw["Agency Code"]=labelencoder_Y.fit_transform(Xw["Agency Code"])
Xw["Agency Name"]=labelencoder_Y.fit_transform(Xw["Agency Name"])
Xw["Agency Type"]=labelencoder_Y.fit_transform(Xw["Agency Type"])
Xw["City"]=labelencoder_Y.fit_transform(Xw["City"])
Xw["State"]=labelencoder_Y.fit_transform(Xw["State"])
Xw["Month"]=labelencoder_Y.fit_transform(Xw["Month"])
Xw["Victim Sex"]=labelencoder_Y.fit_transform(Xw["Victim Sex"])
Xw["Victim_Race"]=labelencoder_Y.fit_transform(Xw["Victim_Race"])
Xw["Victim Ethnicity"]=labelencoder_Y.fit_transform(Xw["Victim Ethnicity"])
Xw["Perpetrator Sex"]=labelencoder_Y.fit_transform(Xw["Perpetrator Sex"])
Xw["Perpetrator Race"]=labelencoder_Y.fit_transform(Xw["Perpetrator Race"])
Xw["Perpetrator Ethnicity"]=labelencoder_Y.fit_transform(Xw["Perpetrator Ethnicity"])
Xw["Relationship"]=labelencoder_Y.fit_transform(Xw["Relationship"])
Xw["Weapon"]=labelencoder_Y.fit_transform(Xw["Weapon"])
Xw["Record Source"]=labelencoder_Y.fit_transform(Xw["Record Source"])



#Convert to a float
Xw['Crime Type'] = Xw['Crime Type'].astype(float)
Xw["Agency Code"]=Xw["Agency Code"].astype(float)
Xw["Agency Name"]=Xw["Agency Name"].astype(float)
Xw["Agency Type"]=Xw["Agency Type"].astype(float)
Xw["City"]= Xw["City"].astype(float)
Xw["State"]=Xw["State"].astype(float)
Xw["Month"]=Xw["Month"].astype(float)
Xw["Victim Sex"]=Xw["Victim Sex"].astype(float)
Xw["Victim_Race"]=Xw["Victim_Race"].astype(float)
Xw["Victim Ethnicity"]=Xw["Victim Ethnicity"].astype(float)
Xw["Perpetrator Sex"]=Xw["Perpetrator Sex"].astype(float)
Xw["Perpetrator Race"]=Xw["Perpetrator Race"].astype(float)
Xw["Perpetrator Ethnicity"]=Xw["Perpetrator Ethnicity"].astype(float)
Xw["Relationship"]=Xw["Relationship"].astype(float)
Xw["Weapon"]=Xw["Weapon"].astype(float)
Xw["Record Source"]=Xw["Record Source"].astype(float)
Xw["Year"]=Xw["Year"].astype(float)
Xw["Incident"]=Xw["Incident"].astype(float)
Xw["Victim Age"]=Xw["Victim Age"].astype(float)
Xw["Victim Count"]=Xw["Victim Count"].astype(float)
Xw["Perpetrator Count"]=Xw["Perpetrator Count"].astype(float)
Xw["Perpetrator Age"]=Xw["Perpetrator Age"].astype(str)
Xw["Perpetrator Age"]=Xw["Perpetrator Age"].replace(" ",str(101))
Xw["Perpetrator Age"]=Xw["Perpetrator Age"].astype(float)

from sklearn.preprocessing import StandardScaler
Sc_X=StandardScaler()
Xw=Sc_X.fit_transform(Xw)
Xw=Sc_X.fit_transform(Xw) 



X_trainW, X_testW, y_trainW, y_testW = train_test_split(
    Xw, Yw ,test_size=0.2, random_state=1234, shuffle=True)

# Tester le SVM

classifiersvm=SVC(kernel='linear',random_state=None)
classifiersvm.fit(X_trainW,y_trainW)

y_predW=classifiersvm.predict(X_testW)
cmW=confusion_matrix(y_testW,y_predW)
cmW
print("Erreur carrée moyenne de test: %.5f"
      % mean_squared_error(y_testW, y_predW))


# Testons à présent lorsque la race est "Black"

# Séparatiion du jeu de données knife en X et Y
HomicideB=Black.drop("Crime Solved", axis=1)
Xb=HomicideB.iloc[:,]
Xb=Xb.drop("Record ID", axis=1)

Yb=Black.iloc[:,10]

labelencoder_Y=LabelEncoder()
Yb=labelencoder_Y.fit_transform(Yb)

# Conversion des variables textuelles en variables catégorielles

Xb["Crime Type"]=labelencoder_Y.fit_transform(Xb["Crime Type"])
Xb["Agency Code"]=labelencoder_Y.fit_transform(Xb["Agency Code"])
Xb["Agency Name"]=labelencoder_Y.fit_transform(Xb["Agency Name"])
Xb["Agency Type"]=labelencoder_Y.fit_transform(Xb["Agency Type"])
Xb["City"]=labelencoder_Y.fit_transform(Xb["City"])
Xb["State"]=labelencoder_Y.fit_transform(Xb["State"])
Xb["Month"]=labelencoder_Y.fit_transform(Xb["Month"])
Xb["Victim Sex"]=labelencoder_Y.fit_transform(Xb["Victim Sex"])
Xb["Victim_Race"]=labelencoder_Y.fit_transform(Xb["Victim_Race"])
Xb["Victim Ethnicity"]=labelencoder_Y.fit_transform(Xb["Victim Ethnicity"])
Xb["Perpetrator Sex"]=labelencoder_Y.fit_transform(Xb["Perpetrator Sex"])
Xb["Perpetrator Race"]=labelencoder_Y.fit_transform(Xb["Perpetrator Race"])
Xb["Perpetrator Ethnicity"]=labelencoder_Y.fit_transform(Xb["Perpetrator Ethnicity"])
Xb["Relationship"]=labelencoder_Y.fit_transform(Xb["Relationship"])
Xb["Weapon"]=labelencoder_Y.fit_transform(Xb["Weapon"])
Xb["Record Source"]=labelencoder_Y.fit_transform(Xb["Record Source"])



#Convert to a float
Xb['Crime Type'] = Xb['Crime Type'].astype(float)
Xb["Agency Code"]=Xb["Agency Code"].astype(float)
Xb["Agency Name"]=Xb["Agency Name"].astype(float)
Xb["Agency Type"]=Xb["Agency Type"].astype(float)
Xb["City"]= Xb["City"].astype(float)
Xb["State"]=Xb["State"].astype(float)
Xb["Month"]=Xb["Month"].astype(float)
Xb["Victim Sex"]=Xb["Victim Sex"].astype(float)
Xb["Victim_Race"]=Xb["Victim_Race"].astype(float)
Xb["Victim Ethnicity"]=Xb["Victim Ethnicity"].astype(float)
Xb["Perpetrator Sex"]=Xb["Perpetrator Sex"].astype(float)
Xb["Perpetrator Race"]=Xb["Perpetrator Race"].astype(float)
Xb["Perpetrator Ethnicity"]=Xb["Perpetrator Ethnicity"].astype(float)
Xb["Relationship"]=Xb["Relationship"].astype(float)
Xb["Weapon"]=Xb["Weapon"].astype(float)
Xb["Record Source"]=Xb["Record Source"].astype(float)
Xb["Year"]=Xb["Year"].astype(float)
Xb["Incident"]=Xb["Incident"].astype(float)
Xb["Victim Age"]=Xb["Victim Age"].astype(float)
Xb["Victim Count"]=Xb["Victim Count"].astype(float)
Xb["Perpetrator Count"]=Xb["Perpetrator Count"].astype(float)
Xb["Perpetrator Age"]=Xb["Perpetrator Age"].astype(str)
Xb["Perpetrator Age"]=Xb["Perpetrator Age"].replace(" ",str(101))
Xb["Perpetrator Age"]=Xb["Perpetrator Age"].astype(float)

from sklearn.preprocessing import StandardScaler
Sc_X=StandardScaler()
Xb=Sc_X.fit_transform(Xb)
Xb=Sc_X.fit_transform(Xb) 



X_trainB, X_testB, y_trainB, y_testB = train_test_split(
    Xb, Yb ,test_size=0.2, random_state=1234, shuffle=True)

# Tester le SVM

classifiersvm=SVC(kernel='linear',random_state=None,probability=True)
classifiersvm.fit(X_trainB,y_trainB)

y_predB=classifiersvm.predict(X_testB)
cmB=confusion_matrix(y_testB,y_predB)
cmB
print("Erreur carrée moyenne de test: %.5f"
      % mean_squared_error(y_testB, y_predB))

# AUC curve
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

probs = classifiersvm.predict_proba(X_testB)
auc = roc_auc_score(y_testB, probs)
print('AUC: %.7f' % auc)
plot_roc_curve(fpr, tpr)

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
pip install mlxtend  
from mlxtend.plotting import plot_decision_regions


# plot SVM


# Select 2 features / variable for the 2D plot that we are going to create.
Xplot=Homicide.iloc[:,12:17]
Xplot=Xplot.drop("Victim Race", axis=1)
Xplot=Xplot.drop("Victim Ethnicity", axis=1)
Xplot=Xplot.drop("Perpetrator Sex", axis=1)
Yplot=Homicide.iloc[:,10]
#Xplot["Victim Sex"]=labelencoder_Y.fit_transform(Xplot["Victim Sex"])
Yplot=labelencoder_Y.fit_transform(Yplot)


Xplot["Perpetrator Age"]=Xplot["Perpetrator Age"].astype(str)
Xplot["Perpetrator Age"]=Xplot["Perpetrator Age"].replace(" ",str(101))
Xplot["Perpetrator Age"]=Xplot["Perpetrator Age"].astype(int)

from sklearn.preprocessing import StandardScaler
Sc_X=StandardScaler()
Xplot=Sc_X.fit_transform(Xplot)

X_trainplot, X_testplot, y_trainplot, y_testplot = train_test_split(
    Xplot, Yplot ,test_size=0.01, random_state=1234, shuffle=True)

#Visualisation

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs


# we create 40 separable points
X_testplot, y_testplot = make_blobs(n_samples=300, centers=2, random_state=6)

# fit the model, don't regularize for illustration purposes
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X_testplot, y_testplot)

plt.scatter(X_testplot[:, 0], X_testplot[:, 1], c=y_testplot, s=30, cmap=plt.cm.Paired)

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.3,
           linestyles=['--', '-', '--'])
# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.savefig('foo.png')



# Most important features


from matplotlib import pyplot as plt
from sklearn import svm


def f_importances(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.savefig('imp.png')


features_names = ['Victim Age', 'Perpetrator Age']
imp=svm.SVC(kernel='linear', C=1, gamma=1)
imp.fit(X_testplot, y_testplot)
f_importances(imp.coef_, features_names)


X_testplot=X_testplot.index.to_numpy()
X_testplot = X_testplot.reshape(-1, 1)
































