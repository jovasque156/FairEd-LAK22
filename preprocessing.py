#Importing Libraries

#Data Handling
import pandas as pd
import numpy as np
import sklearn
import math

#Pipelines
import ownpipes as op

#Sampling
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, Y_final, test_size = 0.25, stratify= Y_final, random_state = 123)
A_train = A_dummy.loc[X_train.index,:]
A_test = A_dummy.loc[X_test.index,:]

X_train_notaware = X_train #this is applied for three cases of sensitive attribtue
X_train_aware = pd.concat([X_train, A_train['gender'], A_train['public_school']], axis=1)
X_train_aware_gender = pd.concat([X_train,A_train['gender']],axis=1)
X_train_aware_school = pd.concat([X_train,A_train['public_school']],axis=1)
X_train_aware_elite = pd.concat([X_train,A_train['elite']],axis=1)

#Creating a dictionary for datasets
train_datasets = {
    'unaware': X_train_notaware,
    'aware': X_train_aware,
    'gender': X_train_aware_gender,
    'public_school': X_train_aware_school,
    'elite': X_train_aware_elite
}

for i in idnumerical:
    X_test_aware[X_test_aware.columns[i]] = X_test_aware[X_test_aware.columns[i]].astype('float64')

for i in idnominal:
    X_test_aware[X_test_aware.columns[i]] = X_test_aware[X_test_aware.columns[i]].astype('object')

idnumerical = [0]+list(range(4,14))+[21,23,25,27,29,31,33,34,37,39,41,43,45,47,49,50]
idnominal = list(np.setdiff1d(list(range(0,len(X_train.columns))),idnumerical))

numerical = list(X_train.iloc[:,idnumerical].columns)
nominal = list(X_train.iloc[:,idnominal].columns)
nom_num = [y for x in [numerical, nominal] for y in x]

X_nom = X_train.loc[:,nominal]
X_num = X_train.loc[:,numerical]

imp_nom = SimpleImputer(strategy='most_frequent')
enc = OneHotEncoder(drop='first')
result = imp_nom.fit_transform(X_nom)
result = pd.DataFrame(result, columns = X_nom.columns)
enc.fit_transform(result)

#Preprocesing Pipes
#Creating parameters
normalization = [True, False]

#Dictionary where pickles will be stored
preprocessing_pickles = {}

#Creating pickles for each case
for td in train_datasets:
    print(td)
    data = train_datasets[td]
    idnumerical = [0]+list(range(4,14))+[21,23,25,27,29,31,33,34,37,39,41,43,45,47,49,50]
    idnominal = list(np.setdiff1d(list(range(0,len(data.columns))),idnumerical))

    X, pipe_nom, pipe_num = op.preprocessing(data, idnumerical=idnumerical, idnominal=idnominal, imputation=True, encode = True, normalization = True)
    preprocessing_pickles[td+'_transf'] = (X, pipe_nom, pipe_num)

    X, pipe_nom, pipe_num = op.preprocessing(data, idnumerical=idnumerical, idnominal=idnominal, imputation=True, encode = True, normalization = False)
    preprocessing_pickles[td] = (X, pipe_nom, pipe_num)
