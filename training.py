#Importing Libraries

#Data Handling
import pandas as pd
import numpy as np
import sklearn
import math

#Pipelines
import ownpipes as op
import fairness_metrics as fm
#%cat ownpipes.py
#%cat fairness_metrics.py

#Models
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier

#Get in a dictionary all preprocessed datasets.
datasets = {}
for d in datasets_prepro:
    ds = datasets_prepro[d]
    if 'transf' in d:
        datasets['transformed_'+d] = ds
    else:
        datasets['original_'+d] = ds

for d in datasets_fs:
    ds = datasets_fs[d]
    datasets['featureselection_'+d] = ds

#Adding sophisticated-decomposing techniques
cases = ['gender','school', 'elite']

#betAVAE
#Note: for confidentiality reasons, the database is not shared
data_train_betAVAE = {}
for i in cases:
    X_betavae = np.load('dataset/betAVAE/data_train_chilea_betavae_'+i+'.npz')
    data_train_betAVAE[i] = X_betavae

#adversarial
data_train_adv = {}
for i in cases:
    X_betavae = np.load('dataset/adversary/data_train_chilea_adv_'+i+'.npz')
    data_train_adv[i] = X_betavae

#Adding training to dictionary
for i in data_train_betAVAE:
    datasets['betAVAE_'+i] = data_train_betAVAE[i]['x']

for i in data_train_adv:
    datasets['adv_'+i] = data_train_adv[i]['x']

#generating y for these datasets
y_train_complex={}
for i in data_train_betAVAE:
    y_train_complex['betAVAE_'+i] = data_train_betAVAE[i]['y']

for i in data_train_adv:
    y_train_complex['adv_'+i] = data_train_adv[i]['y']

#Save this in order to identify the different scenarios
datasets={'original_notawared': X_ie,
          'original_genderawared': X_gendawared_ie,
          'original_schoolawared': X_schoolawared_ie,
          'original_gsaware': X_gsawared_ie,
          'transformed_notawared': X_ien,
          'transformed_genderawared': X_gendawared_ien,
          'transformed_schoolawared': X_schoolawared_ien,
          'transformed_gsaware': X_gsawared_ien,
          'featureselection_gender': X_fs_gend_ien,
          'featureselection_school': X_fs_school_ien,
          'featureselection_gs': X_fs_gs_ien,
          'decomposed_notawared': X_ien_svd,
          'decomposed_genderawared':X_gendawared_ien_svd,
          'decomposed_schoolawared':X_schoolawared_ien_svd,
          'decomposed_gsaware': X_gsawared_ien_svd,
          'decomposed_fs_gender': X_fs_gend_ien_svd,
          'decomposed_fs_school':X_fs_school_ien_svd,
          'decomposed_fs_gs':X_fs_gs_ien_svd
         }
## Training Models
#Defining Models and parameters for gid
#SVM
parameters = {'SVM__probability':[True],'SVM__kernel':['rbf', 'sigmoid'], 'SVM__C': [0.01, 0.1, 1], 'SVM__class_weight': ['balanced']}
#parameters = {'SVM__probability':[True], 'SVM__kernel':['rbf', 'sigmoid'], 'SVM__C': [1], 'SVM__class_weight': ['balanced']}
svm = SVC(random_state=0)
svm_res = {}
for ds in ds_total:
    print("Starting for..."+ds+' with '+str(datasets[ds].shape)+'features')
    X = datasets[ds]
    if 'betAVAE' in ds or 'adv' in ds:
        y_train_data = y_train_complex[ds].ravel()
    else:
        y_train_data = y_train

    fit = op.get_grid(X, y_train_data, parameters, svm, 'SVM')
    print("End for "+ds)
    print()

    svm_res[ds] = fit

#Defining Models and parameters for gid
#Logistic Regression
parameters = {'LR__C':[0.01, 0.1, 1], 'LR__fit_intercept':[True, False], 'LR__solver': ['liblinear','lbfgs'], 'LR__class_weight': ['balanced', None], 'LR__max_iter' : [100000]}
lr = LogisticRegression(random_state=0)
lr_res = {}
for ds in ds_total:
    print("Starting for..."+ds+' with '+str(datasets[ds].shape)+'features')
    X = datasets[ds]
    if 'betAVAE' in ds or 'adv' in ds:
        y_train_data = y_train_complex[ds].ravel()
    else:
        y_train_data = y_train

    fit = op.get_grid(X, y_train_data, parameters, lr, 'LR')
    print("End for "+ds)
    print()

    lr_res[ds] = fit

#Defining Models and parameters for gid
#Random Forest
parameters = {'RF__class_weight': ['balanced', None],'RF__n_estimators':[10, 50, 100], 'RF__criterion': ['gini', 'entropy'], 'RF__max_depth': [None, 5, 10, 15]}
rf = RandomForestClassifier(random_state=0)
rf_res = {}
for ds in ds_total:
    print("Starting for..."+ds+' with '+str(datasets[ds].shape)+'features')
    X = datasets[ds]
    if 'betAVAE' in ds or 'adv' in ds:
        y_train_data = y_train_complex[ds].ravel()
    else:
        y_train_data = y_train.ravel()

    fit = op.get_grid(X, y_train_data, parameters, rf, 'RF')
    print("End for "+ds)
    print()

    rf_res[ds] = fit

#Defining Models and parameters for gid
#Decision Tree
parameters = {'DT__class_weight': ['balanced', None],'DT__criterion': ['gini', 'entropy'], 'DT__splitter': ['best', 'random'], 'DT__max_depth': [None, 5, 10, 15]}
dt = DecisionTreeClassifier(random_state=0)
dt_res = {}
for ds in ds_total:
    print("Starting for..."+ds+' with '+str(datasets[ds].shape)+'features')
    X = datasets[ds]
    if 'betAVAE' in ds or 'adv' in ds:
        y_train_data = y_train_complex[ds].ravel()
    else:
        y_train_data = y_train

    fit = op.get_grid(X, y_train_data, parameters, dt, 'DT')
    print("End for "+ds)
    print()

    dt_res[ds] = fit

#Defining Models and parameters for gid
#KNN
parameters = {'KNN__weights': ['uniform', 'distance'],'KNN__n_neighbors': [5, 10, 15, 20, 25, 30]}
knn = KNeighborsClassifier()
knn_res = {}
for ds in ds_total:
    print("Starting for..."+ds+' with '+str(datasets[ds].shape)+'features')
    X = datasets[ds]
    if 'betAVAE' in ds or 'adv' in ds:
        y_train_data = y_train_complex[ds].ravel()
    else:
        y_train_data = y_train

    fit = op.get_grid(X, y_train_data, parameters, knn, 'KNN')
    print("End for "+ds)
    print()

    knn_res[ds] = fit
