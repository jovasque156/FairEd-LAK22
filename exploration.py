#Importing Libraries

#Data Handling
import pandas as pd
import numpy as np
import sklearn
import math

#plotting
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
from IPython.display import Markdown, display
import ipywidgets as widgets
import seaborn as sns
sns.set()


ds = pd.read_csv('dataset/final_dataset.csv', sep = ',', index_col = 0)

#Selecting useful variables
Y = ds.iloc[:,69:73]
A = ds.iloc[:,[0,1]]
X = ds.iloc[:,[2,3]+list(range(8,17))+list(range(19,22))+list(range(24,46))+list(range(47,63))+[64]]

A_dummy = pd.get_dummies(A)
A_dummy = A_dummy.iloc[:,0:2]
A_dummy.columns = ['gender', 'public_school']
A_dummy['gender'] = 1-A_dummy['gender']
A_dummy['elite'] = 1-(1-A_dummy['gender'])*(1-A_dummy['public_school'])
A_dummy

## Let analyze the distribution for Gender

def f(var):
    N = ['male','female']
    drop = np.array(A.loc[Y[var]==1,"Genero"].value_counts()).flatten()
    not_drop = np.array(A.loc[Y[var]==0,"Genero"].value_counts()).flatten()
    plt.bar(N, drop, color = 'r', label='dropout')
    plt.bar(N, not_drop, bottom = drop, color='b', label='not dropout')
    plt.legend()
    plt.show()

l = []
for var in Y.columns:
    l.append((var,var))

interact(f, var=l);

for label in Y.columns:
    res = pd.crosstab(A_dummy['gender'],Y[label])
    print('Y='+label)
    for x in range(len(res)):
        total_des = res.iloc[x,1]
        total_gen = res.iloc[x,:].sum()
        print("For Gender={0}, the rate of dropout is: {1:.2%}".format(x,total_des/total_gen))
    print()
print("Note: 0 is for male, and 1 for female")

## Let analyze for High School type

def f(var):
    N = ['Private','Mix', 'Public']
    drop = np.array(A.loc[Y[var]==1,"Grupo_Dependencia_Colegio_F"].value_counts()).flatten()
    not_drop = np.array(A.loc[Y[var]==0,"Grupo_Dependencia_Colegio_F"].value_counts()).flatten()
    plt.bar(N, drop, color = 'r', label='dropout')
    plt.bar(N, not_drop, bottom = drop, color='b', label='not dropout')
    plt.legend()
    plt.show()

l = []
for var in Y.columns:
    l.append((var,var))

interact(f, var=l);

#Save the numerical and nominal variables
numerical = [0]+list(range(4,14))+[21,23,25,27,29,31,33,34,37,39,41,43,45,47,49,50]

def f(var):
    fig, ax1 = plt.subplots()

    sns.axes_style("darkgrid")
    sns.kdeplot(X[A['Grupo_Dependencia_Colegio_F']=='Municipal'][var], color='g', label = 'Public')
    sns.kdeplot(X[A['Grupo_Dependencia_Colegio_F']=='Particular Subvencionado'][var], color='r', label = 'Mix')
    sns.kdeplot(X[A['Grupo_Dependencia_Colegio_F']=='Particular Pagado'][var], color='b', label = 'Private')


    ax1.legend()

    plt.show()

l = []
for var in X.iloc[:,numerical].columns:
    l.append((var,var))

interact(f, var=l);

def f(var):
    N = ['Not Public', 'Public']
    drop = np.array(A_dummy.loc[Y[var]==1,"public_school"].value_counts()).flatten()
    not_drop = np.array(A_dummy.loc[Y[var]==0,"public_school"].value_counts()).flatten()
    plt.bar(N, drop, color = 'r', label='dropout')
    plt.bar(N, not_drop, bottom = drop, color='b', label='not dropout')
    plt.legend()
    plt.show()

l = []
for var in Y.columns:
    l.append((var,var))

interact(f, var=l);

for label in Y.columns:
    res = pd.crosstab(A_dummy['public_school'],Y[label])
    print('Y='+label)
    for x in list(pd.crosstab(A_dummy['public_school'],Y['Droput']).index):
        total_des = res.loc[x,1]
        total_gen = res.loc[x,:].sum()
        print("For High School type={0}, the rate of dropout is: {1:.2%}".format(x,total_des/total_gen))
    print()

print('Note: 0 is Not Public, and 1 is Public')

## Let analyze for Elite Membership

def f(var):
    N = ['Female or Public', 'Male and Not Public']
    drop = np.array(A_dummy.loc[Y[var]==1,"elite"].value_counts()).flatten()
    not_drop = np.array(A_dummy.loc[Y[var]==0,"elite"].value_counts()).flatten()
    plt.bar(N, drop, color = 'r', label='dropout')
    plt.bar(N, not_drop, bottom = drop, color='b', label='not dropout')
    plt.legend()
    plt.show()

l = []
for var in Y.columns:
    l.append((var,var))

interact(f, var=l);

for label in Y.columns:
    res = pd.crosstab(A_dummy['elite'],Y[label])
    print('Y='+label)
    for x in list(pd.crosstab(A_dummy['elite'],Y['Droput']).index):
        total_des = res.loc[x,1]
        total_gen = res.loc[x,:].sum()
        print("For Elite type={0}, the rate of dropout is: {1:.2%}".format(x,total_des/total_gen))
    print()

print('Note: 0 is Male and Not Public, and 1 is Female or Public')

#Save the numerical and nominal variables
numerical = [0]+list(range(4,14))+[21,23,25,27,29,31,33,34,37,39,41,43,45,47,49,50]

def f(var, sensitive_attr):
    fig, ax1 = plt.subplots()

    if sensitive_attr == 'public_school':
        group1 = 'Public'
        group2 = 'Not Public'
    elif sensitive_attr =='gender':
        group1 = 'Female'
        group2 = 'Male'
    else:
        group1 = 'Female or Public'
        group2 = 'Male and Not Public'


    sns.axes_style("darkgrid")
    sns.kdeplot(X[A_dummy[sensitive_attr]==1][var], color='g', label = group1)
    sns.kdeplot(X[A_dummy[sensitive_attr]==0][var], color='r', label = group2)

    ax1.set_xlabel(var)
    ax1.legend()

    plt.show()

l = []
for var in X.iloc[:,numerical].columns:
    l.append((var,var))

s = [('Gender','gender'), ('School Type','public_school'), ('Elite', 'elite')]

interact(f, var=l, sensitive_attr=s);
