#Importing Libraries

#Data Handling
import pandas as pd
import numpy as np
import sklearn
import math

#Pipelines
import ownpipes as op

## Feature Selection pipes
#Analyzing Correlation
sns.kdeplot(abs(pd.DataFrame(datasets_prepro['gender'].toarray()).corr()[114]), color = 'r', label='gender')
sns.kdeplot(abs(pd.DataFrame(datasets_prepro['public_school'].toarray()).corr()[114]), color = 'g', label='school type')
sns.kdeplot(abs(pd.DataFrame(datasets_prepro['elite'].toarray()).corr()[114]), color = 'b', label='both')
plt.xlabel('correlation')
plt.legend()
plt.show()

#Creating dictionary
pipes_fs = {}

#Let define the thershold of p-value: SL
p_threshold=0.05

for sa in ['gender', 'public_school', 'elite']:
    print(sa)
    X, col_sel, min_vals = op.backwardElimination(datasets_prepro['unaware_transf'], A_train[sa], p_threshold)
    pipes_fs[sa] = (X, col_sel, min_vals)

## Decomposition Pipes
#svd = decomposition by using TruncatedSVD
scenarios = ['unaware', 'gender', 'public_school', 'elite']
pipes_decomposition = {}

for s in scenarios:
    ds = datasets_prepro[s+'_transf']
    X, pipe_svd = op.svd_decomposition(ds, thres = 0.95)
    pipes_decomposition[s] = (X, pipe_svd)

for s in scenarios[1:]:
    ds = datasets_fs[s]
    X, pipe_svd = op.svd_decomposition(ds, thres = 0.95)
    pipes_decomposition[s+'_fs'] = (X, pipe_svd)
