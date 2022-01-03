#Data Handling
import pandas as pd
import numpy as np
import sklearn

#Pipelines
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV

#Transformation
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler

#Imputation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer

#Decomposition
from sklearn.decomposition import TruncatedSVD

#Feature Selection
import statsmodels.api as sm


def deletCorrVar(x, del_col_per_it):
    #del_col_per_it contains the variable to delete for each iterations.
    #the function returns the x with after delete variables in del_col_per_it.

    x=pd.DataFrame(x.toarray()).values

    for i in del_col_per_it:
        x = np.delete(x,i,1)
    return x

def backwardElimination(x, Y, sl):
    #Receives dataset x, dependent variable Y, and the pvalue threshold of sl
    #Returns the set of variables which does not allow to "recreate" the variable

    x = x.toarray()
    Y = Y.values.astype('int64')

    del_col_per_it = []
    min_pvals = []
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        minVar = min(regressor_OLS.pvalues).astype(float)
        if minVar <= sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == minVar):
                    x = np.delete(x, j, 1)
                    del_col_per_it.append(j)
                    min_pvals.append((j,minVar))
                    numVars -=1

    regressor_OLS.summary()
    return x, del_col_per_it, min_pvals

def applypreprocessing(X, idnumerical=None, idnominal=None, nompipe, numpipe):
    numerical = list(X.iloc[:,idnumerical].columns)
    nominal = list(X.iloc[:,idnominal].columns)
    nom_num = [y for x in [numerical, nominal] for y in x]

    #Identifying numerical and nominal variables
    X_nom = X.loc[:,nominal]

    #Numerical
    X_num = X.loc[:,numerical]

    #Apply trained pipes
    X_nom = nompipe.transform(X_nom)
    X_num = numpipe.transform(X_num)
    X_sparse = hstack((X_num, X_nom))

    return X_sparse

def preprocessing(X, idnumerical=None, idnominal=None, imputation=True, encode = True, normalization = True ):
    #Return a sparse matrix using X as a train dataset for fitting estimators.
    #Additionally it is returned the fitted pipelines related to numerical and nominal features.
    numerical = list(X.iloc[:,idnumerical].columns)
    nominal = list(X.iloc[:,idnominal].columns)
    nom_num = [y for x in [numerical, nominal] for y in x]

    X_nom = X.loc[:,nominal]
    X_num = X.loc[:,numerical]

    #Applying estimators for nominal an numerical
    #nominal
    estimators = []
    if imputation == True:
        imp_nom = SimpleImputer(strategy='most_frequent')
        estimators.append(('imputation', imp_nom))
    if encode == True:
        enc = OneHotEncoder(drop='first')
        estimators.append(('encoding', enc))
    if normalization == True:
        unit_norm = Normalizer()
        estimators.append(('unit_normalization', unit_norm))
    pipe_nom = Pipeline(estimators)
    pipe_nom.fit(X_nom)

    #numerical
    imp_num = IterativeImputer(max_iter=100, random_state=1)
    scale=StandardScaler()
    estimators = []
    if imputation == True:
        estimators.append(('impuation', imp_num))
    if normalization == True:
        estimators.append(('scale', scale))
    pipe_num = Pipeline(estimators)
    pipe_num.fit(X_num)

    #Merge both transformations
    X_nom = pipe_nom.transform(X_nom)
    X_num = pipe_num.transform(X_num)
    X_sparse = hstack((X_num, X_nom))
    #improve this by putting numerical first and the nominal

    return X_sparse, pipe_nom, pipe_num


def svd_decomposition(X, thres = 0.95):
    #Return a sparse matrix using X as a train dataset for fitting estimators.
    #Additionally it is returned the fitted pipelines related the decomposition of X.

    svd_dec = TruncatedSVD(n_components=X.shape[1]-1, n_iter=100, random_state=1)
    svd_dec.fit(X)

    #Determining the number of components
    n_comp = 0
    tota_var = 0
    i=0
    while tota_var<thres:
        n_comp += 1
        tota_var += svd_dec.explained_variance_ratio_[i]
        i+=1

    svd_dec = TruncatedSVD(n_components=n_comp, n_iter=100, random_state=1)
    X_dec = svd_dec.fit_transform(X)

    return X_dec, svd_dec

def applysvd_decomposition(X, svd_estimator):
    return svd_estimator.transform(X)

def import_pickle(directory):
    with open(directory, 'rb') as f:
        p = pickle.load(f)

    return p

def get_grid(X, y, parameters, model, model_name, refit = 'f1'):
    pipe_model_train = Pipeline([(model_name, model)])
    grid = GridSearchCV(pipe_model_train,param_grid=parameters, cv=5, verbose = 0, scoring = ['accuracy', 'roc_auc','f1'], refit=refit)

    fit = grid.fit(X,y)

    return fit
