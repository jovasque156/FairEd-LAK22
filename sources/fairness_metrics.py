from sklearn import metrics
import pandas as pd
import numpy as np


#Defining the Grup Fair metrics
def accuracy(y_true, y_pred):
    return metrics.accuracy_score(y_true, y_pred)

def auc(y_true, y_pred):
    return metrics.roc_auc_score(y_true, y_pred)

def f1score(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred, average='macro')

def recall(y_true, y_pred):
    #it returns the recall/TPR
    #it is assumed that positive class is equal to 1

    return y_pred[y_true==1].sum()/y_true.sum()

def fpr(y_true, y_pred):
    #It returns the False Positive Rate
    #it is assumed that positive class is equal to 1

    return y_pred[y_true==0].sum()/(len(y_true)-y_true.sum())

def precision(y_true, y_pred):
    #It returns the precision

    return y_true[y_pred==1].sum()/y_pred.sum()

def selection_rate(y_pred, protected_attr, priv_class):
    #It returns the Selection Ratio for priviliged and unpriviliged group
    #The positive class must be equal to 1, which is used for 'select' the individual
    #Pr(h=1|priv_class=a)

    overall = y_pred.sum()/len(y_pred)
    unpr=y_pred[~(protected_attr==priv_class)].sum()/len(y_pred[~(protected_attr==priv_class)])
    priv=y_pred[protected_attr==priv_class].sum()/len(y_pred[protected_attr==priv_class])

    return overall, unpr, priv

def demographic_parity_dif(y_pred, protected_attr, priv_class):
    #It returns the Statistical Parity Difference considering the prediction
    #It is assumed that positive class is equal to 1
    #Pr(h=1|priv_class=unpriviliged) - Pr(h=1|priv_class=priviliged)

    _, unpr, priv = selection_rate(y_pred, protected_attr, priv_class)

    return unpr-priv

def disparate_impact_rate(y_pred, protected_attr, priv_class):
    #It returns the Disparate Impact Ratio
    #It is assumed that positive class is equal to 1
    # Pr(h=1|priv_class=unpriviliged)/Pr(h=1|priv_class=priviliged)
    # Note that when Disparate Impact Ratio<1, it is considered a negative impact to unpriviliged class
    # This ratio can be compared to a threshold t (most of the time 0.8 or 1.2) in order to identify the presence
    # of disparate treatment.

    _, unpr, priv = selection_rate(y_pred, protected_attr, priv_class)

    return unpr/priv


def equal_opp_dif(y_true, y_pred, protected_attr, priv_class):
    #It returns the Equal Opportunity Difference between the priv and unpriv group
    #This is obtained by substracting the recall/TPR of the priv group to the recall/TPR of the unpriv group

    tpr_priv = recall(y_true[protected_attr==priv_class], y_pred[protected_attr==priv_class])
    tpr_unpriv = recall(y_true[protected_attr!=priv_class], y_pred[protected_attr!=priv_class])

    return tpr_unpriv-tpr_priv

def equalized_odd_dif(y_true, y_pred, protected_attr, priv_class):
    #It returns the Equalizied Odds or Separation
    #The equalized_odd is computed obtaining the Kullback-Leibler divergence D(P || Q) for discrete distributions between two groups

    y_true_priv = y_true[protected_attr==priv_class]
    y_pred_priv = y_pred[protected_attr==priv_class]
    y_true_unpriv = y_true[protected_attr!=priv_class]
    y_pred_unpriv = y_pred[protected_attr!=priv_class]

    tpr_priv = recall(y_true_priv, y_pred_priv)
    tpr_unpriv = recall(y_true_unpriv, y_pred_unpriv)

    fpr_priv = fpr(y_true_priv, y_pred_priv)
    fpr_unpriv = fpr(y_true_unpriv, y_pred_unpriv)

    kl1 = tpr_unpriv * np.log(tpr_unpriv/tpr_priv)+(1-tpr_unpriv)*np.log((1-tpr_unpriv)/(1-tpr_priv))
    kl0 = fpr_unpriv * np.log(fpr_unpriv/fpr_priv)+(1-fpr_unpriv)*np.log((1-fpr_unpriv)/(1-fpr_priv))

    return kl1*y_true.sum() / y_true.shape[0] + kl0*(1-y_true).sum() / y_true.shape[0]



def sufficiency_dif(y_true, y_pred, protected_attr, priv_class):
    #It returns the Sufficiency difference between priv and unpriv groups
    #It is assumed that the positive class is equal to 1
    # Pr(Y=1|h=1,priv_class=unpriv) - Pr(Y=1|h=1,priv_class=priv)

    prec_priv = precision(y_true[protected_attr==priv_class], y_pred[protected_attr==priv_class])
    prec_unpr = precision(y_true[protected_attr!=priv_class], y_pred[protected_attr!=priv_class])

    return prec_unpr-prec_priv
