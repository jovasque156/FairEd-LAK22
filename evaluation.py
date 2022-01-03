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

#Get the datasets considering X_test, y_test, and A_test
test_datasets = {
    'unaware': X_test_notaware,
    'gender': X_test_aware_gender,
    'public_school': X_test_aware_school,
    'elite': X_test_aware_elite
}

#Applying Preprocessing Pipelines
data_test_prepro = {}
for d in pipes_nominal.keys():
    p_nom = pipes_nominal[d]
    p_num = pipes_numerical[d]
    ds = test_datasets[str.replace(d,'_transf','')]

    prep_d = op.applypreprocessing(ds, p_nom, p_num)
    data_test_prepro[d] = prep_d

#Applying Feature Selection Pipelines
data_test_fs = {}
for d in columns_fs.keys():
    cols = columns_fs[d]
    ds = data_test_prepro['unaware_transf']

    fs_d = op.deletCorrVar(ds, cols)
    data_test_fs[d] = fs_d

#Adding sophisticated-decomposing techniques
cases = ['gender','school', 'elite']

#betAVAE
data_test_betAVAE = {}
for i in cases:
    X_betavae = np.load('dataset/betAVAE/data_test_chilea_betavae_'+i+'.npz')
    data_test_betAVAE[i] = X_betavae

#adversarial
data_test_adv = {}
for i in cases:
    X_adv = np.load('dataset/adversary/data_test_chilea_adv_'+i+'.npz')
    data_test_adv[i] = X_adv

#Get in a dictionary all preprocessed datasets.
datasets_test = {}
for d in data_test_prepro:
    ds = data_test_prepro[d]
    if 'transf' in d:
        datasets_test['transformed_'+d] = ds
    else:
        datasets_test['original_'+d] = ds

for d in data_test_fs:
    ds = data_test_fs[d]
    datasets_test['featureselection_'+d] = ds

#Adding testing to dictionary
for i in data_test_betAVAE:
    datasets_test['betAVAE_'+i] = data_test_betAVAE[i]['x']

for i in data_test_adv:
    datasets_test['adv_'+i] = data_test_adv[i]['x']

#generating y for these datasets
y_test_complex={}
for i in data_train_betAVAE:
    y_test_complex['betAVAE_'+i] = data_test_betAVAE[i]['y']

for i in data_train_adv:
    y_test_complex['adv_'+i] = data_test_adv[i]['y']

#Adding sensitive attributes
A_sens = {}
for i in data_train_betAVAE:
    A_sens['betAVAE_'+i] = data_test_betAVAE[i]['sensitive']

for i in data_train_adv:
    A_sens['adv_'+i] = data_test_adv[i]['sensitive']

results_test_complete = pd.DataFrame(columns = ['model', 'scenario', 'f1_score', 'f1_score_priv','f1_score_unpriv','accuracy', 'accuracy_priv','accuracy_unpriv', 'auc', 'auc_priv', 'auc_unpriv'])
results_test = pd.DataFrame(columns = ['model', 'dataset', 'unaware', 'feature_selection', 'decomposition', 'betAVAE', 'adversarial', 'f1_score', 'accuracy_score', 'auc'])
results_per_group_gender = pd.DataFrame(columns = ['model', 'dataset', 'unaware', 'feature_selection', 'decomposition', 'betAVAE', 'adversarial', 'f1_score', 'f1_score_priv','f1_score_unpriv', 'accuracy','accuracy_priv','accuracy_unpriv', 'auc', 'auc_priv', 'auc_unpriv'])
results_per_group_school = pd.DataFrame(columns = ['model', 'dataset', 'unaware', 'feature_selection', 'decomposition', 'betAVAE', 'adversarial','f1_score',  'f1_score_priv','f1_score_unpriv', 'accuracy','accuracy_priv','accuracy_unpriv', 'auc','auc_priv', 'auc_unpriv'])
results_per_group_elite = pd.DataFrame(columns = ['model', 'dataset', 'unaware', 'feature_selection', 'decomposition', 'betAVAE', 'adversarial','f1_score', 'f1_score_priv','f1_score_unpriv', 'accuracy','accuracy_priv', 'accuracy_unpriv', 'auc','auc_priv', 'auc_unpriv'])

for m in train_models:
    #get grid of a specific model
    grid_model = train_models[m]
    print()
    print('Running for {0} model:'.format(m))

    #predict for each dataset
    for ds in datasets_test:
        if not('original' in ds):
            print("  Evaluating for {0} datasets...".format(ds))
            unaware = 1
            fs = 0
            dec = 0
            betAVAE = 0
            adv = 0

            ds_test = datasets_test[ds]
            estimator_grid = grid_model[ds]
            y_pred = estimator_grid.predict(ds_test)
            y_pred_prob = estimator_grid.predict_proba(ds_test)[:,1]

            #Overall performances
            if ('betAVAE' in ds) or ('adv' in ds):
                y_test_data = y_test_complex[ds]
            else:
                y_test_data = y_test

            f1 = f1_score(y_test_data, estimator_grid.predict(ds_test))
            auc = roc_auc_score(y_test_data, estimator_grid.predict_proba(ds_test)[:, 1])
            acc = accuracy_score(y_test_data, estimator_grid.predict(ds_test))

            #Identifying aware, feature selection and decomposition
            if ('gender' in ds) or ('public_school' in ds) or ('elite' in ds): unaware = 0
            if ('featureselection' in ds) or ('fs' in ds): fs = 1
            if 'decomposed' in ds: dec = 1
            if 'betAVAE' in ds: betAVAE = 1
            if 'adv' in ds: adv = 1

            #Performances per group
            if ('gender' in ds) or ('unaware' in ds):
                print('  Saving results for gender...')

                if betAVAE+adv>=1:
                    A_fairness = A_sens[ds]
                else:
                    A_fairness = A_test['gender']

                f1_gend_f = f1_score(y_test_data[A_fairness==1], y_pred[np.where(A_fairness==1)])
                f1_gend_m = f1_score(y_test_data[A_fairness==0], y_pred[np.where(A_fairness==0)])

                auc_gend_f = roc_auc_score(y_test_data[A_fairness==1], y_pred_prob[np.where(A_fairness==1)])
                auc_gend_m = roc_auc_score(y_test_data[A_fairness==0], y_pred_prob[np.where(A_fairness==0)])

                acc_f = accuracy_score(y_test_data[A_fairness==1], y_pred[np.where(A_fairness==1)])
                acc_m = accuracy_score(y_test_data[A_fairness==0], y_pred[np.where(A_fairness==0)])

                r_gender = {'model': m, 'dataset':ds, 'unaware': unaware,'betAVAE': betAVAE, 'adversarial': adv, 'feature_selection': fs, 'decomposition': dec,'f1_score': f1, 'f1_score_priv':f1_gend_m, 'f1_score_unpriv':f1_gend_f, 'accuracy':acc ,'accuracy_priv': acc_m,'accuracy_unpriv':acc_f , 'auc': auc, 'auc_priv': auc_gend_m, 'auc_unpriv': auc_gend_f}
                results_per_group_gender = results_per_group_gender.append(r_gender, ignore_index = True)


            if ('school' in ds) or ('unaware' in ds):
                print('  Saving results for school...')

                if betAVAE+adv>=1:
                    A_fairness = A_sens[ds]
                else:
                    A_fairness = A_test['public_school']

                f1_gend_p = f1_score(y_test_data[A_fairness==1], y_pred[np.where(A_fairness==1)])
                f1_gend_np = f1_score(y_test_data[A_fairness==0], y_pred[np.where(A_fairness==0)])

                auc_gend_p = roc_auc_score(y_test_data[A_fairness==1], y_pred_prob[np.where(A_fairness==1)])
                auc_gend_np = roc_auc_score(y_test_data[A_fairness==0], y_pred_prob[np.where(A_fairness==0)])

                acc_p = accuracy_score(y_test_data[A_fairness==1], y_pred[np.where(A_fairness==1)])
                acc_np = accuracy_score(y_test_data[A_fairness==0], y_pred[np.where(A_fairness==0)])

                r_school = {'model': m, 'dataset':ds, 'unaware': unaware, 'feature_selection': fs, 'decomposition': dec, 'betAVAE': betAVAE, 'adversarial': adv, 'f1_score': f1, 'f1_score_priv': f1_gend_np, 'f1_score_unpriv':f1_gend_p, 'accuracy':acc ,'accuracy_priv':acc_np ,'accuracy_unpriv': acc_p, 'auc': auc, 'auc_priv': auc_gend_np, 'auc_unpriv': auc_gend_p}
                results_per_group_school = results_per_group_school.append(r_school, ignore_index = True)

            if ('elite' in ds) or ('unaware' in ds):
                print('  Saving results for elite...')

                if betAVAE+adv>=1:
                    A_fairness = A_sens[ds]
                else:
                    A_fairness = A_test['elite']

                f1_elite_p = f1_score(y_test_data[A_fairness==1], y_pred[np.where(A_fairness==1)])
                f1_elite_np = f1_score(y_test_data[A_fairness==0], y_pred[np.where(A_fairness==0)])

                auc_elite_p = roc_auc_score(y_test_data[A_fairness==1], y_pred_prob[np.where(A_fairness==1)])
                auc_elite_np = roc_auc_score(y_test_data[A_fairness==0], y_pred_prob[np.where(A_fairness==0)])

                acc_elite_p = accuracy_score(y_test_data[A_fairness==1], y_pred[np.where(A_fairness==1)])
                acc_elite_np = accuracy_score(y_test_data[A_fairness==0], y_pred[np.where(A_fairness==0)])
                r_elite = {'model': m, 'dataset':ds, 'unaware': unaware, 'feature_selection': fs, 'decomposition': dec, 'betAVAE': betAVAE, 'adversarial': adv, 'f1_score': f1,  'f1_score_priv':f1_elite_np,'f1_score_unpriv':f1_elite_p, 'accuracy':acc ,'accuracy_priv':acc_elite_np ,'accuracy_unpriv': acc_elite_p,'auc': auc,  'auc_priv': auc_elite_np , 'auc_unpriv': auc_elite_p}
                results_per_group_elite = results_per_group_elite.append(r_elite, ignore_index = True)

            if 'transformed' in ds and ('gender' in ds or 'school' in ds or 'elite' in ds):
                scenario=''
                if 'gender' in ds:
                    scenario = 'gender'
                    f1_priv = f1_gend_m
                    f1_unpriv = f1_gend_f
                    acc_priv = acc_m
                    acc_unpriv = acc_f
                    auc_priv = auc_gend_m
                    auc_unpriv = auc_gend_f
                elif 'school' in ds:
                    scenario = 'public_school'
                    f1_priv = f1_gend_np
                    f1_unpriv = f1_gend_p
                    acc_priv = acc_np
                    acc_unpriv = acc_p
                    auc_priv = auc_gend_np
                    auc_unpriv = auc_gend_p
                elif 'elite' in ds:
                    scenario = 'elite'
                    f1_priv = f1_elite_np
                    f1_unpriv = f1_elite_p
                    acc_priv = acc_elite_np
                    acc_unpriv = acc_elite_p
                    auc_priv = auc_elite_np
                    auc_unpriv = auc_elite_p

                r_complete = {'model': m, 'scenario': scenario, 'f1_score': f1, 'f1_score_priv': f1_priv,'f1_score_unpriv': f1_unpriv,'accuracy': acc,'accuracy_priv': acc_priv,'accuracy_unpriv': acc_unpriv, 'auc': auc, 'auc_priv': auc_priv, 'auc_unpriv':auc_unpriv}
                results_test_complete = results_test_complete.append(r_complete, ignore_index = True)

            r = {'model': m, 'dataset': ds, 'unaware': unaware, 'feature_selection': fs, 'decomposition': dec, 'betAVAE': betAVAE, 'adversarial': adv, 'f1_score': f1, 'accuracy_score': acc, 'auc': auc}
            results_test = results_test.append(r, ignore_index = True)

#Fairness Metrics Compuation
results_fairness_gender = pd.DataFrame(columns = ['model', 'dataset', 'fairness_mitigation', 'f1', 'accuracy', 'auc', 'demographic_parity_dif', 'disparate_impact_rate', 'equal_opp_dif', 'equalized_odd_dif', 'sufficiency_dif'])
results_fairness_school = pd.DataFrame(columns = ['model', 'dataset', 'fairness_mitigation', 'f1', 'accuracy', 'auc','demographic_parity_dif', 'disparate_impact_rate', 'equal_opp_dif', 'equalized_odd_dif', 'sufficiency_dif'])
results_fairness_elite = pd.DataFrame(columns = ['model', 'dataset', 'fairness_mitigation', 'f1', 'accuracy', 'auc','demographic_parity_dif', 'disparate_impact_rate', 'equal_opp_dif', 'equalized_odd_dif', 'sufficiency_dif'])

#Get train datasets and sensitive attributes
for m in train_models:
    #get grid of a specific model
    grid_model = train_models[m]
    print()
    print('Running for {0} model:'.format(m))

    for ds in datasets_test:
        if ds != 'betavae_gender' and not('original' in ds):

            print("  Evaluating for {0} datasets...".format(ds))
            ds_test = datasets_test[ds]
            estimator_grid = grid_model[ds]

            #Identify sensitive attribute
            gender = 0
            school = 0
            elite=0
            if ('gender' in ds) or ('unaware' in ds):
                gender = 1
                sens = A_test['gender']
            if ('school' in ds) or ('unaware' in ds):
                school = 1
                sens = A_test['public_school']
            if ('elite' in ds) or ('unaware' in ds):
                elite = 1
                sens = A_test['elite']

            #Identify fairness mitigation
            fairness_mitigation = ''
            if 'betAVAE' in ds:
                fairness_mitigation = 'betAVAE'
                sens = A_sens[ds]
            elif 'adv' in ds:
                fairness_mitigation = 'adv'
                sens = A_sens[ds]
            elif ('fs' in ds or 'featureselection' in ds) and 'decomposed' in ds:
                fairness_mitigation = 'unaware\nfeature selection\ndecomposition'
            elif not(('fs' in ds or 'featureselection' in ds)) and ('decomposed' in ds) and (('gender' in ds) or ('public_school' in ds) or ('elite' in ds)):
                fairness_mitigation = 'decomposition'
            elif 'unaware' in ds and not(('fs' in ds or 'featureselection' in ds) or ('decomposed' in ds)):
                fairness_mitigation = 'unaware'
            elif 'decomposed' in ds and 'unaware' in ds and not('fs' in ds or 'featureselection' in ds):
                fairness_mitigation = 'unaware\ndecomposition'
            elif ('fs' in ds or 'featureselection' in ds) and not('decomposed' in ds):
                fairness_mitigation = 'unaware\nfeature selection'
            elif (not('fs' in ds or 'featureselection' in ds) or not('decomposed' in ds)) and ('gender' in ds) or ('public_school' in ds) or ('elite' in ds):
                fairness_mitigation = 'aware'

            #Get Prediction
            if ('betAVAE' in ds) or ('adv' in ds):
                y_test_data = y_test_complex[ds].reshape(-1,)
            else:
                y_test_data = y_test

            prediction = estimator_grid.predict(ds_test)

            #Getting Overall performances #Overall performances
            f1 = f1_score(y_test_data, prediction)
            auc = roc_auc_score(y_test_data, estimator_grid.predict_proba(ds_test)[:,1])
            acc = accuracy_score(y_test_data, prediction)

            #Evaluate Fairness metrics.
            #Let asume we are evaluating based on demographic_parity_dif
            if gender == 1:
                fairness_perfo_dpd = fm.demographic_parity_dif(prediction, sens, 0)
                fairness_perfo_di = fm.disparate_impact_rate(prediction, sens, 0)-1
                fairness_perfo_eopd = fm.equal_opp_dif(y_test_data, prediction, sens, 0)
                fairness_perfo_eod = fm.equalized_odd_dif(y_test_data, prediction, sens,0)
                fairness_perfo_suff_dif = fm.sufficiency_dif(y_test_data, prediction, sens, 0)

                new_res = {'model': m, 'dataset':ds, 'fairness_mitigation':fairness_mitigation, 'f1':f1, 'accuracy':acc, 'auc':auc, 'demographic_parity_dif':fairness_perfo_dpd, 'disparate_impact_rate':fairness_perfo_di, 'equal_opp_dif':fairness_perfo_eopd, 'equalized_odd_dif':fairness_perfo_eod, 'sufficiency_dif':fairness_perfo_suff_dif}
                results_fairness_gender=results_fairness_gender.append(new_res, ignore_index=True)
            if school == 1:
                fairness_perfo_dpd = fm.demographic_parity_dif(prediction, sens, 0)
                fairness_perfo_di = fm.disparate_impact_rate(prediction, sens, 0)-1
                fairness_perfo_eopd = fm.equal_opp_dif(y_test_data, prediction, sens, 0)
                fairness_perfo_eod = fm.equalized_odd_dif(y_test_data, prediction, sens, 0)
                fairness_perfo_suff_dif = fm.sufficiency_dif(y_test_data, prediction, sens, 0)

                new_res = {'model': m, 'dataset':ds, 'fairness_mitigation':fairness_mitigation, 'f1':f1, 'accuracy':acc, 'auc':auc, 'demographic_parity_dif':fairness_perfo_dpd, 'disparate_impact_rate':fairness_perfo_di, 'equal_opp_dif':fairness_perfo_eopd, 'equalized_odd_dif':fairness_perfo_eod, 'sufficiency_dif':fairness_perfo_suff_dif}
                results_fairness_school=results_fairness_school.append(new_res, ignore_index=True)

            if elite==1:
                fairness_perfo_dpd = fm.demographic_parity_dif(prediction, sens, 0)
                fairness_perfo_di = fm.disparate_impact_rate(prediction, sens, 0)-1
                fairness_perfo_eopd = fm.equal_opp_dif(y_test_data, prediction, sens, 0)
                fairness_perfo_eod = fm.equalized_odd_dif(y_test_data, prediction, sens, 0)
                fairness_perfo_suff_dif = fm.sufficiency_dif(y_test_data, prediction, sens, 0)

                new_res = {'model': m, 'dataset':ds, 'fairness_mitigation':fairness_mitigation, 'f1':f1, 'accuracy':acc, 'auc':auc, 'demographic_parity_dif':fairness_perfo_dpd, 'disparate_impact_rate':fairness_perfo_di, 'equal_opp_dif':fairness_perfo_eopd, 'equalized_odd_dif':fairness_perfo_eod, 'sufficiency_dif':fairness_perfo_suff_dif}
                results_fairness_elite=results_fairness_elite.append(new_res, ignore_index=True)
