# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 09:56:01 2021

@author: A32575
"""


import os
#import sys
import pandas as pd
import numpy as np
#import catboost, lightgbm
import xgboost as xgb
import pickle


base_path = r'S:/A32575/A32575_Server/ML_Pipeline/Python_Workspace/'
data_in_path = os.path.join(base_path,r'data/in/')
data_out_path = os.path.join(base_path,r'data/out/')

#os.chdir(os.path.join(base_path,r"scripts"))
from crain_automl_pipeline import (kstable,
                                      univariate_bivariate,
                                      create_psi_table)


#print(data_out_path)

## START | User inputs

# whether provided variables are original variables OR one hot encoded variables
# Original = True, One hot encoded = False
raw_vars_flag = False

#train_path = os.path.join(data_in_path,r"ns3_train.csv")
#oos_path = os.path.join(data_in_path,r"ns3_test.csv")
#oot_path = os.path.join(data_in_path,r"ns3_oot.csv")
#pdv_path = os.path.join(data_in_path,r"ns3_pdv.csv")

pred_prob_file_path = os.path.join(data_out_path,r'prob_decimals_test/Base_data/xgboost_singleparams/Pred_Prob_Report_xgboost.xlsx')

#pred_prob_file_path = os.path.join(data_out_path,r'Rutuja_30_09_2021/new/Pred_Prob_Report_xgboost.xlsx')


## Final shortlisted variables
#final_vars = ['MEAN_UTILIZATION_L12_EOM',
#                'SFR_1_L3',
#                'R_CNT_ECS_SI_L12',
#                'SFR_1_L9',
#                'SFR_L3',
#                'Final_tag_SOYABEAN',
#                'tot_deb_cred_txn_Q5',
#                'sd_Investment_Debit_amt',
#                'STD_AHB',
#                'SFR_1_L15']


final_vars = ["R_CHARGES_AMB_L12",
                "LOG_AQB",	
                "SLOPE_R_CASH_CARD_L6",	
                "INVESTMENTS",	
                "R_AMB_ACB",	
                "NUM_CNT_TRANS_INCR25_L12"]

#method
method=["xgboost"]

# Unique key
uniq_key = ['V_F_ACCT_FIC_CUSTOMER_REF_CODE']
#uniq_key = ['V_F_ACCT_FIC_CUSTOMER_REF_CODE_x']
#target var
tar = ['dpd0_cbc3']
#tar = ['bad_flag_rev']

# trained model path
model_path = os.path.join(data_out_path,r'prob_decimals_test/Base_data/xgboost_singleparams/final_model_xgboost.sav')
#model_path = os.path.join(data_out_path,r'Rutuja_30_09_2021/new/final_model_xgboost.sav')

## END | User inputs


## Input data
train = pd.read_excel(pred_prob_file_path,sheet_name='TRAIN')
oos = pd.read_excel(pred_prob_file_path,sheet_name='OOS')
oot= pd.read_excel(pred_prob_file_path,sheet_name='OOT')
pdv= pd.read_excel(pred_prob_file_path,sheet_name='PDV')


Y_train = train[tar]
Y_oos = oos[tar]
Y_oot = oot[tar]
Y_pdv = pdv[tar]


# Based on raw_vars_flag variable, create one hot encoded variables

if raw_vars_flag == True:
    X_train_2 = pd.get_dummies(train[final_vars])
    X_oos_2 = pd.get_dummies(oos[final_vars])
    X_oot_2 = pd.get_dummies(oot[final_vars])
    X_pdv_2 = pd.get_dummies(pdv[final_vars])
else:
    X_train_2 = train[final_vars]
    X_oos_2 = oos[final_vars]
    X_oot_2 = oot[final_vars]
    X_pdv_2 = pdv[final_vars]    

train_2 = X_train_2.merge(Y_train,left_index=True,right_index=True)
oos_2 = X_oos_2.merge(Y_oos,left_index=True,right_index=True)
oot_2 = X_oot_2.merge(Y_oot,left_index=True,right_index=True)
pdv_2 = X_pdv_2.merge(Y_pdv,left_index=True,right_index=True)


# Load the trained model
boost_model = pickle.load(open(model_path,'rb'))


## Feature importance report
feat_imp = pd.DataFrame(list(zip(X_train_2.columns,boost_model.feature_importances_)))
feat_imp = feat_imp.sort_values(by=1,ascending=False)
feat_imp.to_csv(os.path.join(data_out_path,r'prob_decimals_test/Base_data/xgboost_singleparams/reports_outside_psi_missing_cat/feat_imp_outside.csv'),index=False)

## Predicted probability report
tr_pred = pd.DataFrame(data=boost_model.predict_proba(X_train_2)[:,1],
                        columns = ['PRED_PROB'])
tr_pred.iloc[:,0] = tr_pred.iloc[:,0].apply(lambda x: np.round(x,9))

train_pred_report = train[uniq_key].merge(X_train_2,left_index=True,right_index=True)\
                                    .merge(tr_pred,how='left',left_index=True,right_index=True)  

oos_pred = pd.DataFrame(data=boost_model.predict_proba(X_oos_2)[:,1],
                        columns = ['PRED_PROB'])
oos_pred.iloc[:,0] = oos_pred.iloc[:,0].apply(lambda x: np.round(x,9))

oos_pred_report = oos[uniq_key].merge(X_oos_2,left_index=True,right_index=True)\
                                    .merge(oos_pred,how='left',left_index=True,right_index=True)  

oot_pred = pd.DataFrame(data=boost_model.predict_proba(X_oot_2)[:,1],
                        columns = ['PRED_PROB'])
oot_pred.iloc[:,0] = oot_pred.iloc[:,0].apply(lambda x: np.round(x,9))

oot_pred_report = oot[uniq_key].merge(X_oot_2,left_index=True,right_index=True)\
                                    .merge(oot_pred,how='left',left_index=True,right_index=True)  

pdv_pred = pd.DataFrame(data=boost_model.predict_proba(X_pdv_2)[:,1],
                        columns = ['PRED_PROB'])
pdv_pred.iloc[:,0] = pdv_pred.iloc[:,0].apply(lambda x: np.round(x,9))

pdv_pred_report = pdv[uniq_key].merge(X_pdv_2,left_index=True,right_index=True)\
                                    .merge(pdv_pred,how='left',left_index=True,right_index=True)  


pred_report_loc = os.path.join(data_out_path,r'prob_decimals_test/Base_data/xgboost_singleparams/reports_outside_psi_missing_cat/Pred_Prob_Report_outside.xlsx')
with pd.ExcelWriter(pred_report_loc) as writer:  # doctest: +SKIP
    train_pred_report.to_excel(writer, sheet_name='TRAIN',index=False)
    if oos is not None:
        oos_pred_report.to_excel(writer, sheet_name='OOS',index=False)
    if oot is not None:
        oot_pred_report.to_excel(writer, sheet_name='OOT',index=False)
    if pdv is not None:
        pdv_pred_report.to_excel(writer, sheet_name='PDV',index=False)
#train_pred_report.to_csv(os.path.join(data_out_path,r'ml_enhancements_new/XGBOOST/reports_outside_2/train_pred_report_outside.csv'),index=False)

# KS Report
ks_tr,k_bin = kstable(tr_pred,Y_train,bad=1)  
ks_tr['Dataset'] = 'TRAIN'

ks_oos = kstable(oos_pred,Y_oos,1,k_bin=k_bin)
ks_oos['Dataset'] = 'OOS'
                
ks_oot = kstable(oot_pred,Y_oot,1,k_bin=k_bin)
ks_oot['Dataset'] = 'OOT' 

ks_pdv = kstable(pdv_pred,Y_pdv,1,k_bin=k_bin)
ks_pdv['Dataset'] = 'PDV' 
  

if pdv is not None:                                      
    #final_ks_report=score.append([score1,blank,score2,blank,score3])
    final_ks_report=ks_tr.append([ks_oos,ks_oot,ks_pdv])
else:
    
    final_ks_report=ks_tr.append([ks_oos,ks_oot])
                
                
final_ks_report.to_csv(os.path.join(data_out_path,r'prob_decimals_test/Base_data/xgboost_singleparams/reports_outside_psi_missing_cat/final_ks_report_outside.csv'),index=False)


## Univar-Bivariate Report
loc_unibi_final_vars = os.path.join(data_out_path,r'prob_decimals_test/Base_data/xgboost_singleparams/reports_outside_psi_missing_cat/Univar_Bivar_Report_Final_Vars_TRAIN.xlsx')
data_final_feat = pd.concat([X_train_2,
                             Y_train], axis=1)
univariate_bivariate(data_final_feat,tar,[],
                     loc_unibi_final_vars,
                     None)

loc_unibi_final_vars_oos = os.path.join(data_out_path,r'prob_decimals_test/Base_data/xgboost_singleparams/reports_outside_psi_missing_cat/Univar_Bivar_Report_Final_Vars_OOS.xlsx')
data_final_feat_oos = pd.concat([X_oos_2,
                             Y_oos], axis=1)
univariate_bivariate(data_final_feat_oos,tar,[],
                     loc_unibi_final_vars_oos,
                     None)

loc_unibi_final_vars_oot = os.path.join(data_out_path,r'prob_decimals_test/Base_data/xgboost_singleparams/reports_outside_psi_missing_cat/Univar_Bivar_Report_Final_Vars_OOT.xlsx')
data_final_feat_oot = pd.concat([X_oot_2,
                             Y_oot], axis=1)
univariate_bivariate(data_final_feat_oot,tar,[],
                     loc_unibi_final_vars_oot,
                     None)

loc_unibi_final_vars_pdv = os.path.join(data_out_path,r'prob_decimals_test/Base_data/xgboost_singleparams/reports_outside_psi_missing_cat/Univar_Bivar_Report_Final_Vars_PDV.xlsx')
data_final_feat_pdv = pd.concat([X_pdv_2,
                             Y_pdv], axis=1)
univariate_bivariate(data_final_feat_pdv,tar,[],
                     loc_unibi_final_vars_pdv,
                     None)

## Correlation matrix report

numeric_col_list = [x for x in X_train_2.columns if np.issubdtype(X_train_2[x].dtype,np.number)]

cor_mat_train = X_train_2[numeric_col_list].corr()
cor_mat_oos = X_oos_2[numeric_col_list].corr()
cor_mat_oot = X_oot_2[numeric_col_list].corr()
cor_mat_pdv = X_pdv_2[numeric_col_list].corr()

corr_report_loc = os.path.join(data_out_path,r'prob_decimals_test/Base_data/xgboost_singleparams/reports_outside_psi_missing_cat/Corr_Matrix_Report_outside.xlsx')
with pd.ExcelWriter(corr_report_loc) as writer:  # doctest: +SKIP
    cor_mat_train.to_excel(writer, sheet_name='TRAIN',index=False)
    cor_mat_oos.to_excel(writer, sheet_name='OOS',index=False)
    cor_mat_oot.to_excel(writer, sheet_name='OOT',index=False)
    cor_mat_pdv.to_excel(writer, sheet_name='PDV',index=False)

## PSI Report
psi_table = create_psi_table(train_2,oos_2,oot_2,pdv_2,final_vars,tar,5,pdv_present_flag=True)

psi_cols = ['VAR_NAME','PSI_OOS','PSI_OOT','PSI_PDV']
psi_agg_df = psi_table[psi_cols]
psi_agg_df = psi_agg_df.groupby(['VAR_NAME'],as_index=False).agg({'PSI_OOS':'sum',
                                                             'PSI_OOT':'sum','PSI_PDV':'sum'})
loc_psi = os.path.join(data_out_path,r'prob_decimals_test/Base_data/xgboost_singleparams/reports_outside_psi_missing_cat/psi_table_report_outside.xlsx')
with pd.ExcelWriter(loc_psi) as writer:  # doctest: +SKIP
    psi_table.to_excel(writer, sheet_name='PSI_BIN_LEVEL',index=False)   
    psi_agg_df.to_excel(writer, sheet_name='PSI_VAR_LEVEL',index=False) 
 


