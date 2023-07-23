# -*- coding: utf-8 -*-

import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['NUMEXPR_NUM_THREADS']='1'

import pandas as pd

## Specify paths
base_path = r'S:/A32575/A32575_Server/ML_Pipeline/Python_Workspace/'
data_in_path = os.path.join(base_path,r'data/in/')
data_out_path = os.path.join(base_path,r'data/out/')

# Read input data - train, OOS, OOT, PDV
X_train = pd.read_csv(os.path.join(data_in_path,r"ns3_train.csv"))
X_test = pd.read_csv(os.path.join(data_in_path,r"ns3_test.csv"))
oot= pd.read_csv(os.path.join(data_in_path,r"ns3_oot.csv"))
pdv= pd.read_csv(os.path.join(data_in_path,r"ns3_pdv.csv"))

# Specify variables to be excluded
vars_exc = ['V_F_ACCT_FIC_CUSTOMER_REF_CODE','D_F_FN_ACCT_MIS_DATE']

# Specify primary key in the input data
uniq_key = ['V_F_ACCT_FIC_CUSTOMER_REF_CODE']

# Specify name of target variable
tar = ['dpd0_cbc3']

# Specify the list of algorithms
#method=["xgboost"]
#method=["lightgbm"]
method=["xgboost","catboost","lightgbm"]


# Specify the location of univariate-bivariate output file. 
# All the other output reports will be saved at the same location
loc = os.path.join(data_out_path,r'final_test_run/01_10_2021/univar_bivar_analysis_new.xlsx')

# Import CRAIN Auto-ML pipeline with relevant parameters
#os.chdir(os.path.join(base_path,r"scripts"))
from crain_automl_pipeline import model_development

final_out_test = model_development(uniq_key=uniq_key,dev=X_train,target=tar,vars_exc=vars_exc,
                                   oos=X_test,oot=oot,pdv=pdv,
                                   bad=1,method=method,
                                   loc=loc,IV_min=0.02,NA_cutoff=0.8,
                                   max_feat=50, final_model_elem=25,
                                   run_till_bivariate=False)

#loc = os.path.join(data_out_path,r'ml_pipeline_new_user_hyperparams/univar_bivar_analysis_new.xlsx')
#
#grid_search_param_cat= {'depth':[2,3,4,5],'learning_rate':[0.01,0.03,0.05,0.07,0.1,0.15,0.2],
#                                'n_estimators':[25,50,75,100],'reg_lambda':[1,2,3,4]}
#
#grid_search_param_gbm= {'max_depth':[2,3,4,5],'learning_rate':[0.01,0.03,0.05,0.07,0.1,0.15,0.2],
#                                'n_estimators':[25,50,75,100],'bagging_fraction':[0.5,0.75,1],'reg_lambda':[1,2,3,4]}
#
#grid_search_param_xgb= {'max_depth':[2,3,4,5],'learning_rate':[0.01,0.03,0.05,0.07,0.1,0.15,0.2],
#                    'n_estimators':[25,50,75,100],'colsample_bytree':[0.5,0.75,1],'reg_lambda':[1,2,3,4]}
#
#final_out_test = model_development(uniq_key=uniq_key,dev=X_train,target=tar,vars_exc=vars_exc,
#                                   oos=X_test,oot=oot,pdv=pdv,
#                                   bad=1,psi_num_bins=10,method=method,
#                                   loc=loc,IV_min=0.02,NA_cutoff=0.8,
#                                   max_feat=50, final_model_elem=25,
#                                   run_till_bivariate=False,
#                                   grid_search_param_cat=grid_search_param_cat,
#                                   grid_search_param_gbm=grid_search_param_gbm,
#                                   grid_search_param_xgb=grid_search_param_xgb)

#k=np.quantile(X_train[['R_AMB_ACB']].dropna(),[np.round(i*round(1/5,1),1) for i in range(5+1)])
#k
#print('{0:.15f}'.format(k[0]))
#print('{0:.15f}'.format(k[1]))
#print('{0:.15f}'.format(k[2]))
#print('{0:.15f}'.format(k[3]))
#print('{0:.15f}'.format(k[4]))
#print('{0:.15f}'.format(k[5]))
#
#pd.cut(X_train['R_AMB_ACB'],k,include_lowest=True, precision=10,duplicates='drop').unique()
