import pandas as pd
import joblib
import lightgbm as lgbm
import pyterrier as pt
from variation_1 import MyScorer_1, get_var_1_feature_list
from variation_2 import MyScorer_2, get_var_2_feature_list
from variation_3 import MyScorer_3, get_var_3_feature_list
from variation_4 import MyScorer_4, get_var_4_feature_list
from variation_5 import MyScorer_5, get_var_5_feature_list
from variation_6 import MyScorer_6, get_var_6_feature_list
from variation_7 import MyScorer_7, get_var_7_feature_list

model = joblib.load('data-models/Models/ml/lightgbm_model_var_1.joblib')
var_features_list = [get_var_1_feature_list(), get_var_2_feature_list(), get_var_3_feature_list()
                     , get_var_4_feature_list(), get_var_5_feature_list(), get_var_6_feature_list(),
                     get_var_7_feature_list()]

relevance_name_list=['TF', 'PL2']
for i in range(0,8):
    if i==0:
        model = joblib.load('data-models/Models/ml/lightgbm_model_var_1.joblib')
        model.feature_names = relevance_name_list
        
    else:
        model = joblib.load('data-models/Models/ml/lightgbm_model_var_'+str(i)+'.joblib')
        model.feature_names = relevance_name_list + var_features_list[i-1]
    print(model.feature_names)
    print(model.feature_importances_)