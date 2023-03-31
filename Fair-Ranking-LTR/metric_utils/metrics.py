import pandas as pd
import os
import numpy as np
from scipy.stats import skew, ttest_rel


def get_feature_list():
      return ['qual_cat','source_subcont_regions','occupations','gender','page_subcont_regions','relative_pageviews_category', 'creation_date',
              'years_category']


def calculate_per(dictionary):
    total = sum(dictionary.values())
    return (pd.Series(dictionary)/total).to_dict()

# Sub-function for calculating skewness
def skewness(ranked_df, feature_list):
    score = []
    for f in feature_list:
        if f in ['source_subcont_regions','page_subcont_regions']:
               
            feature_df = ranked_df[f].str.join('|').str.get_dummies()
        else:
            feature_df = ranked_df[f].str.join('').str.get_dummies()
        
        data = []
        for i in feature_df.columns:
            data.append(sum(feature_df[i]))

        s = skew(np.array(data), bias=False)
        score.append(s)
    return np.mean(np.array(score)), score


# Calculate skewness
def skewness_eval(model_names):
    skew_model = {}
    scores = []
    scores_lists = []
    for key in model_names.keys():  
        model_path = "data-models/Ranked-Results/fairness/"+key+'.pkl'
        if os.path.getsize(model_path)>0:
            res = pd.DataFrame(pd.read_pickle(model_path))
        else:
            print('pickle file is empty')
        s, s_list = skewness(res, get_feature_list())
        scores.append(s)
        scores_lists.append(s_list)
    skew_model['name'] = model_names.keys()
    skew_model['skew_coef'] = scores
    
    skew_df = pd.DataFrame(skew_model, index=np.arange(0,len(model_names)))
    return skew_df, scores_lists

# Perform t-test for fairness metrics
def fairness_t_test(scores,skew_scores, var_model_names, baseline):
    scores_dict= scores.to_dict()
    # Get awrf baseline
    base_list = []
    for feature in get_feature_list():
        base_list.append(scores_dict[feature][baseline])
    awrf_dict ={}
    names = list(var_model_names.keys())

    # Perform awrf t-test
    for i in range(2,9):
        model_list = []

        for feature in get_feature_list():
            val = scores_dict[feature][i]
            if pd.isna(val):
                continue
            else:
                model_list.append(val)
        t_statistic, p_value = ttest_rel(base_list, model_list)
        awrf_dict[names[i]]={'t_stat':t_statistic, 'p_value':p_value}
    awrf_df_lgb = pd.DataFrame(awrf_dict)
    
    # Get skew baseline
    base_skew = skew_scores[baseline]

    # Perform skew t-test
    skewness_dict = {}
    for i in range(2,9):
        model_list=skew_scores[i]
        t_statistic, p_value = ttest_rel(base_skew, model_list)
        skewness_dict[names[i]]={'t_stat':t_statistic, 'p_value':p_value}
    skew_df_lgb = pd.DataFrame(skewness_dict)

    print('AWRF T-statistics')
    print(awrf_df_lgb)
    print('Skewed T-statistics')
    print(skew_df_lgb)
