import pyterrier as pt
from pyterrier.measures import *
from metric_utils.metrics import get_feature_list, skewness
import os
import numpy as np
from metric_utils import awrf
import pandas as pd




def var_skewness_eval(model_names, exp_df):
    skew_model = {}
    scores = []
    for key in model_names.keys():  
        model_path = "data-models/Ranked-Results/"+key+'.pkl'
        if os.path.getsize(model_path)>0:
            res = pd.DataFrame(pd.read_pickle(model_path))
        else:
            print('pickle file is empty')
        
        scores.append(skewness(res, model_names[key]))
    skew_model['name'] = model_names.keys()
    skew_model['skew_coef'] = scores
    
    skew_df = pd.DataFrame(skew_model, index=np.arange(0,len(model_names)))
    return pd.merge(exp_df, skew_df, on='name')

def base_skewness_eval(model_names, exp_df, base=True):
    skew_model = {}
    scores = []
    for key in model_names: 
        if base: 
            model_path = "data-models/Ranked-Results/"+key+'.pkl'
        else:
            model_path = "data-models/Ranked-Results/exp_"+key+'.pkl'
        if os.path.getsize(model_path)>0:
            res = pd.DataFrame(pd.read_pickle(model_path))
        else:
            print('pickle file is empty')
        
        scores.append(skewness(res, get_feature_list()))
    skew_model['name'] = model_names
    skew_model['skew_coef'] = scores
    
    skew_df = pd.DataFrame(skew_model, index=np.arange(0,len(model_names)))
    return pd.merge(exp_df, skew_df, on='name')



def variation_evaluation(models,model_names,test_topics, qrels):
    results = pt.Experiment(
    models, 
    test_topics, qrels,
    eval_metrics=['recip_rank','map', nDCG@10],
    names=model_names.keys(),
    #baseline=0,
    )
    print(results)
    #skewed_metrics(res, get_feature_list(), feature_stats)   

    # converting a fairnes category to group binary data
    model_feature_score =[]
    for key in model_names.keys():
    
        model_path = "data-models/Ranked-Results/"+key+'.pkl'
        if os.path.getsize(model_path)>0:
            res = pd.DataFrame(pd.read_pickle(model_path))

        else:
            print('pickle file is empty')

        score_dict = {}
        score_dict['name'] = key
        
        # Get AWRF metrics
        for feature in model_names[key]:   
            if feature in ['source_subcont_regions','page_subcont_regions']:
               
                feature_df = res[feature].str.join('|').str.get_dummies()
            else:
                feature_df = res[feature].str.join('').str.get_dummies()
            score = awrf.vlambda(feature_df, distance=awrf.subtraction)
            score_dict[feature] = score
        model_feature_score.append(score_dict)
    fair_df = pd.DataFrame(model_feature_score, index = [0,1,2,3])
    fair_df['model_awrf'] = fair_df[get_feature_list()].mean(axis=1)
    metrics_df = pd.merge(results, fair_df[['name','model_awrf']], on='name')

    return var_skewness_eval(model_names, metrics_df)

def baseline_evaluation(models,model_names,test_topics, qrels, base=True):
    results = pt.Experiment(
    models, 
    test_topics, qrels,
    eval_metrics=['recip_rank','map', nDCG@10],
    names=model_names,
    #baseline=0,
    )
    
    #skewed_metrics(res, get_feature_list(), feature_stats)   

    # converting a fairnes category to group binary data
    model_feature_score =[]
    model_skewed_score = {}
    for i in range(len(models)):
        if base:
            model_path = "data-models/Ranked-Results/"+model_names[i]+'.pkl'
        else:
            model_path = "data-models/Ranked-Results/exp_"+model_names[i]+'.pkl'
        if os.path.getsize(model_path)>0:
            res = pd.DataFrame(pd.read_pickle(model_path))
             
        else:
            print('pickle file is empty')

        score_dict = {}
        score_dict['name'] = model_names[i]
        # Get Skewness metrics
        model_skewed_score['name'] = model_names[i]
       # model_skewed_score['skewed_fair'] = skewed_metrics(res, feature_stats)
        

        # Get AWRF metrics
        for feature in get_feature_list(): 
            if feature in ['source_subcont_regions','page_subcont_regions']:
               
                feature_df = res[feature].str.join('|').str.get_dummies()
            else:
                feature_df = res[feature].str.join('').str.get_dummies()
            score = awrf.vlambda(feature_df, distance=awrf.subtraction)
            score_dict[feature] = score
        model_feature_score.append(score_dict)
    fair_df = pd.DataFrame(model_feature_score, index = [0,1,2,3])
    #skewed_df = pd.DataFrame(model_skewed_score, index = [0,1,2,3])
    fair_df['model_awrf'] = fair_df[get_feature_list()].mean(axis=1)
    metrics_df = pd.merge(results, fair_df[['name','model_awrf']], on='name')
    #print(skewed_df)
    
   # metrics_df = pd.merge(metrics_df, skewed_df)
    return base_skewness_eval(model_names, metrics_df, base)

