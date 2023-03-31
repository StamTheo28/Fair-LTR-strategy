import pyterrier as pt
import os
import pandas as pd
from pyterrier.measures import *
from metric_utils.metrics import get_feature_list
from metric_utils import awrf


def relevance_evaluation(models, model_names, test_topics, qrels, baseline):
    results = pt.Experiment(
            models, 
            test_topics, 
            qrels,
            eval_metrics=['recip_rank', nDCG@10, 'num_rel_ret'],
            names=model_names.keys(),
            baseline=baseline,
            )
    
    return results

def fairness_evaluation(model_names):
    # converting a fairnes category to group binary data
    model_feature_score =[]
    for key in model_names.keys():
    
        model_path = "data-models/Ranked-Results/fairness/"+key+'.pkl'
        if os.path.getsize(model_path)>0:
            res = pd.DataFrame(pd.read_pickle(model_path))

        else:
            print('pickle file is empty')

        score_dict = {}
        score_dict['name'] = key
        
        # Get AWRF metrics
        for feature in get_feature_list():   
            if feature in ['source_subcont_regions','page_subcont_regions']:
            
                feature_df = res[feature].str.join('|').str.get_dummies()
            else:
                feature_df = res[feature].str.join('').str.get_dummies()
            score = awrf.vlambda(feature_df, distance=awrf.subtraction)
            
            score_dict[feature] = score
        model_feature_score.append(score_dict)
            

    fair_df = pd.DataFrame(model_feature_score, index = range(len(model_feature_score)))
    fair_list = fair_df[get_feature_list()]
    fair_df['mean awrf'] = fair_df[get_feature_list()].mean(axis=1)
    return fair_df, fair_list
