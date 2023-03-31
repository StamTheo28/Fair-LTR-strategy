import pandas as pd
import pickle
import numpy as np
import os
import gc
import pyterrier as pt
from population_stats import variation_scores

def get_var_2_feature_list():
      return ['source_subcont_regions','page_subcont_regions']

# Custom transformer that impements the ASTL strategy
class MyScorer_2(pt.Transformer):    
    def transform(self, input):   
        print(input)
        print(len(input))
        print(input.columns)
        scores = get_fairness_scores()  
        for index, row in input.iterrows():
            docid = row['docid']
            features = get_var_2_feature_list()
            f_scores = scores[get_var_2_feature_list()][scores.docid == docid]
            f_list = []
            for f in features:
                if len(f_scores[f]) == 0:
                    f_list.append(0.0)
                else:
                    f_list.append(float(f_scores[f]))        
            relevance_scores = list(input.iloc[index]['features'])
            combined_scores = relevance_scores + list(f_list)
           
            input.at[index, 'features'] = np.array(combined_scores)
        return input
        #return input.merge(all_features_var(), input, on='docid') # details on how to mergepipelien = bm25 >> MyScorer()


def get_fairness_scores():
    var_df_path = 'data-models/Data/var_df.pkl'

    if not os.path.exists(var_df_path):
        variation_scores()
    else:
        var_df = pd.read_hdf(var_df_path, key='df')
        print('Variation scores loaded')
    return var_df