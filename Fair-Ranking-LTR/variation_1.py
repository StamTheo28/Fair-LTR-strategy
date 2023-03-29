import pandas as pd
import pickle
import numpy as np
import os
import gc
import pyterrier as pt
from population_stats import variation_scores

def get_var_1_feature_list():
      return ['qual_cat','source_subcont_regions','occupations','gender','page_subcont_regions','relative_pageviews_category', 'creation_date',
              'years_category']

class MyScorer_1(pt.Transformer):    
    def transform(self, input):   
        count = 0
        scores = get_fairness_scores()  
        print('Scores loaded')
        for index, row in input.iterrows():
            if count%1000==0:
                print(count/100, 'Topics have been transformed')
            docid = row['docid']
            features = get_var_1_feature_list()
            f_scores = scores[get_var_1_feature_list()][scores.docid == docid]
            f_list = []
            for f in features:
                if len(f_scores[f]) == 0:
                    f_list.append(0.0)
                else:
                    f_list.append(float(f_scores[f]))        
            relevance_scores = list(input.iloc[index]['features'])
            combined_scores = relevance_scores + list(f_list)
           
            input.at[index, 'features'] = np.array(combined_scores)
            count+=1
        return input




def get_fairness_scores():
    var_df_path = 'data-models/Data/var_df.pkl'

    if not os.path.exists(var_df_path):
        variation_scores()
    else:
        var_df = pd.read_hdf(var_df_path, key='df')
        print('Variation scores loaded')
    return var_df


