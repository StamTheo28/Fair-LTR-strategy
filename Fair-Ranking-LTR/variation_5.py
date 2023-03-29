import pandas as pd
import pickle
import numpy as np
import os
import gc
import pyterrier as pt
from population_stats import variation_scores

def get_var_5_feature_list():
      return ['source_subcont_regions','page_subcont_regions']

class MyScorer_5(pt.Transformer):    
    def transform(self, input):   
        computed_path = "data-models/Data/computed_df.pkl"
        df = pd.read_pickle(computed_path)
        data, boost_scores = get_fairness_scores()  
        features = get_var_5_feature_list()
        count = 0
        print(input)
        for index, row in input.iterrows():
            if count%10000==0:
                print((count/35000)*100, '% Documents have been transformed')
            docid = row['docid']
                    
            doc = df[['source_subcont_regions','page_subcont_regions']][df.docid==docid]
            

            f_scores = data[get_var_5_feature_list()][data.docid == docid]
            f_list = []
            for f in features:
                if len(f_scores[f]) == 0:
                    f_list.append(0.0)
                else:
                    score=f_scores[f]
                    score = 0.0
                    for location in doc[f].values:
                        if boost_scores[location] != 0.0:
                            score += boost_scores[location]
                    
                    f_list.append(float(f_scores[f])+score)        
            relevance_scores = list(input.iloc[index]['features'])
            combined_scores = relevance_scores + list(f_list)
           
            input.at[index, 'features'] = np.array(combined_scores)
            count+=1
        return input
        #return input.merge(all_features_var(), input, on='docid') # details on how to mergepipelien = bm25 >> MyScorer()



def get_fairness_scores():
    var_df_path = 'data-models/Data/var_df.pkl'

    if not os.path.exists(var_df_path):
        variation_scores()
    else:
        var_df = pd.read_hdf(var_df_path, key='df')
        print('Variation scores loaded')
    
    stats_path = "data-models/Data/features_stats.pkl"
    pop_stats = pd.read_pickle(stats_path)
    boost_scores = {}
    f1 = 'page_subcont_regions'
    f2 = 'source_subcont_regions'
    for cat in pop_stats[f1].keys():
        score_1 = pop_stats[f1][cat]
        score_2 = pop_stats[f2][cat]
        if score_1 < 0.05 and score_2 <0.05:
            boost_scores[cat] = abs(score_1-score_2)
        else:
            boost_scores[cat] = 0.0

    return var_df, boost_scores

"""
pt.init()

print('Loading Trec-Fair 2022 dataset')
dataset = pt.get_dataset('irds:trec-fair/2022')
train_dataset = pt.get_dataset('irds:trec-fair/2022/train')
   # Index trec-fair/2022
indexer = pt.IterDictIndexer('./indices/trec-fair_2022', meta={'docno':25})
if not os.path.exists('./indices/trec-fair_2022'):
    index = indexer.index(dataset.get_corpus_iter(), fields=['title', 'text'])
else:
    index =pt.IndexFactory.of('./indices/trec-fair_2022')
# print collection statistics
print(index.getCollectionStatistics().toString())

topics = train_dataset.get_topics('text')
qrels = train_dataset.get_qrels()
    
all_features_var()
# BM25 retrieval model
pipeline = pt.FeaturesBatchRetrieve(index, wmodel='BM25', features=["WMODEL:Tf","WMODEL:PL2"])  >> MyScorer_5 #>> pt.text.get_text(train_dataset, get_var_5_feature_list()) % 100
print(pipeline.search('woman'))
"""