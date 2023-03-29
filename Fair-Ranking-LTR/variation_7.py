import pandas as pd
import pickle
import numpy as np
import os
import gc
import pyterrier as pt
from population_stats import variation_scores

def get_var_7_feature_list():
      return ['qual_cat','occupations','gender','relative_pageviews_category']

class MyScorer_7(pt.Transformer):    
    def transform(self, input):   
        count = 0
        data = all_features_var()  
        for index, row in input.iterrows():
            if count%10000==0:
                print((count/35000)*100, '% Documents have been transformed')
            docid = row['docid']
            features = get_var_7_feature_list()
            f_scores = data[get_var_7_feature_list()][data.docid == docid]
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




def all_features_var():
    var_df_path = 'data-models/Data/vardf_7.pkl'


    if not os.path.exists(var_df_path):
        variation_scores(True)
    else:
        var_df = pd.read_hdf(var_df_path, key='df')
        print('Variation scores loaded')


    var_df.loc[var_df['gender']==0.950373] -= 0.2
    var_df.loc[var_df['gender']==0.049519] += 0.15
    var_df.loc[var_df['gender']==0.000109] += 0.05
    return var_df
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

pipeline = pt.FeaturesBatchRetrieve(index, wmodel='BM25', features=["WMODEL:Tf","WMODEL:PL2"])  >> MyScorer_7 #>> pt.text.get_text(train_dataset, get_var_5_feature_list()) % 100
print(pipeline.search('woman'))


"""