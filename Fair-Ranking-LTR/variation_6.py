import pandas as pd
import pickle
import numpy as np
import os
import gc
from scipy.stats import chi2_contingency
import pyterrier as pt
from population_stats import variation_scores

def get_var_6_feature_list():
      return ['creation_date', 'years_category']

# Custom transformer that impements the UACT strategy
class MyScorer_6(pt.Transformer):    
    def transform(self, input):   
        count = 0
        data = get_fairness_scores()  
        for index, row in input.iterrows():
            if count%10000==0:
                print(count/100, 'Topics have been transformed')
            docid = row['docid']
            features = get_var_6_feature_list()
            f_scores = data[get_var_6_feature_list()][data.docid == docid]
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
    
    vardf_path = 'data-models/Data/vardf_6.pkl'
    if not os.path.exists(vardf_path):
        for i, row in var_df.iterrows():
            val = row['years_category']
            if val > 0.20:
                var_df.at[i, 'years_category'] = val-0.1
            elif val != 0.0:
                var_df.at[i, 'years_category'] = val+0.1

        f = open(vardf_path,"wb")
        # write df to file
        pickle.dump(var_df,f)

        # close file
        f.close()
    else:
        var_df = pd.read_pickle(vardf_path)
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

"""