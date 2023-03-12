import pandas as pd
import pickle
import numpy as np
import os
import gc
import pyterrier as pt

def get_var_5_feature_list():
      return ['source_subcont_regions','page_subcont_regions']

class MyScorer_5(pt.Transformer):    
    def transform(self, input):   
        computed_path = "data-models/Data/computed_df.pkl"
        df = pd.read_pickle(computed_path)
        data, boost_scores = all_features_var()  
        for index, row in input.iterrows():
            docid = row['docid']
            features = get_var_5_feature_list()
            
            
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
        return input
        #return input.merge(all_features_var(), input, on='docid') # details on how to mergepipelien = bm25 >> MyScorer()


def all_features_var():
    var_5_path = 'data-models/Data/var_2.pkl'
    vardf_5_path = 'data-models/Data/vardf_2.pkl'

    if not os.path.exists(vardf_5_path):
        if not os.path.exists(var_5_path):
            print('Global dataset statistics not available for variaton 5')

            data_path = "data-models/Data/computed_df.pkl"
            stats_path = "data-models/Data/features_stats.pkl"

            # Import all the data and create the feature scores
            df = pd.read_pickle(data_path)
            pop_stats = pd.read_pickle(stats_path)
            print('Dataframe Loaded Successfully!')
        
            
            columns = pop_stats.keys()
            

            var_dict ={}
            for index, row in df.iterrows():
                if index%100000==0:
                    print(index)

                row_dict = {}
                # Add 0 value for Unkown/Not Applicable/ None values
                null = ['Unkown', np.nan,'N/A', None]
                
                
                for f in columns:
                    if f == 'docid':
                        val = row[f]
                        
                    elif row[f] in null:
                        val = 0
                    else:
                        val = pop_stats[f][row[f]]
                    row_dict[f] = val
                var_dict[index] = row_dict
            
        
            f = open(var_5_path,"wb")
            # write the python object (dict) to pickle file
            pickle.dump(var_dict,f)

            # close file
            f.close()
            print('Variation 5 dictionary created and saved!')
            del pop_stats
            del df
            gc.collect()
        else:    
            var_dict = pd.read_pickle(var_5_path)   
            print('Dictionary has been imported')

        chunk_size = 10000  
        
        
        with pd.HDFStore(vardf_5_path, mode='w') as store:

            def dict_chunk(var_dict, chunk_size):
                keys =list(var_dict.keys())
                for i in range(0, len(keys), chunk_size):
                    yield pd.DataFrame.from_dict({k: var_dict[k] for k in keys[i:i+chunk_size]}, orient='index')

            for i, df_chunk in enumerate(dict_chunk(var_dict, chunk_size)):
                store.append('df', df_chunk, format='table',data_columns=True)
        # Free up memory
        del var_dict
        gc.collect()
        var_df = pd.read_hdf(vardf_5_path, key='df')
        print('Variation 5 scores created and loaded')

    else:
        var_df = pd.read_hdf(vardf_5_path, key='df')
        print('Variation 5 scores loaded')

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