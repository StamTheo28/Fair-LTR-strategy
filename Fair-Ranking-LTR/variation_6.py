import pandas as pd
import pickle
import numpy as np
import os
import gc
from scipy.stats import chi2_contingency
import pyterrier as pt

def get_var_6_feature_list():
      return ['creation_date', 'years_category']

class MyScorer_6(pt.Transformer):    
    def transform(self, input):   
        count = 0
        data = all_features_var()  
        for index, row in input.iterrows():
            if count%10000==0:
                print(count)
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




def all_features_var():
    var_6_path = 'data-models/Data/var_3.pkl'
    vardf_6_path = 'data-models/Data/vardf_3.pkl'
    data_path = "data-models/Data/computed_df.pkl"
    stats_path = "data-models/Data/features_stats.pkl"

    if not os.path.exists(vardf_6_path):
        if not os.path.exists(var_6_path):
            print('Global dataset statistics not available for variation 3')



            # Import all the data and create the feature scores
            df = pd.read_pickle(data_path)
            pop_stats = pd.read_pickle(stats_path)
            print('Dataframe Loaded Successfully!')
            
            columns = pop_stats.keys()
            print(columns)

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
            
        
            f = open(var_6_path,"wb")
            # write the python object (dict) to pickle file
            pickle.dump(var_dict,f)

            # close file
            f.close()
            print('Variation 6 dictionary created and saved!')
            del pop_stats
            del df
            gc.collect()
        else:    
            var_dict = pd.read_pickle(var_6_path)   
            print('Dictionary has been imported')

        chunk_size = 10000  
        vardf_path = 'data-models/Data/vardf_3.pkl'
        
        with pd.HDFStore(vardf_6_path, mode='w') as store:

            def dict_chunk(var_dict, chunk_size):
                keys =list(var_dict.keys())
                for i in range(0, len(keys), chunk_size):
                    yield pd.DataFrame.from_dict({k: var_dict[k] for k in keys[i:i+chunk_size]}, orient='index')

            for i, df_chunk in enumerate(dict_chunk(var_dict, chunk_size)):
                store.append('df', df_chunk, format='table',data_columns=True)
        # Free up memory
        del var_dict
        gc.collect()
        var_df = pd.read_hdf(vardf_6_path, key='df')
        print('Variation 6 scores reated and loaded')

    else:
        var_df = pd.read_hdf(vardf_6_path, key='df')
        print('Variation 6 scores loaded')

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
pipeline = pt.FeaturesBatchRetrieve(index, wmodel='BM25', features=["WMODEL:Tf","WMODEL:PL2"])  >> MyScorer_6 #>> pt.text.get_text(train_dataset, get_var_5_feature_list()) % 100
print(pipeline.search('woman'))

