import pandas as pd
import pickle
import numpy as np
import os
import gc
import pyterrier as pt

def get_var_1_feature_list():
      return ['qual_cat','source_subcont_regions','occupations','gender','page_subcont_regions','relative_pageviews_category', 'creation_date',
              'years_category','first_letter_category']

class MyScorer_1(pt.Transformer):    
    def transform(self, input):   
        count = 0
        data = all_features_var()  
        for index, row in input.iterrows():
            if count%10000==0:
                print(count)
            docid = row['docid']
            features = get_var_1_feature_list()
            f_scores = data[get_var_1_feature_list()][data.docid == docid]
            f_list = []
            for f in features:
                if len(f_scores[f]) == 0:
                    f_list.append(0.0)
                else:
                    f_list.append(float(f_scores[f]))        
            relevance_scores = list(input.iloc[index]['features'])
            combined_scores = np.array(relevance_scores + list(f_list))
           
            input.at[index, 'features'] = combined_scores
            count+=1
        return input




def all_features_var():
    var_1_path = 'data-models/Data/var_1.pkl'
    vardf_1_path = 'data-models/Data/vardf_1.pkl'

    if not os.path.exists(vardf_1_path):
        if not os.path.exists(var_1_path):
            print('Global dataset statistics not available')

            data_path = "data-models/Data/computed_df.pkl"
            stats_path = "data-models/Data/features_stats.pkl"

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
            
        
            f = open(var_1_path,"wb")
            # write the python object (dict) to pickle file
            pickle.dump(var_dict,f)

            # close file
            f.close()
            print('Variation 1 dictionary created and saved!')
            del pop_stats
            del df
            gc.collect()
        else:    
            var_dict = pd.read_pickle(var_1_path)   
            print('Dictionary has been imported')

        chunk_size = 10000  
        vardf_1_path = 'data-models/Data/vardf_1.pkl'
        
        with pd.HDFStore(vardf_1_path, mode='w') as store:

            def dict_chunk(var_dict, chunk_size):
                keys =list(var_dict.keys())
                for i in range(0, len(keys), chunk_size):
                    yield pd.DataFrame.from_dict({k: var_dict[k] for k in keys[i:i+chunk_size]}, orient='index')

            for i, df_chunk in enumerate(dict_chunk(var_dict, chunk_size)):
                store.append('df', df_chunk, format='table',data_columns=True)
        # Free up memory
        del var_dict
        gc.collect()
        var_df = pd.read_hdf(vardf_1_path, key='df')
        print('Variation 1 scores reated and loaded')

    else:
        var_df = pd.read_hdf(vardf_1_path, key='df')
        print('Variation 1 scores loaded')
    return var_df


