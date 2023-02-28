import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# The aim of this script is to perform data analysis and observe relative relationships between features in the dataset

computed_df_path = "data-models/Data/computed_df.pkl"
feature_stats_path = "data-models/Data/features_stats.pkl"

if not os.path.exists(computed_df_path) or not os.path.exists(feature_stats_path):
    print('Dataset feature statistics are not available!')
else:
    computed_df = pd.read_pickle(computed_df_path)
    pop_stats = pd.read_pickle(feature_stats_path)


# Make Graphs for each category
i=0
graph_path = "data-models/Graphs/Population/"
null_keys = [None, 'UNKnOWN', "N/A",float('nan'),'UNK','Unknown']


for key in pop_stats.keys():    
    if key != 'docid':
        data={}
        plt.figure(i)
        # Remove Nan/Unknown values
        for sub_key, val in pop_stats[key].items():
            if sub_key in null_keys or pd.isna(sub_key):
                continue
            else:
                data[sub_key] = val
        X_axis = np.arange(len(data))
        plt.bar(X_axis , data.values(), label = '%of articles')
        if key in ['source_subcont_regions','page_subcont_regions','occupations']:
            plt.xticks(X_axis, data.keys(), rotation=90, fontsize=15)
        else:
            plt.xticks(X_axis, data.keys(), rotation=0, fontsize=16)
        plt.xlabel(key)
        plt.ylabel("% distribution")
        plt.title("% Wikipedia Articles " + key)
        plt.rcParams['figure.figsize'] = (10,6)
        plt.tight_layout()
        plt.legend()
        plt.savefig(graph_path+key)
        i+=1

