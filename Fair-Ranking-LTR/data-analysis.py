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

print(computed_df.head(10))
print(pop_stats.keys())

# Make Graphs for each category
i=0
graph_path = "data-models/Graphs/Population/"
for key in pop_stats.keys():
    if key != 'docid':
        plt.figure(i)
        data = pop_stats[key]
        X_axis = np.arange(len(data))
        plt.bar(X_axis , data.values(), label = 'Population data')
        plt.xticks(X_axis, data.keys(), rotation=90)
        plt.xlabel(key)
        plt.ylabel("% distribution")
        plt.title("% Wikipedia Articles " + key)
        plt.tight_layout()
        plt.legend()
        plt.savefig(graph_path+key)
        i+=1