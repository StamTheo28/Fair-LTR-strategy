import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency
from collections import Counter
from variation_6 import get_var_6_feature_list
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
null_keys = [None, 'UNKNOWN', "N/A",float('nan'),'UNK','Unknown','nan']


# Create Bar charts for feature category comparison
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
            plt.xticks(X_axis, data.keys(), rotation=0, fontsize=20)
        plt.xlabel(key)
        plt.ylabel("% distribution")
        plt.title("% Wikipedia Articles " + key, fontsize=26)
        plt.rcParams['figure.figsize'] = (10,6)
        plt.tight_layout()
        plt.legend()
        plt.savefig(graph_path+key)
        i+=1

# Perform Chi-Test for creattion date and topic age
counts = computed_df.groupby(get_var_6_feature_list()).size().reset_index(name='Count')
table = counts.pivot(index=get_var_6_feature_list()[0], columns=get_var_6_feature_list()[1], values='Count')
print('Create Chi-Square')
stat, p, dof, expected = chi2_contingency(table)
print("Chi square statistic: ",stat)
print("P-Values: ",p)
print("Degress of freedom: ",dof)
print("Expected frequencies: ",expected)

# Variatipn 6 years_category
var_6_path = 'data-models/Data/vardf_6.pkl'
var_6 = pd.read_pickle(var_6_path)
print(var_6['years_category'].unique())


# Create A table of indicating the percentages of Unknown values
null_dict = {}
for key in pop_stats.keys():    
    if key != 'docid':
        data = {}
        counter = Counter(pop_stats[key])
        null_val = 0
        val = 0
        for category in counter.keys():
            if category in null_keys:
                null_val += counter[category]
            else:
                val += counter[category]
        data['Unknown'] = null_val
        data['Known'] = val
        null_dict[key] = data
null_df = pd.DataFrame(null_dict)
print(null_df)

