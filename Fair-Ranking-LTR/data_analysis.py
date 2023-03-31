import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
from collections import Counter
from variation_1 import get_var_1_feature_list
from variation_2 import get_var_2_feature_list
from variation_3 import get_var_3_feature_list
from variation_4 import get_var_4_feature_list
from variation_5 import get_var_5_feature_list
from variation_6 import get_var_6_feature_list
from variation_7 import get_var_7_feature_list

# Global paths
graph_path = "data-models/Graphs/Population/"
null_keys = [None, 'UNKNOWN', "N/A", float('nan'),'UNK','Unknown','nan']
computed_df_path = "data-models/Data/computed_df.pkl"
feature_stats_path = "data-models/Data/features_stats.pkl"

# Produce graphs for population distriutions
def population_graphs():
    if not os.path.exists(computed_df_path) or not os.path.exists(feature_stats_path):
        print('Dataset feature statistics are not available!')
    else:
        pop_stats = pd.read_pickle(feature_stats_path)
    # Make Graphs for each category
    i=0

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


def get_unknown_distributions():
    # Create A table of indicating the percentages of Unknown values
    if not os.path.exists(computed_df_path) or not os.path.exists(feature_stats_path):
        print('Dataset feature statistics are not available!')
    else:
        pop_stats = pd.read_pickle(feature_stats_path)
    null_dict = {}
    for key in pop_stats.keys():    
        if key != 'docid':
            data = {}
            counter = Counter(pop_stats[key])
            null_val = 0
            val = 0
            for category in counter.keys():
                if key == 'years_category':
                    print(category, str(category)=='nan')
                if category in null_keys or str(category)=='nan':
                    null_val += counter[category]
                else:
                    val += counter[category]
            data['Unknown'] = null_val
            data['Known'] = val
            null_dict[key] = data
    null_df = pd.DataFrame(null_dict)
    print(null_df)

# Create plots for imputed data
def imputation_analysis():
    df_path = 'data-models/Data/imputed_df.pkl'
    df = pd.read_pickle(df_path)
    gender = df['gender'].value_counts().to_dict()
    occupations = df['occupations'].value_counts().to_dict()
    data = [gender, occupations]
    key = ['gender', 'occupations']
    for i in range(2):
        plt.figure(i+7 )
        # Remove Nan/Unknown values

        X_axis = np.arange(len(data[i]))
        plt.bar(X_axis , data[i].values(), label = '%of articles')
        if key[i] == 'occupations':
            plt.xticks(X_axis, data[i].keys(), rotation=90, fontsize=15)
        else:
            plt.xticks(X_axis, data[i].keys(), rotation=0, fontsize=20)
        plt.xlabel(key)
        plt.ylabel("% distribution")
        plt.title("% Wikipedia Articles " + key[i], fontsize=26)
        plt.rcParams['figure.figsize'] = (10,6)
        plt.tight_layout()
        plt.legend()
        plt.savefig(graph_path+"imputed"+key[i])


    
# Create plots for feature importances
def get_feature_importance_graphs():
    graph_path = "data-models/Graphs/feature-importance/"
    var_features_list = [get_var_1_feature_list(), get_var_2_feature_list(), get_var_3_feature_list()
                        , get_var_4_feature_list(), get_var_5_feature_list(), get_var_6_feature_list(),
                        get_var_7_feature_list()]
    model_names = ['LightGBM Baseline', 'AFI', 'ASTL', 'ACT', 'AAD', 'DASTL', 'UACT', 'BIAAD']
    relevance_name_list=['TF', 'PL2']
    for i in range(0,8):
        if i==0:
            model = joblib.load('data-models/Models/ml/lightgbm_model.joblib')
            model.feature_names = relevance_name_list
            
        else:
            model = joblib.load('data-models/Models/ml/lightgbm_model_var_'+str(i)+'.joblib')
            model.feature_names = relevance_name_list + var_features_list[i-1]

        x = model.feature_names
        y = model.feature_importances_
        print(y)
        plt.figure(i+15)
        X_axis = np.arange(len(x))
        plt.bar(X_axis , y, label = 'Importance value', width=0.5)
        if i==7:
            plt.xticks(X_axis, x, rotation=45, fontsize=10)
        else:
            plt.xticks(X_axis, x, rotation=45)
        plt.xlabel(model_names[i])
        plt.ylabel("Importance value")
        plt.title("Fature Importance for " + model_names[i])
        #plt.rcParams['figure.figsize'] = (10,6)
        plt.tight_layout()
        plt.legend()
        plt.savefig(graph_path+model_names[i])





imputation_analysis()

