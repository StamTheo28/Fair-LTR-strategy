import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import Counter


def get_feature_list():
      return ['qual_cat','source_subcont_regions','page_subcont_regions', 'num_sitelinks_category','relative_pageviews_category', 'creation_date_category',
              'years_category','first_letter_category']

def calculate_per(dictionary):
    total = sum(dictionary.values())
    return (pd.Series(dictionary)/total).to_dict()



def skewed_metrics(ranked_df, global_stats):
    feature_list = get_feature_list()
    ranked_df['fair_score'] = 0
    skewed_score = []
    for f in feature_list:
        # create a statistics df fr the ranked_df
        sample_stats = ranked_df[f].value_counts(dropna=True).to_dict()
        if f == 'years_category':
            sample_stats.pop('Unknown')
       
        # Convert each category to a percentatge
        sample_per = calculate_per(sample_stats)
        global_per = calculate_per(global_stats[f])


        # Add any categories missing from the saple df
        X = global_per.keys()
        Y = sample_per.keys()
  
        if X != Y:
            for i in X:
                if i not in Y:
                    sample_per[i] = 0
        
        for i, row in ranked_df.iterrows():
            # Unfortunely gaps in the data
            if row[f]==None or row[f]=='Unknown':
                continue
            
            score = global_per[row[f]] - sample_per[row[f]]
            prev_score = row['fair_score']
            ranked_df.loc[i, 'fair_score'] = prev_score + score
    
        
        #make_plots(X, samples)
    print(ranked_df['fair_score'].mean())
    return ranked_df['fair_score'].mean()



def make_plots(X, samples):
    graph_path = '/home/ubuntu/Desktop/Fair-Ranking-Repo/fair-learning-to-rank-strategy/graphs/'

    feature_dict = {'qual_cat':'Quality categories','relative_pageviews_category':'Popularity',
                    'creation_date_category': 'Date creation category',
                    'years_category':'Age of topic article','first_letter_category':'First letter category of article'
                    }
            # Plot  a bar chart
    X_axis = np.arange(len(X))
    plt.figure()
    plt.rcParams["savefig.directory"] = os.chdir(os.path.dirname(graph_path))
    plt.bar(X_axis - 0.2, samples, 0.4, label = 'Sample data')
    plt.bar(X_axis + 0.2, globals, 0.4, label = 'Population data')
    plt.xticks(X_axis, X)
    plt.xlabel(feature_dict[f])
    plt.ylabel("% distribution")
    plt.title("% Wikipedia Articles " + feature_dict[f])
    plt.legend()
    plt.savefig(graph_path+ f + '_bar.png')
        