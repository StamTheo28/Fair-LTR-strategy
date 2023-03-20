import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import Counter
from scipy.stats import skew


def get_feature_list():
      return ['qual_cat','source_subcont_regions','occupations','gender','page_subcont_regions','relative_pageviews_category', 'creation_date',
              'years_category','first_letter_category']

def get_feature():
      return ['qual_cat','source_subcont_regions','occupations','gender','page_subcont_regions','relative_pageviews_category', 'creation_date',
              'years_category']

def calculate_per(dictionary):
    total = sum(dictionary.values())
    return (pd.Series(dictionary)/total).to_dict()

def skewness(ranked_df, feature_list):
    score = []
    for f in feature_list:
        if f in ['source_subcont_regions','page_subcont_regions']:
               
            feature_df = ranked_df[f].str.join('|').str.get_dummies()
        else:
            feature_df = ranked_df[f].str.join('').str.get_dummies()
        
        data = []
        for i in feature_df.columns:
            data.append(sum(feature_df[i]))

        s = skew(np.array(data), bias=False)
        score.append(s)
    return np.mean(np.array(score))




