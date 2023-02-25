from collections import Counter
import json
import locale
import gzip
import random
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

locale.setlocale(locale.LC_ALL, "")

computed_df_path = "data-models/Data/computed_df.pkl"
if not os.path.exists(computed_df_path):
    path_file = '/media/sf_Trec-fair-2022/trec_2022_articles_discrete.json.gz'

    man_gender_labels = ['male', 'cisgender male', 'transgender male', 'male organism', 'assigned male at birth']
    woman_gender_labels = ['female', 'cisgender female', 'transgender female', 'assigned female at birth']

    def categorize_gender(gender_list):
        if gender_list:
            genders = set()
            for g in gender_list:
                if g in man_gender_labels:
                    genders.add('man')
                elif g in woman_gender_labels:
                    genders.add('woman')
                else:
                    return 'Non-binary'
            if len(genders) > 1:
                return 'Non-binary'
            elif 'man' in genders:
                return 'Man'
            elif 'woman' in genders:
                return 'Woman'
        return 'Unknown'

    def categorize_pvs(relative_pvs):
        if relative_pvs < 0.125:
            return 'Low'
        elif relative_pvs < 0.25:
            return 'Medium-Low'
        elif relative_pvs < 0.5:
            return 'Medium-High'
        else:
            return 'High'
        
    def categorize_sitelinks(num_sitelinks):
        if num_sitelinks == 1:
            return 'English only'
        elif num_sitelinks <= 4:
            return '2-4 languages'
        else:
            return '5+ languages'
        
    def categorize_years(years):
        if years:
            med = np.median(years)
            if med < 1900:
                return 'Pre-1900s'
            elif med < 2000:
                return '20th century'
            else:
                return '21st century'
        else:
            return 'Unknown'
        
    def categorize_creation(year):
        if year <= 2006:
            return '2001-2006'
        elif year <= 2011:
            return '2007-2011'
        elif year <= 2016:
            return '2012-2016'
        elif year <= 2022:
            return '2017-2022'
        else:
            return year
        
        
    def categorize_alpha(letter):
        if locale.strcoll('e', letter) > 0:
            return 'a-d'  # note: also includes numbers etc.
        elif locale.strcoll('l', letter) > 0:
            return 'e-k'
        elif locale.strcoll('s', letter) > 0:
            return 'l-r'
        else:
            return 's-'  # note: also includes non-latin characters etc.
        
    def get_rand(vals):
        if vals:
            return random.choice(list(vals))
        else:
            return 'N/A'
        
    def update_qual_label(qual_score):
        if qual_score <= 0.42:
            return 'Stub'
        elif qual_score <= 0.56:
            return 'Start'
        elif qual_score <= 0.73:
            return 'C'
        elif qual_score <= 0.85:
            return 'B'
        elif qual_score <= 0.93:
            return 'GA'
        elif qual_score <= 1:
            return 'FA'
        else:
            return None
        
    data = []
    with gzip.open(path_file, 'rt') as fin:
        count = 0
        for i, line in enumerate(fin, start=1):
            j = json.loads(line)
            docid = count
            j['qual_cat'] = update_qual_label(j['pred_qual'])
            qc = j['qual_cat']
            fl = categorize_alpha(j['first_letter'])
            occ = get_rand(j['occupations'])
            src_geo = get_rand(j['source_subcont_regions'])
            art_geo = get_rand(j['page_subcont_regions'])
            gen = categorize_gender(j['gender'])
            year = categorize_creation(int(j['creation_date'][:4]))
            age = categorize_years(j['years'])
            pvs = categorize_pvs(j['relative_pageviews'])
            nsl = categorize_sitelinks(j['num_sitelinks'])
            j['first_letter_category'] = fl
            j['gender_category'] = gen
            j['creation_date_category'] = year
            j['years_category'] = age
            j['relative_pageviews_category'] = pvs
            #j['num_sitelinks_category'] = nsl
            j['source_countries'] = Counter(j['source_countries'])
            j['source_subcont_regions'] = Counter(j['source_subcont_regions'])
            data.append([docid, qc, fl, occ, src_geo, art_geo, gen, year, age, pvs]) #nsl
            count +=1
    
            
    temp = pd.DataFrame(data, columns=['docid','quality', 'first-letter', 'occupation', 'source-geo', 'article-geo',
                                 'gender', 'creation_date', 'article-topic-age', 'popularity']) #num-langs
    df = pd.DataFrame(data, columns=['docid','qual_cat', 'first_letter_category', 'occupations', 'source_subcont_regions', 'page_subcont_regions',
                                 'gender', 'creation_date', 'years_category', 'relative_pageviews_category']) # 'num_sitelinks_category'
    geo_cats = ['UNK', 'N/A',
        'Northern America', 'Caribbean', 'Central America', 'South America',
        'Northern Europe', 'Western Europe', 'Southern Europe', 'Eastern Europe',
        'Western Asia', 'Southern Asia', 'South-eastern Asia', 'Eastern Asia', 'Central Asia',
        'Australia and New Zealand', 'Polynesia', 'Melanesia', 'Micronesia',
        'Western Africa', 'Eastern Africa', 'Southern Africa', 'Middle Africa', 'Northern Africa',
        'Antarctica']
    #print(temp['docid'])
    df['qual_cat'] = pd.Categorical(temp['quality'], ['Stub', 'Start', 'C', 'B', 'GA', 'FA'])
    df['relative_pageviews_category'] = pd.Categorical(temp['popularity'], ['Low', 'Medium-Low', 'Medium-High', 'High'])
    df['page_subcont_regions'] = pd.Categorical(temp['article-geo'], geo_cats)
    df['source_subcont_regions'] = pd.Categorical(temp['source-geo'], geo_cats)
    df['years_category'] = pd.Categorical(temp['article-topic-age'], ['Pre-1900s', '20th century', '21st century'])
    df['first_letter_category'] = pd.Categorical(temp['first-letter'], ['a-d', 'e-k', 'l-r', 's-'])
    #df['num_sitelinks_category'] = pd.Categorical(temp['num-langs'], ['English only', '2-4 languages', '5+ languages'])
    df['creation_date'] = pd.Categorical(temp['creation_date'], ['2001-2006','2007-2011','2012-2016','2017-2022'])
    df.to_pickle("data-models/Data/computed_df.pkl")
    print('Dataframe created and saved!')
else:
    df = pd.read_pickle(computed_df_path)
    print('Dataframe Loaded Successfully!')

print(df)
# Create a dictionary of dummies for each category of fairness feature
# Covert these dummies into proportionality values
features_dict = {}
for feature in df.columns:
    features_dict[feature] = df[feature].value_counts(dropna=False).to_dict()



for key, values in features_dict.items():
    total = sum(values.values())
    for category, value in values.items():
        features_dict[key][category] = value/total
# Write the feature statistics into a folder
features_stats_path = 'data-models/Data/features_stats.pkl'
   
f = open(features_stats_path,"wb")
# write the python object (dict) to pickle file
pickle.dump(features_dict,f)

# close file
f.close()





        

            
