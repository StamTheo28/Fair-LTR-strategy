import pyterrier as pt
import os
import pandas as pd
import numpy as np
from metric_utils.metrics import get_feature_list, skewed_metrics, calculate_per
from metric_utils import awrf

print('Get data')
if not pt.started():
    pt.init()
dataset = pt.get_dataset('irds:trec-fair/2022')
train_dataset = pt.get_dataset('irds:trec-fair/2022/train')

# Index trec-fair/2022
indexer = pt.IterDictIndexer('./indices/trec-fair_2022', meta={'docno':25})
if not os.path.exists('./indices/trec-fair_2022'):
    index = indexer.index(dataset.get_corpus_iter(), fields=['title', 'text'])
else:
    index =pt.IndexFactory.of('./indices/trec-fair_2022')
# print collection statistics

topics = train_dataset.get_topics('text')
qrels = train_dataset.get_qrels()

pipeline = pt.FeaturesBatchRetrieve(index, wmodel='BM25', features=["WMODEL:Tf","WMODEL:PL2"])  >> pt.text.get_text(train_dataset, get_feature_list())
#res = bm25(topics)
res = pipeline.search('agriculture')#transform(topics)
print(res)


stats_path = 'data-models/features_stats.pkl'
if not os.path.exists(stats_path):
    print('Global dataset statistics not available')
else:
    feature_stats = pd.read_pickle(stats_path)
    print('Global statistics loaded')
    #print(feature_stats)


# Implementing First Fairness Algorithm

feature_list = get_feature_list()

res['fair_score'] = 0

for f in feature_list:
    # create a statistics df fr the ranked_df
    sample_stats = res[f].value_counts(dropna=True).to_dict()
    if f == 'years_category':
        sample_stats.pop('Unknown')
       
    # Convert each category to a percentatge
    sample_per = calculate_per(sample_stats)
    global_per = calculate_per(feature_stats[f])

    # Add any categories missing from the saple df
    X = global_per.keys()
    Y = sample_per.keys()
  
    if X != Y:
        for i in X:
            if i not in Y:
                sample_per[i] = 0

    for i, row in res.iterrows():
        # Unfortunely gaps in the data
        if row[f]==None or row[f]=='Unknown':
            continue
        
        score = global_per[row[f]] - sample_per[row[f]]
        prev_score = row['fair_score']
        res.loc[i, 'fair_score'] = prev_score + score


features_lists = []
# Add the computed fair_scores to the feature list
for index, row in res.iterrows():
    score_list = row['features'].tolist()
    fair_score = row['fair_score']
    score_list.append(fair_score)
    res.at[index, 'features'] = score_list
    
print(res.head())
print(res['features'].head())


def fairness_scores():
    scores = [row for row in res['fair_score'].tolist()]
    return {'fair_score':scores}


# Now rerank the list with the new scores
res_features = res['fair_score']
x = "WMODEL:Tf","WMODEL:PL2", "res_features"

index =pt.IndexFactory.of('./indices/trec-fair_2022')
rerank = pt.FeaturesBatchRetrieve(pipeline, wmodel='BM25', features=["WMODEL:Tf","WMODEL:PL2", 'fairness_scores'])
#,'fairness_scores'
se = rerank.search('agriculture')
print(se)



# converting a fairnes category to group binary data
for feature in get_feature_list():
    if feature == 'pred_qual':
        continue

    #print(feature)

    feature_df = pd.get_dummies(res[feature].values.tolist())
   
    break
    #for i in res['qual_cat']:
    align = feature_df
    print('The AWRF score of ',feature,' is: ', awrf.vlambda(align, distance=awrf.subtraction))



