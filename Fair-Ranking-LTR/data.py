import pyterrier as pt
import os
import pandas as pd
import numpy as np
from metric_utils.metrics import get_feature_list, skewed_metrics
from metric_utils.metrics import metric_analysis as ma
from metric_utils.metrics import awrf

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

bm25 = pt.BatchRetrieve(index, wmodel='BM25') >> pt.text.get_text(train_dataset, get_feature_list())
#res = bm25(topics)
res = bm25.search('agriculture')
print(res)


stats_path = 'data-models/features_stats.pkl'
if not os.path.exists(stats_path):
    print('Global dataset statistics not available')
else:
    feature_stats = pd.read_pickle(stats_path)
    print('Global statistics loaded')
    #print(feature_stats)

skewed_metrics(res, get_feature_list(), feature_stats )


# converting a fairnes category to group binary data
for feature in get_feature_list():
    if feature == 'pred_qual':
        continue

    #print(feature)
    feature_df = res[feature].str.join('|').str.get_dummies()
    #print(feature_df)

    #for i in res['qual_cat']:
    align = feature_df
    print('The AWRF score of ',feature,' is: ', awrf.vlambda(align, distance=awrf.subtraction))



