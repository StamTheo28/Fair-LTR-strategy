import pyterrier as pt
import os
import pandas as pd
from metric_utils.metrics import get_feature_list
from metric_utils.metrics import metric_analysis as ma

import metric_utils.position as pos
import metric_utils.groupinfo as gi

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
#skewed_metrics(res, get_feature_list())
stats_path = 'data-models/features_stats.pkl'
if not os.path.exists(stats_path):
    print('Global dataset statistics not available')
else:
    feature_stats = pd.read_pickle(stats_path)
    print('Global statistics loaded')
    #print(feature_stats)

quality_df = pd.Series(feature_stats['quality'])
feature_1 = gi.GroupInfo(quality_df, quality_df.idxmax(), quality_df.idxmin(), 'quality')
print(feature_1.tgt_p_binomial)

MA = ma(res, feature_1)
default_results= MA.run_default_setting(len(res))
default_results


