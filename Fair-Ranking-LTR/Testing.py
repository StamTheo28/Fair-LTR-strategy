import pyterrier as pt
import os
import joblib

pt.init()


print('Loading Trec-Fair 2022 dataset')
dataset = pt.get_dataset('irds:trec-fair/2022')
train_dataset = pt.get_dataset('irds:trec-fair/2022/train')

# Index trec-fair/2022
indexer = pt.IterDictIndexer('./indices/trec-fair_2022', meta={'docno':25})
if not os.path.exists('./indices/trec-fair_2022'):
  index = indexer.index(dataset.get_corpus_iter(), fields=['title', 'text'])
else:
 index =pt.IndexFactory.of('./indices/trec-fair_2022')
# print collection statistics
print(index.getCollectionStatistics().toString())
topics = train_dataset.get_topics('text')
qrels = train_dataset.get_qrels()

#rf_pipe = joblib.load("data-models/LGBM-LTR-VAR-1.pkl")

pipe = joblib.load("data-models/lightgbm_model_var_1.joblib")
print(pipe.search('woman'))
print(type(pipe))