import pyterrier as pt
import pandas as pd


def download_data_p():
    print('Getting Data')
    train_topics_path = "/media/sf_Trec-fair-2022/trec_2022_train_reldocs.json"
    corpus_path = "/media/sf_Trec-fair-2022/trec_corpus_20220301_plain.json/trec_corpus_20220301_plain.json"
    articles_path = "/media/sf_Trec-fair-2022/trec_2022_articles_discrete.json/trec_2022_articles_discrete_V2.json"
    data = pd.read_json(corpus_path, orient='records', lines = True, chunksize=5)
    articles = pd.read_json(articles_path, orient='records', lines = True)
    train_topics = pd.read_json(train_topics_path, orient='records', lines = True, chunksize=5)
    #print(train_topics.columns)
    count = 0
    for a in articles:
        count += len(a)
    print(count)
    #print(len(train_topics))






def download_data(dataset_name):
    dataset = pt.get_dataset(dataset_name)
    indexer = pt.IterDictIndexer('./indices/trec-fair_2022', threads=8)
    index_ref = indexer.index(dataset.get_corpus_iter(), fields=['title', 'text', 'url'])
    print('Get Data')
    print(index_ref)
