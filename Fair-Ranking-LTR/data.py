import pyterrier as pt
import pandas as pd
import ir_datasets
import json


def download_data_p():
    train_topics_path = "/media/sf_Trec-fair-2022/trec_2022_train_reldocs.json"
    corpus_path = "/media/sf_Trec-fair-2022/trec_corpus_20220301_plain.json/trec_corpus_20220301_plain.json"
    articles_path = "/media/sf_Trec-fair-2022/trec_2022_articles_discrete.json/trec_2022_articles_discrete_V2.json"
    queries_path = "/media/sf_Trec-fair-2022/trec_2022_train_reldocs.json"
    data = pd.read_json(corpus_path, orient='records', lines = True, chunksize=5)
    metadata = pd.read_json(articles_path, orient='records', lines = True, chunksize=5)
    train_topics = pd.read_json(train_topics_path, orient='records', lines = True, chunksize=5)
    queries = pd.read_json(queries_path, orient='records', lines = True, chunksize=5)
    print('Data retrieved')
    return data, metadata, train_topics, queries
    
def download(dataset_name, pt):
    print('downloading')
    pt.download.trec_fair_2022()
    print('downloaded')
    #dataset = pt.get_dataset('irds:trec-fair/2022/train')
    #print(dataset)

def download_data(dataset_name):
    print('Loading Data')
    indexer = pt.TRECCollectionIndexer(dataset_name, blocks=True)
 
    #indexer = pt.IterDictIndexer('./indices/trec-fair_2022')
    
    # index_ref = indexer.index(dataset, fields=['title', 'text', 'url'])
    print('Data Successfully loaded')
    #print(index_ref)





#pt.init()
