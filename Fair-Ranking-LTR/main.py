import pyterrier as pt
from pyterrier.measures import *
from data import download_data_p, download
import os
import pandas as pd
import ir_datasets

def main():  

    print('Get data')
    
    dataset = pt.get_dataset('irds:trec-fair/2022')
    train_dataset = pt.get_dataset('irds:trec-fair/2022/train')
    # Index trec-fair/2022
    indexer = pt.IterDictIndexer('./indices/trec-fair_2022', meta={'docno':25})
    if not os.path.exists('./indices/trec-fair_2022'):
        index = indexer.index(dataset.get_corpus_iter(), fields=['title', 'text'])
    else:
        index =pt.IndexFactory.of('./indices/trec-fair_2022')
    print(index.getCollectionStatistics().toString())

    topics = train_dataset.get_topics('text')
    qrels = train_dataset.get_qrels()
    print('Printing qrels: \n')
    print("QID 84:",len(qrels[qrels['qid']=="84"]))
    print("QID 100:",len(qrels[qrels['qid']=="100"]))


    pipeline = pt.FeaturesBatchRetrieve(index, wmodel='BM25', features=["WMODEL:Tf","WMODEL:PL2"])
    res = pipeline(topics)
    print(res)
    pt.Experiment(
        [pipeline],
        train_dataset.get_topics('text'),
        train_dataset.get_qrels(),
        [MAP, nDCG@20]
    )


    print('Process Successful')
    return


if __name__ == "__main__":
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64/"
    if pt.is_windows():
        print('Operating System not Supported')
    else:
        print('Operating system supported!')
    pt.init()
    print('Pyterrier started')
    main()


