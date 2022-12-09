import pyterrier as pt


def download_data(dataset):
    dataset = pt.get_dataset(dataset)
    indexer = pt.IterDictIndexer('./indices/trec-fair_2021')
    index_ref = indexer.index(dataset.get_corpus_iter(), fields=['title', 'text', 'url'])
    print('Get Data')
    print(index_ref)