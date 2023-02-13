import pyterrier as pt
from pyterrier.measures import *
from sklearn.ensemble import RandomForestRegressor
from metric_utils.metrics import get_feature_list, skewed_metrics
import os
import numpy as np
from metric_utils import awrf
import pandas as pd
import lightgbm as lgb
import joblib

def baseline_stats(models, test_topics, qrels):
    print(pt.Experiment(
    models, 
    test_topics, qrels,
    eval_metrics=['recip_rank',MAP, nDCG@10],
    names=['BM25','TF',"RF-LTR","LGBM-LTR"],
    #baseline=0,
    ))

    skewed_metrics(res, get_feature_list(), feature_stats )   

    # converting a fairnes category to group binary data
    for feature in get_feature_list():
        
        feature_df = res[feature].str.join('|').str.get_dummies()

        #for i in res['qual_cat']:
        align = feature_df
        print('The AWRF score of ',feature,' is: ', awrf.vlambda(align, distance=awrf.subtraction))

    

def get_ltr_rdmf_model(filename, pipeline, train_topics, qrels, dataset):
    if not os.path.exists("data-models/"+filename):
        rf = RandomForestRegressor(n_estimators=400)
        rf_pipe = pipeline >> pt.ltr.apply_learned_model(rf) 
        rf_pipe.fit(train_topics, qrels)
        print(type(rf_pipe))
        joblib.dump(rf_pipe, filename=filename)
        print('Random Forest Model created and saved')
    else:
        rf_pipe = joblib.load("data-models/"+filename)
        print('Random Forest Model loaded')
    return rf_pipe

def get_ltr_lgbm_model(filename, pipeline, train_topics, qrels, dataset):
    if not os.path.exists("data-models/"+filename):
        clf = lgb.LGBMClassifier()
        clf_pipe = pipeline  >>  pt.ltr.apply_learned_model(clf) 
        clf_pipe.fit(train_topics, qrels)
        joblib.dump(clf_pipe, filename=filename)
        print('LightGBM Model created and saved')
    else:
        clf_pipe = joblib.load("data-models/"+filename)
        print('LightGBM Model loaded')
    return clf_pipe




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
    # print collection statistics
    print(index.getCollectionStatistics().toString())

    topics = train_dataset.get_topics('text')
    qrels = train_dataset.get_qrels()
    pipeline = pt.FeaturesBatchRetrieve(index, wmodel='BM25', features=["WMODEL:Tf","WMODEL:PL2"]) 
       
    
    # BM25 retrieval model
    bm25 = pt.BatchRetrieve(index, wmodel='BM25')

    # Tf retrieval model
    tf = pt.BatchRetrieve(index, wmodel='Tf')
    print('FeatureBatchRetrieval done')

    # Split topics 
    # Need to re-think this
    print(np.split(topics, [int(.6*len(topics)), int(.8*len(topics))]))
    train_topics, valid_topics, test_topics = np.split(topics, [int(.6*len(topics)), int(.8*len(topics))])
    print('Topics split')

    model_filename_1 = "randomForest_model.joblib"
    model_filename_2 = "lightgbm_model.joblib"


    # Get trained Random Forest model
    rf_pipe = get_ltr_rdmf_model(model_filename_1, pipeline, train_topics, qrels, train_dataset) >> pt.text.get_text(train_dataset, get_feature_list())

    # Get trained LightGBM model
    clf_pipe = get_ltr_lgbm_model(model_filename_2, pipeline, train_topics, qrels, train_dataset) >> pt.text.get_text(train_dataset, get_feature_list())

    models = [bm25,tf, rf_pipe, clf_pipe]


    
    stats_path = 'data-models/features_stats.pkl'
    if not os.path.exists(stats_path):
        print('Global dataset statistics not available')
    else:
        feature_stats = pd.read_pickle(stats_path)
        print('Global statistics loaded')
        #print(feature_stats)

    res = clf_pipe.search('agriculture')  

    skewed_metrics(res, get_feature_list(), feature_stats )
    

    # converting a fairnes category to group binary data
    for feature in get_feature_list():
        
        feature_df = res[feature].str.join('|').str.get_dummies()

        #for i in res['qual_cat']:
        align = feature_df
        print('The AWRF score of ',feature,' is: ', awrf.vlambda(align, distance=awrf.subtraction))
    

    baseline_stats(models, test_topics, qrels, res)

    


if __name__ == "__main__":
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64/"
    if pt.is_windows():
        print('Operating System not Supported')
    else:
        print('Operating system supported!')
    pt.init()
    print('Pyterrier started')
    main()


