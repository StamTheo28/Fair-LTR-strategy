import pyterrier as pt
from pyterrier.measures import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from metric_utils.metrics import get_feature_list, skewed_metrics
import os
from variation_1 import MyScorer_1
from variation_2 import MyScorer_2
import numpy as np
from metric_utils import awrf
import pandas as pd
import pickle
import lightgbm as lgb
import joblib

def baseline_evaluation(models,model_names, test_topics, qrels, feature_stats):
    results = pt.Experiment(
    models, 
    test_topics, qrels,
    eval_metrics=['recip_rank',MAP, nDCG@10],
    names=model_names,
    #baseline=0,
    )
    return results
    #skewed_metrics(res, get_feature_list(), feature_stats)   

    # converting a fairnes category to group binary data
    model_feature_score =[]
    model_skewed_score = {}
    for i in range(len(models)):
        model_path = "data-models/"+model_names[i]+'.pkl'
        if os.path.getsize(model_path)>0:
            res = pd.DataFrame(pd.read_pickle(model_path))
            
        else:
            print('pickle file is empty')

        score_dict = {}
        score_dict['name'] = model_names[i]
        # Get Skewness metrics
        model_skewed_score['name'] = model_names[i]
       # model_skewed_score['skewed_fair'] = skewed_metrics(res, feature_stats)
        

        # Get AWRF metrics
        for feature in get_feature_list():  
            feature_df = pd.get_dummies(res[feature].values.tolist())
            #feature_df = res[feature].str.join('|').str.get_dummies()
            score = awrf.vlambda(feature_df, distance=awrf.subtraction)
            score_dict[feature] = score
        model_feature_score.append(score_dict)
    fair_df = pd.DataFrame(model_feature_score, index = [0,1,2,3])
    #skewed_df = pd.DataFrame(model_skewed_score, index = [0,1,2,3])
    fair_df['model_awrf'] = fair_df[get_feature_list()].mean(axis=1)
    metrics_df = pd.merge(results, fair_df[['name','model_awrf']], on='name')
    #print(skewed_df)
    print(metrics_df)
   # metrics_df = pd.merge(metrics_df, skewed_df)
    return metrics_df
    

def get_ltr_rdmf_model(filename, pipeline, train_topics, qrels, dataset):
    if not os.path.exists("data-models/Models"+filename):
        rf = RandomForestRegressor(n_estimators=400)
        rf_pipe = pipeline >> pt.ltr.apply_learned_model(rf) 
        rf_pipe.fit(train_topics, qrels)
        joblib.dump(rf_pipe, filename=filename)
        print('Random Forest Model created and saved')
    else:
        rf_pipe = joblib.load("data-models/Models"+filename)
        print('Random Forest Model loaded')
    return rf_pipe

def get_ltr_lgbm_model(filename, pipeline, train_topics, qrels, dataset, baseline=False, var=0):
    if not os.path.exists("data-models/Models"+filename):
        clf = lgb.LGBMClassifier()
        if var==1:
            clf_pipe = pipeline >> MyScorer_1()  >>  pt.ltr.apply_learned_model(clf) 
        elif var==2:
            clf_pipe = pipeline >> MyScorer_2()  >>  pt.ltr.apply_learned_model(clf) 
        elif var==0:
            clf_pipe = pipeline  >>  pt.ltr.apply_learned_model(clf) 
        clf_pipe.fit(train_topics, qrels)
        print('Storing LGBM model')
        joblib.dump(clf_pipe, filename=filename)
        print('LightGBM Model created and saved')
    else:
        clf_pipe = joblib.load("data-models/Models/"+filename)
        print('LightGBM Model loaded')
    return clf_pipe




def main():  

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
    
    pipeline = pt.FeaturesBatchRetrieve(index, wmodel='BM25', features=["WMODEL:Tf","WMODEL:PL2"]) 
    print('FeatureBatchRetrieval done')
    
    # BM25 retrieval model
    bm25 = pt.BatchRetrieve(index, wmodel='BM25') >> pt.text.get_text(train_dataset, get_feature_list())
    print('BM25 done')
    # Tf retrieval model
    tf = pt.BatchRetrieve(index, wmodel='Tf') >> pt.text.get_text(train_dataset, get_feature_list())
    print('TF done')
    #.search('agriculture')

    # Split topics 
    # Need to re-think this
    train_topics, test_topics = train_test_split(topics, test_size=0.3, random_state=23)
    print('Topics split')
 
    model_filename_1 = "randomForest_model.joblib"
    model_filename_2 = "lightgbm_model.joblib"
    model_clf_var_1 = "lightgbm_model_var_1.joblib"


    # Get trained Random Forest model
    rf_pipe = get_ltr_rdmf_model(model_filename_1, pipeline, train_topics, qrels, train_dataset) >>  pt.text.get_text(train_dataset, get_feature_list())

    # Get trained LightGBM model
    clf_pipe = get_ltr_lgbm_model(model_filename_2, pipeline, train_topics, qrels, train_dataset) >> pt.text.get_text(train_dataset, get_feature_list())

    # Get trained LightGBM model for variaton 1
    clf_var_1_pipe = get_ltr_lgbm_model(model_clf_var_1, pipeline, train_topics, qrels, train_dataset,1) >> pt.text.get_text(train_dataset, get_feature_list())

    # Get trained LightGBM model for variaton 2
    clf_var_2_pipe = get_ltr_lgbm_model(model_clf_var_1, pipeline, train_topics, qrels, train_dataset,2) >> pt.text.get_text(train_dataset, get_feature_list())
    print('All LTR models loaded')
 
    models = [bm25, tf,rf_pipe, clf_pipe,clf_var_1_pipe,clf_var_2_pipe]
    model_names = ["BM25","TF","RF-LTR","LGBM-LTR","LGBM-LTR-VAR-1","LGBM-LTR-VAR-2"]
    query = 'agriculture'


    ### Get the relevance scors of the models
    stats_path = 'data-models/Data/features_stats.pkl'
    if not os.path.exists(stats_path):
        print('Global dataset statistics not available')
    else:
        feature_stats = pd.read_pickle(stats_path)
        print('Global statistics loaded')
        #print(feature_stats)

    
    # Evaluate the baseline models without any fairness algorithms implemented
    eval_df = baseline_evaluation(models, model_names, test_topics, qrels, feature_stats)
    print(eval_df)



    for i in range(len(models)):
        model_path = "data-models/Ranked-Results"+model_names[i]+'.pkl'
        if not os.path.exists(model_path):
            m = models[i].search(query).to_dict()
            f = open(model_path,"wb")
            pickle.dump(m,f)
            f.close()
            print(model_names[i],'for query ', query ,' successfully created and saved')
        else:

            print(model_names[i],'for query ', query ,' exists')



    # Implement the first Fair algorithm

    


if __name__ == "__main__":
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64/"
    if pt.is_windows():
        print('Operating System not Supported')
    else:
        print('Operating system supported!')
    pt.init()
    print('Pyterrier started')
    main()



