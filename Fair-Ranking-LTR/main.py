import pyterrier as pt
from pyterrier.measures import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from metric_utils.metrics import get_feature_list
import os
from variation_1 import MyScorer_1, get_var_1_feature_list
from variation_2 import MyScorer_2, get_var_2_feature_list
from variation_3 import MyScorer_3, get_var_3_feature_list
from variation_4 import MyScorer_4, get_var_4_feature_list
from variation_5 import MyScorer_5, get_var_5_feature_list
from variation_6 import MyScorer_6, get_var_6_feature_list
from variation_7 import MyScorer_7, get_var_7_feature_list
from evaluation import  variation_evaluation, baseline_evaluation
import pandas as pd
import pickle
import lightgbm as lgb
import joblib

    

def get_ltr_rdmf_model(filename, pipeline, train_topics, qrels):
    if not os.path.exists("data-models/Models/"+filename):
        rf = RandomForestRegressor(n_estimators=400)
        rf_pipe = pipeline >> pt.ltr.apply_learned_model(rf) 
        rf_pipe.fit(train_topics, qrels)
        joblib.dump(rf_pipe, filename="data-models/Models/"+filename)
        print('Random Forest Model created and saved')
    else:
        rf_pipe = joblib.load("data-models/Models/"+filename)
        print('Random Forest Model loaded')
    return rf_pipe

def get_ltr_lgbm_model(filename, pipeline, train_topics, qrels, var=0):
    if not os.path.exists("data-models/Models/"+filename):
        clf = lgb.LGBMClassifier()
        if var==1:
            clf_pipe = pipeline >> MyScorer_1()  >>  pt.ltr.apply_learned_model(clf) 
        elif var==2:
            clf_pipe = pipeline >> MyScorer_2()  >>  pt.ltr.apply_learned_model(clf) 
        elif var==3:
            clf_pipe = pipeline >> MyScorer_3()  >>  pt.ltr.apply_learned_model(clf) 
        elif var==4:
            clf_pipe = pipeline >> MyScorer_4()  >>  pt.ltr.apply_learned_model(clf) 
        elif var==5:
            clf_pipe = pipeline >> MyScorer_5()  >>  pt.ltr.apply_learned_model(clf) 
        elif var==6:
            clf_pipe = pipeline >> MyScorer_6()  >>  pt.ltr.apply_learned_model(clf) 
        elif var==7:
            clf_pipe = pipeline >> MyScorer_7()  >>  pt.ltr.apply_learned_model(clf) 
        elif var==0:
            clf_pipe = pipeline  >>  pt.ltr.apply_learned_model(clf) 
        clf_pipe.fit(train_topics, qrels)
        print('Storing LGBM model')
        joblib.dump(clf_pipe, filename="data-models/Models/"+filename)
        print('LightGBM Model ' + filename + ' created and saved')
    else:
        clf_pipe = joblib.load("data-models/Models/"+filename)
        print('LightGBM ' + filename + ' Model loaded')
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
    bm25 = pt.BatchRetrieve(index, wmodel='BM25') >> pt.text.get_text(train_dataset, get_feature_list()) % 100
    print('BM25 done')
    # Tf retrieval model
    tf = pt.BatchRetrieve(index, wmodel='Tf') >> pt.text.get_text(train_dataset, get_feature_list()) % 100
    print('TF done')
    #.search('agriculture')

    # Split topics 
    # Need to re-think this
    train_topics, test_topics = train_test_split(topics, test_size=0.3, random_state=23)
    print('Topics split')
 
    model_filename_1 = "randomForest_model.joblib"
    model_filename_2 = "lightgbm_model.joblib"
    model_clf_var_1 = "lightgbm_model_var_1.joblib"
    model_clf_var_2 = "lightgbm_model_var_2.joblib"
    model_clf_var_3 = "lightgbm_model_var_3.joblib"
    model_clf_var_4 = "lightgbm_model_var_4.joblib"
    model_clf_var_5 = "lightgbm_model_var_5.joblib"
    model_clf_var_6 = "lightgbm_model_var_6.joblib"
    model_clf_var_7 = "lightgbm_model_var_7.joblib"


    # Get trained Random Forest model
    rf_pipe = get_ltr_rdmf_model(model_filename_1, pipeline, train_topics, qrels) >>  pt.text.get_text(train_dataset, get_feature_list()) % 100

    # Get trained LightGBM model
    clf_pipe = get_ltr_lgbm_model(model_filename_2, pipeline, train_topics, qrels,0) >> pt.text.get_text(train_dataset, get_feature_list()) % 100

    # Get trained LightGBM model for variaton 1
    clf_var_1_pipe = get_ltr_lgbm_model(model_clf_var_1, pipeline, train_topics, qrels, 1) >> pt.text.get_text(train_dataset, get_var_1_feature_list()) % 100

    # Get trained LightGBM model for variaton 2
    clf_var_2_pipe = get_ltr_lgbm_model(model_clf_var_2, pipeline, train_topics, qrels, 2) >> pt.text.get_text(train_dataset, get_var_2_feature_list()) % 100

    # Get trained LightGBM model for variaton 3
    clf_var_3_pipe = get_ltr_lgbm_model(model_clf_var_3, pipeline, train_topics, qrels, 3) >> pt.text.get_text(train_dataset, get_var_3_feature_list()) % 100

    # Get trained LightGBM model for variaton 4
    clf_var_4_pipe = get_ltr_lgbm_model(model_clf_var_4, pipeline, train_topics, qrels, 4) >> pt.text.get_text(train_dataset, get_var_4_feature_list()) % 100

    # Get trained LightGBM model for variaton 5
    clf_var_5_pipe = get_ltr_lgbm_model(model_clf_var_5, pipeline, train_topics, qrels, 5) >> pt.text.get_text(train_dataset, get_var_5_feature_list()) % 100

    # Get trained LightGBM model for variaton 6
    clf_var_6_pipe = get_ltr_lgbm_model(model_clf_var_6, pipeline, train_topics, qrels, 6) >> pt.text.get_text(train_dataset, get_var_6_feature_list()) % 100

        # Get trained LightGBM model for variaton 7
    clf_var_7_pipe = get_ltr_lgbm_model(model_clf_var_7, pipeline, train_topics, qrels, 7) >> pt.text.get_text(train_dataset, get_var_7_feature_list()) % 100
    print('All LTR models loaded')
   
    base_models = [bm25, tf, rf_pipe, clf_pipe]
    var_models = [ clf_var_1_pipe, clf_var_2_pipe, clf_var_3_pipe, clf_var_4_pipe]
    var_2_models = [ clf_var_5_pipe, clf_var_6_pipe, clf_var_7_pipe]
    base_model_names = ["BM25","TF","RF-LTR","LGBM-LTR"]
    var_model_names = {"LGBM-LTR-VAR-1":get_var_1_feature_list(),"LGBM-LTR-VAR-2":get_var_2_feature_list()
                        , "LGBM-LTR-VAR-3":get_var_3_feature_list(), "LGBM-LTR-VAR-4":get_var_4_feature_list()}
    var_2_model_names =  {"LGBM-LTR-VAR-5":get_var_5_feature_list(),"LGBM-LTR-VAR-6":get_var_6_feature_list()
                        , "LGBM-LTR-VAR-7":get_var_7_feature_list()}


    # Create  eexperinment foorr baselines
    for i in range(len(base_models)):
        model_path = "data-models/Ranked-Results/"+base_model_names[i]+'.pkl'
        if not os.path.exists(model_path):
            m = base_models[i].transform(test_topics).to_dict()
            f = open(model_path,"wb")
            pickle.dump(m,f)
            f.close()
            print(base_model_names[i],' successfully created and saved for all test_topics')
        else:

            print(base_model_names[i],'Ranked results for all test_topics exist')
    
    # Create results for expectations
    for i in range(len(base_models)):
        model_path = "data-models/Ranked-Results/exp_"+base_model_names[i]+'.pkl'
        if not os.path.exists(model_path):
            m = base_models[i].transform(train_topics).to_dict()
            f = open(model_path,"wb")
            pickle.dump(m,f)
            f.close()
            print(base_model_names[i],' successfully created and saved for all train_topics')
        else:

            print(base_model_names[i],'Ranked results for all train_topics exist')
    
    # Creaete results for first set of variations
    count=0
    for key in var_2_model_names.keys():
        model_path = "data-models/Ranked-Results/"+key+'.pkl'
        if not os.path.exists(model_path):
            m = var_2_models[count].transform(test_topics).to_dict()
            f = open(model_path,"wb")
            pickle.dump(m,f)
            f.close()
            print(key,' successfully created and saved for all test_topics')
        else:

            print(key,'Ranked results for all topics exist')
        count+=1
        

    
        
    # Evaluate the baseline models without any fairness algorithms implemented
    base_exp_path = "data-models/experinment-results/base.pkl"
    expected_exp_path = "data-models/experinment-results/expected.pkl"
    variation_exp_path = "data-models/experinment-results/variation.pkl"
    variation_2_exp_path = "data-models/experinment-results/variation_2.pkl"

    # Perform evaluation on the baseline models
    print('Performing Base Evaluation')
    if not os.path.exists(base_exp_path):
        eval_df = baseline_evaluation(base_models, base_model_names, test_topics, qrels)
        f = open(base_exp_path,"wb")
        pickle.dump(eval_df,f)
        f.close()
        print('Base models evaluations saved')
    else:
        eval_df = pd.read_pickle(base_exp_path)
        print('Base Evaluation Loaded')
    print(eval_df)

    # Perform evaluation on the baseline models, using the training dataset to gather an experinment expectation
    print('Performing Base Evaluation with expectation parameters')
    if not os.path.exists(expected_exp_path):
        exp_eval_df = baseline_evaluation(base_models, base_model_names, train_topics, qrels, base=False)
        f = open(expected_exp_path,"wb")
        pickle.dump(exp_eval_df,f)
        f.close()
        print('Base models evaluations saved')
    else:
        exp_eval_df = pd.read_pickle(expected_exp_path)
        print('Base Evaluation Loaded')
    print(exp_eval_df)

    # Perform evaluation on the first set of variation models
    if not os.path.exists(variation_exp_path):
        variation_df = variation_evaluation(var_models, var_model_names, test_topics, qrels)
        f = open(variation_exp_path,"wb")
        pickle.dump(variation_df,f)
        f.close()
        print('Variation models evaluations saved')
    else:
        variation_df = pd.read_pickle(variation_exp_path)

        print('Variation Evaluations Loaded')
    print(variation_df)

    # Perform evaluation on the seconds set of variation models
    if not os.path.exists(variation_2_exp_path):
        variation_2_df = variation_evaluation(var_2_models, var_2_model_names, test_topics, qrels)
        f = open(variation_2_exp_path,"wb")
        pickle.dump(variation_2_df,f)
        f.close()
        print('Variation 2 models evaluations saved')
    else:
        variation_2_df = pd.read_pickle(variation_2_exp_path)

        print('Variation 2 Evaluations Loaded')
    print(variation_2_df)
    

    


if __name__ == "__main__":
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64/"
    if pt.is_windows():
        print('Operating System not Supported')
    else:
        print('Operating system supported!')
    pt.init()
    print('Pyterrier started')
    # Before running this, you need to run population_stats.py
    main()



