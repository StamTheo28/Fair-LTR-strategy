import pyterrier as pt
import pytrec_eval
from pyterrier.measures import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from metric_utils.metrics import get_feature_list, get_feature_importance
from population_stats import create_population_statistics
import os
from variation_1 import MyScorer_1, get_var_1_feature_list
from variation_2 import MyScorer_2, get_var_2_feature_list
from variation_3 import MyScorer_3, get_var_3_feature_list
from variation_4 import MyScorer_4, get_var_4_feature_list
from variation_5 import MyScorer_5, get_var_5_feature_list
from variation_6 import MyScorer_6, get_var_6_feature_list
from variation_7 import MyScorer_7, get_var_7_feature_list
from evaluation import  variation_evaluation, baseline_evaluation, perform_t_statistic, relevance_evaluation, fairness_evaluation
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

def get_ltr_lgbm_model(name, pipeline, train_topics, qrels, var=0):
    filename = "data-models/Models/"+name
    if not os.path.exists(filename):
        print('Creating pipeline for '+ name)
        lgbm = lgb.LGBMClassifier()
        if var==1:
            lgbm_pipe = pipeline >> MyScorer_1()  >>  pt.ltr.apply_learned_model(lgbm) 
        elif var==2:
            lgbm_pipe = pipeline >> MyScorer_2()  >>  pt.ltr.apply_learned_model(lgbm) 
        elif var==3:
            lgbm_pipe = pipeline >> MyScorer_3()  >>  pt.ltr.apply_learned_model(lgbm) 
        elif var==4:
            lgbm_pipe = pipeline >> MyScorer_4()  >>  pt.ltr.apply_learned_model(lgbm) 
        elif var==5:
            lgbm_pipe = pipeline >> MyScorer_5()  >>  pt.ltr.apply_learned_model(lgbm) 
        elif var==6:
            lgbm_pipe = pipeline >> MyScorer_6()  >>  pt.ltr.apply_learned_model(lgbm) 
        elif var==7:
            lgbm_pipe = pipeline >> MyScorer_7()  >>  pt.ltr.apply_learned_model(lgbm) 
        elif var==0:
            lgbm_pipe = pipeline  >>  pt.ltr.apply_learned_model(lgbm) 
        lgbm_pipe.fit(train_topics, qrels)
        print('Storing LGBM model')
        joblib.dump(lgbm_pipe, filename=filename)
        joblib.dump(lgbm, filename='data-models/Models/ml/'+name)
        print('LightGBM Model ' + filename + ' created and saved')
    else:
        lgbm_pipe = joblib.load(filename)
        print('LightGBM ' + name + ' Model loaded')
    return lgbm_pipe




def main():  


    print('Loading Trec-Fair 2022 dataset')
    dataset = pt.get_dataset('irds:trec-fair/2022')
    train_dataset = pt.get_dataset('irds:trec-fair/2022/train')

    # Create or Load index of Trec-Fair 2022
    indexer = pt.IterDictIndexer('./indices/trec-fair_2022', meta={'docno':25})
    if not os.path.exists('./indices/trec-fair_2022'):
        index = indexer.index(dataset.get_corpus_iter(), fields=['title', 'text'])
    else:
        index =pt.IndexFactory.of('./indices/trec-fair_2022')
    # print collection statistics
    print(index.getCollectionStatistics().toString())

    # Get topics and qrels
    topics = train_dataset.get_topics('text')
    qrels = train_dataset.get_qrels()


    # check if the population statistics exist
    clean_data_path = "data-models/Data/computed_df.pkl"
    if not os.path.exists(clean_data_path):
        # Create population statistics
        create_population_statistics()
    
    # Create a featureBatchRetrieve model with BM25 and Tf, PL2 as features
    pipeline = pt.FeaturesBatchRetrieve(index, wmodel='BM25', features=["WMODEL:Tf","WMODEL:PL2"]) 
    print('FeatureBatchRetrieval done')
    
    # BM25 retrieval model
    bm25 = pt.BatchRetrieve(index, wmodel='BM25') >> pt.text.get_text(train_dataset, get_feature_list()) % 100
    print('BM25 done')
    # Tf retrieval model
    tf = pt.BatchRetrieve(index, wmodel='Tf') >> pt.text.get_text(train_dataset, get_feature_list()) % 100
    print('TF done')
    

    # Split topics to train and test data
    train_topics, test_topics = train_test_split(topics, test_size=0.3, random_state=23)
    print('Created train topics and test topics')
    
    # Filenames of the ml models to be used
    model_filename_1 = "randomForest_model.joblib"
    model_filename_2 = "lightgbm_model.joblib"
    model_lgbm_var_1 = "lightgbm_model_var_1.joblib"
    model_lgbm_var_2 = "lightgbm_model_var_2.joblib"
    model_lgbm_var_3 = "lightgbm_model_var_3.joblib"
    model_lgbm_var_4 = "lightgbm_model_var_4.joblib"
    model_lgbm_var_5 = "lightgbm_model_var_5.joblib"
    model_lgbm_var_6 = "lightgbm_model_var_6.joblib"
    model_lgbm_var_7 = "lightgbm_model_var_7.joblib"


    # Get trained Random Forest model 
    rf_pipe = get_ltr_rdmf_model(model_filename_1, pipeline, train_topics, qrels) >>  pt.text.get_text(train_dataset, get_feature_list()) % 100

    # Get trained LightGBM model
    lgbm_pipe = get_ltr_lgbm_model(model_filename_2, pipeline, train_topics, qrels,0) >> pt.text.get_text(train_dataset, get_feature_list()) % 100

    # Get trained LightGBM model for variaton 1
    lgbm_var_1_pipe = get_ltr_lgbm_model(model_lgbm_var_1, pipeline, train_topics, qrels, 1) >> pt.text.get_text(train_dataset, get_var_1_feature_list()) % 100
   

    # Get trained LightGBM model for variaton 2
    lgbm_var_2_pipe = get_ltr_lgbm_model(model_lgbm_var_2, pipeline, train_topics, qrels, 2) >> pt.text.get_text(train_dataset, get_var_2_feature_list()) % 100

    # Get trained LightGBM model for variaton 3
    lgbm_var_3_pipe = get_ltr_lgbm_model(model_lgbm_var_3, pipeline, train_topics, qrels, 3) >> pt.text.get_text(train_dataset, get_var_3_feature_list()) % 100
    
    # Get trained LightGBM model for variaton 4
    lgbm_var_4_pipe = get_ltr_lgbm_model(model_lgbm_var_4, pipeline, train_topics, qrels, 4) >> pt.text.get_text(train_dataset, get_var_4_feature_list()) % 100
    
    # Get trained LightGBM model for variaton 5
    lgbm_var_5_pipe = get_ltr_lgbm_model(model_lgbm_var_5, pipeline, train_topics, qrels, 5) >> pt.text.get_text(train_dataset, get_var_5_feature_list()) % 100
    
    # Get trained LightGBM model for variaton 6
    lgbm_var_6_pipe = get_ltr_lgbm_model(model_lgbm_var_6, pipeline, train_topics, qrels, 6) >> pt.text.get_text(train_dataset, get_var_6_feature_list()) % 100
    
    # Get trained LightGBM model for variaton 7
    lgbm_var_7_pipe = get_ltr_lgbm_model(model_lgbm_var_7, pipeline, train_topics, qrels, 7) >> pt.text.get_text(train_dataset, get_var_7_feature_list()) % 100
    print('All LTR models loaded')
   
    # Create lists of LTR pipelines and their names to be used in evaluation
    #base_models = [bm25, tf, rf_pipe,]
    var_models = [bm25, lgbm_pipe, lgbm_var_1_pipe, lgbm_var_2_pipe, lgbm_var_3_pipe, lgbm_var_4_pipe, lgbm_var_5_pipe, lgbm_var_6_pipe, lgbm_var_7_pipe]
    #base_model_names = ["BM25","TF","RF-LTR","LGBM-LTR"]
    var_model_names = {"BM25":get_feature_list(),"LGBM-LTR":get_feature_list(),"LGBM-LTR-VAR-1":get_var_1_feature_list(),"LGBM-LTR-VAR-2":get_var_2_feature_list()
                        , "LGBM-LTR-VAR-3":get_var_3_feature_list(), "LGBM-LTR-VAR-4":get_var_4_feature_list(), "LGBM-LTR-VAR-5":get_var_5_feature_list()
                        ,"LGBM-LTR-VAR-6":get_var_6_feature_list(), "LGBM-LTR-VAR-7":get_var_7_feature_list()}
    
    #var_2_model_names =  {"LGBM-LTR-VAR-5":get_var_5_feature_list(),"LGBM-LTR-VAR-6":get_var_6_feature_list()
    #                    , "LGBM-LTR-VAR-7":get_var_7_feature_list()}

    
    model_names = ['bm25','lgbm', 'AFI var 1', 'ASTL var_2', 'ACT var_3', 'AAD var_4', 'DASTL var_5', 'var_6', 'var_7']
    
    # perform statistical analysis
    print('Performing T-testing')
    #perform_t_statistic(var_models, model_names, test_topics, qrels)
    
    """
    # Create  experinment for baseline models
    for i in range(len(base_models)):
        model_path = "data-models/Ranked-Results/"+base_model_names[i]+'.pkl'
        if not os.path.exists(model_path):
            m = base_models[i].transform(test_topics)
            pt.io.write_results(model_path)
            f = open(model_path,"wb")
            pickle.dump(m,f)
            f.close()
            print(base_model_names[i],' successfully created and saved for all test_topics')
        else:

            print(base_model_names[i],'Ranked results for all test_topics exist')
    """
    """
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
    """
    # Creaete results for first set of variations
    count=0
    for key in var_model_names.keys():
        model_path = "data-models/Ranked-Results/"+key+'.pkl'
        if not os.path.exists(model_path):
            print(key, 'not found, create ranked results')
            m = var_models[count].transform(test_topics).to_dict()
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
    relevance_exp_path = "data-models/experinment-results/rev_exp.pkl"
    ipr_exp_path = "data-models/experinment-results/irp_exp.pkl"
    fairness_exp_path = "data-models/experinment-results/fairness_exp.pkl"
    
    if not os.path.exists(fairness_exp_path):
        fairness_df = pd.DataFrame()
        awrf, fair_list = fairness_evaluation(var_model_names)
        fairness_df['mean awrf'] = awrf['mean awrf']
        print(fairness_df)
        print(fair_list)
    
    # Perform relevance evaluation
    if not os.path.exists(relevance_exp_path):
                
        print('Creating Relevance data')
        relevance_df = relevance_evaluation(var_models, var_model_names, test_topics, qrels)
        f = open(relevance_exp_path,"wb")
        pickle.dump(relevance_df, f)
        f.close()
        print('Relevance evaluations saved')
    else:
        relevance_df = pd.read_pickle(relevance_exp_path)
        print(relevance_df)




    """
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
    """

    # Retrieve the feature importance of all models
    #get_feature_importance(lgb_list)
    


if __name__ == "__main__":
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64/"
    if pt.is_windows():
        print('Operating System not Supported')
    else:
        print('Operating system supported!')
        if not pt.started():
            pt.init()
            print('Pyterrier started')
        else: 
            print('PyTerrier is alredy running')
        main()



