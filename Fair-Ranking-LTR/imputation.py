from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
from variation_7 import get_var_7_feature_list
import os
import pickle
from sklearn.preprocessing import LabelEncoder
# Encode gender column as integers
def replace(row, null_list):
    if row in null_list  or pd.isna(row):
        return np.nan
    else:
        return row

def get_imputed_data():
    data_path = "data-models/Data/computed_df.pkl"
    imputed_df_path = "data-models/Data/imputed_df.pkl"
    pop_stats_path = "data-models/Data/pop_stats_var_7.pkl"

    if not os.path.exists(imputed_df_path):

        df = pd.read_pickle(data_path)[['docid']+get_var_7_feature_list()]
        null_list =[None, 'UNKNOWN', "N/A",float('nan'),'UNK','Unknown','nan']

        # Replace all unkown values with np.nan
        df = df[['docid']+get_var_7_feature_list()].applymap(lambda row: replace(row, null_list))

        # Encode gender column as integers
        gender_encoder = LabelEncoder()
        df['gender'] = gender_encoder.fit_transform(df['gender'])
        # Encode occupations column as integers
        occupation_encoder = LabelEncoder()
        df['occupations'] = occupation_encoder.fit_transform(df['occupations'])
        # Encode qual_cat column as integers
        qual_cat_encoder = LabelEncoder()
        df['qual_cat'] = qual_cat_encoder.fit_transform(df['qual_cat'])
        # Encode relative_pageviews_category column as integers
        relative_pageviews_category_encoder = LabelEncoder()
        df['relative_pageviews_category'] = relative_pageviews_category_encoder.fit_transform(df['relative_pageviews_category'])


        # Split the dataset into two subsets: one with missing values, and one without
        df_missing = df[(df['gender'] == 3)| (df['occupations'] == 32 )]

        df_not_missing = df[~((df['gender']==3) | (df['occupations'] == 32))]
        # Create a decision tree regressor to predict the missing values
        regressor = DecisionTreeRegressor()
        # Fit the regressor on the non-missing data
        X_train = df_not_missing[['relative_pageviews_category', 'qual_cat']]
        y_train = df_not_missing[['gender', 'occupations']]

        regressor.fit(X_train, y_train)
        # Use the regressor to predict the missing values
        X_test = df_missing[['relative_pageviews_category', 'qual_cat']]
        
        imputed_values = regressor.predict(X_test)
        # Replace the missing values with the imputed values
        df_missing[['gender', 'occupations']] = imputed_values.astype(int)
        # Combine the two subsets back into a single DataFrame
        df_imputed = pd.concat([df_missing, df_not_missing])
        
        # Convert all integer values back to there string values
        df_imputed['gender'] = gender_encoder.inverse_transform(df_imputed['gender'])
        df_imputed['occupations'] = occupation_encoder.inverse_transform(df_imputed['occupations'])
        df_imputed['qual_cat'] = qual_cat_encoder.inverse_transform(df_imputed['qual_cat'])
        df_imputed['relative_pageviews_category'] = relative_pageviews_category_encoder.inverse_transform(df_imputed['relative_pageviews_category'])
        df_imputed = pd.merge(df['docid'], df_imputed)
        
        f = open(imputed_df_path,"wb")
        # write the python object (dict) to pickle file
        pickle.dump(df_imputed,f)
        # close file
        f.close()
        print("Imputed dataframe has been createed and saved.")
    else:
        df_imputed = pd.read_pickle(imputed_df_path)
        print('Imputed data loaded')

    if not os.path.exists(pop_stats_path):
        features_dict = {}
        for feature in df_imputed.columns:
            features_dict[feature] = df_imputed[feature].value_counts().to_dict()

        for key, values in features_dict.items():
            total = sum(values.values())
            for category, value in values.items():
                features_dict[key][category] = value/total
        f = open(pop_stats_path,"wb")
        # write the python object (dict) to pickle file
        pickle.dump(features_dict,f)

        # close file
        f.close()

    return df_imputed

get_imputed_data()