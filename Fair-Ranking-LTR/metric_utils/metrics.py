import pandas as pd
import sapiezynski_metric as sp
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import Counter




import pandas as pd
import numpy as np

import sapiezynski_metric as sp


import metric_utils.groupinfo as gi
import metric_utils.position as pos



class metric_analysis:
    
    ranked_list = None
    test_rates = None
    group = None
    arg = None
    arg_val = None
    
    def __init__(self, ranked_list, group, original_data=None, AWRF=True):
        
        self.ranked_list = ranked_list
        self.original_data = original_data
        self.group = group
        self.AWRF=AWRF
        

    def run_awrf(self, ranked_list, pweight):
        
        """
        Measure fairness using AWRF metric.
        Args:
            ranked_list(panda.DataFrame): truncated ranked lists of retrieved results.
            pweight(position object): user browsing model to measure position weight.
        Return:
            pandas.Series: fairness score of ranked lists for each algorithms
        """
        
        weight_vector = pweight(ranked_list)
        user_awrf = pd.Series({'AWRF': sp.awrf(ranked_list, self.group, weight_vector)})
        return user_awrf
    
    def run_stochastic_metric(self, ranked_list, pweight):
        
        """
        Measure fairness using single ranking metrics.
        Args:
            ranked_list(panda.DataFrame): truncated ranked lists of retrieved results.
            pweight(position object): user browsing model to measure position weight.
        Return:
            pandas.Series: fairness scores of ranked lists for each algorithms
        """
        
        result = pd.Series()
        if pweight == 'default':
            if self.IAA == True:
                result = self.run_IAA(ranked_list)
            if self.EE == True:
                result = result.append(self.run_EE(ranked_list))
            if self.DRR == True:
                result = result.append(self.run_dp_eur_rur(ranked_list))
            return result
        if self.IAA == True:
            result = self.run_IAA(ranked_list, pweight)
        if self.EE == True:
            result = result.append(self.run_EE(ranked_list, pweight))
        if self.DRR == True:
            result = result.append(self.run_dp_eur_rur(ranked_list, pweight))
        
        return result




    def run_default_setting(self, listsize):
        
        truncated = self.ranked_list[self.ranked_list['rank']<=listsize]
        #truncated = self.ranked_list
        
        stochastic_metrics = truncated.groupby(['docid', 'qid']).progress_apply(self.run_stochastic_metric, pweight='default')
        stochastic_metrics_mean = stochastic_metrics.groupby('docid').mean()
        stochastic_metrics_score = stochastic_metrics_mean.reset_index().melt(id_vars=['docid'], var_name='Metric')
        
        if self.AWRF == False:
            user_awrf_fair = truncated.groupby(['qid']).progress_apply(self.run_awrf_fair)
            user_agg = user_awrf_fair.groupby(['docid']).mean()
            AWRF_FAIR = user_agg.reset_index().melt(id_vars=['docid'], var_name='Metric')
            final_metric = pd.concat([AWRF_FAIR, stochastic_metrics_score], ignore_index=True)
        else:
            final_metric = stochastic_metrics_score
            
        final_metric['ranked_size'] = listsize
        
        return final_metric



def get_feature_list():
      return ["pred_qual",'qual_cat', 'page_subcont_regions', 
              'source_subcont_regions', 'years','creation_date_category',
              'relative_pageviews_category','first_letter_category']



def skewed_metrics(ranked_df, feature_list):
        my_path = '~/Desktop/Fair-Ranking-Repo/fair-learning-to-rank-strategy/graphs/'
        for feature in feature_list:
            print(feature)
            if feature == 'page_subcont_regions':
                counts = ranked_df[feature].value_counts(dropna=False)
                x_pos = np.arange(len(counts))
                plt.figure(feature)
                plt.bar(x_pos, counts)
                plt.rcParams['figure.figsize'] = (20,3)
                plt.xticks(x_pos, counts.index, rotation=90)
                plt.title('Page subcontintent counts')
                plt.ylabel('Frequency of articles')
                plt.xlabel('Subcontinental regions') 
                plt.subplots_adjust(bottom=0.4, top=0.99)
                plt.savefig(feature + '_hist.png')

            elif feature == 'source_subcont_regions':
                region_list=[]
                for region in ranked_df[feature]:
                    if region == None:
                         region_list.append('UNK')
                    else:
                        for r in region:       
                            region_list.append(r)           
                counter = Counter(region_list)
                x_pos = np.arange(len(counter))
                plt.figure(feature)
                plt.bar(x_pos, counter.values())
                plt.xticks(x_pos, counter.keys(), rotation=90)
                plt.title('Source subcontintent counts')
                plt.ylabel('Frequency of articles')
                plt.xlabel('Subcontinental regions')
                plt.subplots_adjust(bottom=0.4, top=0.99)
                plt.savefig(feature + '_hist.png')
                
            elif feature=='qual_cat':
                counts = ranked_df[feature].value_counts(dropna=False)
                x_pos = np.arange(len(counts))
                plt.figure(feature)
                plt.bar(x_pos, counts)
                plt.rcParams['figure.figsize'] = (20,5)
                plt.xticks(x_pos, counts.index, rotation=90)
                plt.title('Categorical quality')
                plt.ylabel('Frequency')
                plt.xlabel('Quality articles') 
                plt.subplots_adjust(bottom=0.4, top=0.99)
                plt.savefig(feature + '.png')
                 
            elif feature == 'pred_qual':   
                print(ranked_df[feature].mean())
                plt.figure(feature)
                plt.hist(ranked_df[feature])
                plt.title('Quality histrgram')
                plt.ylabel('Frequency')
                plt.xlabel('Quality scores') 
                plt.savefig(feature + '_hist.png')

            elif feature == 'creation_date_category' or feature=='first_letter_category' or feature=='relative_pageviews_category':
                counts = ranked_df[feature].value_counts().to_dict()
                print(counts)
            
            
            
            
            
            

