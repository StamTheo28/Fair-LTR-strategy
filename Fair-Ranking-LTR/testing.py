import pandas as pd
from metric_utils.metrics import skewed_metrics, skewness
from metric_utils.metrics import get_feature_list
from collections import Counter

model_path = "data-models/Ranked-Results/BM25.pkl"
pop_stats_path = "data-models/Data/features_stats.pkl"
df_path = "data-models/Data/computed_df.pkl"

comp = pd.read_pickle(df_path)
pop_stats = pd.read_pickle(pop_stats_path)

#print(comp)

res =pd.read_pickle(model_path)


# Create Global distribution
global_dist = {}
for key in pop_stats.keys():
    if key != 'docid':
        global_dist[key] = Counter(pop_stats[key])
res_df = pd.DataFrame(res)
print(skewness(res_df, get_feature_list()))
#skewed_metrics(res_df, global_dist)
#print(global_dist)



print()




# Create local distribution

#print(res)
#print(skewed_metrics(res))

