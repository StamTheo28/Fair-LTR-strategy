import pandas as pd


model_path = "data-models/Ranked-Results/BM25.pkl"
pop_stats_path = "data-models/Data/features_stats.pkl"
df_path = "data-models/Data/computed_df.pkl"

comp = pd.read_pickle(df_path)
pop_stats = pd.read_pickle(pop_stats_path)

#print(comp)
doc = comp[['source_subcont_regions','page_subcont_regions']][comp.docid==14099]
print(doc['source_subcont_regions'].values[0], 'This')






"""

print(pop_stats.keys())
boost_scores = {}
f1 = 'page_subcont_regions'
f2 = 'source_subcont_regions'
for cat in pop_stats[f1].keys():
    score_1 = pop_stats[f1][cat]
    score_2 = pop_stats[f2][cat]
    if score_1 < 0.05 and score_2 <0.05:
        boost_scores[cat] = abs(score_1-score_2)
print(boost_scores)
    
"""



# Create local distribution

#print(res)
#print(skewed_metrics(res))

