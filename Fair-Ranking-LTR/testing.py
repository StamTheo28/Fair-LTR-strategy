import pandas as pd
from metric_utils.metrics import get_feature_list

model_path = "data-models/Ranked-Results/BM25"+'.pkl'
res = pd.DataFrame(pd.read_pickle(model_path))
print(res)

for f in get_feature_list():
    if f in ['source_subcont_regions','page_subcont_regions']:
        print(res[f].str.join('|').str.get_dummies())
    else:

        #print(res[f].values.tolist())
        #print(res[f].values)
        #print(res[f])
        print(res[f].str.join('').str.get_dummies())
   
